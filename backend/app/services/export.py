"""Report export: Markdown and PDF (ReportLab).

The PDF contains the executive summary, themed sections, the comparison table,
a verification table (claim, verdict, evidence) and the reference list.
ReportLab's built-in fonts only cover Western European characters, so a Unicode
TrueType font (DejaVu Sans on Linux/Docker, Arial on Windows/macOS) is used when
available; otherwise text is transliterated to the built-in Helvetica.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.agents.critic import count_labels
from app.schemas import ResearchReport

logger = logging.getLogger(__name__)

FONT_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("DejaVuSans", "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("Arial", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("Arial", "/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
)
LABEL_COLOURS = {"supported": "#15803d", "partial": "#b45309", "unsupported": "#b91c1c"}
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PdfFonts:
    """Font names to use, and whether they cover Unicode."""

    regular: str
    bold: str
    unicode: bool


@lru_cache(maxsize=1)
def pdf_fonts() -> PdfFonts:
    """Register the first available Unicode TrueType font, else fall back to Helvetica."""
    for family, regular_path, bold_path in FONT_CANDIDATES:
        if Path(regular_path).is_file() and Path(bold_path).is_file():
            try:
                pdfmetrics.registerFont(TTFont(family, regular_path))
                pdfmetrics.registerFont(TTFont(f"{family}-Bold", bold_path))
                pdfmetrics.registerFontFamily(family, normal=family, bold=f"{family}-Bold")
                return PdfFonts(family, f"{family}-Bold", unicode=True)
            except Exception as error:  # A broken font file must not break exports.
                logger.warning("Could not load font %s: %s", regular_path, error)
    logger.warning("No Unicode TTF font found; PDF export will transliterate non-Latin characters")
    return PdfFonts("Helvetica", "Helvetica-Bold", unicode=False)


def export_filename(report: ResearchReport, extension: str) -> str:
    """A safe download file name such as ``hallucination-in-rag.pdf``."""
    slug = _SLUG_RE.sub("-", report.title.lower()).strip("-")[:60].rstrip("-") or "research-report"
    return f"{slug}.{extension}"


def report_to_markdown(report: ResearchReport) -> bytes:
    """The report's Markdown as UTF-8 bytes."""
    return report.markdown.encode("utf-8")


def to_latin1(value: str) -> str:
    """Keep Latin-1 characters (é, ü, ñ...), strip accents from others (ő -> o), else use '?'."""
    characters = []
    for character in unicodedata.normalize("NFKC", value):
        try:
            character.encode("latin-1")
            characters.append(character)
        except UnicodeEncodeError:
            base = unicodedata.normalize("NFKD", character).encode("latin-1", "ignore").decode("latin-1")
            characters.append(base or "?")
    return "".join(characters)


def _text(value: str, fonts: PdfFonts) -> str:
    """Escape text for ReportLab paragraphs (and transliterate if the font lacks Unicode)."""
    return escape(value if fonts.unicode else to_latin1(value))


def _styles(fonts: PdfFonts) -> dict[str, ParagraphStyle]:
    """Paragraph styles for the PDF."""
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName=fonts.regular, fontSize=10, leading=14,
                          alignment=TA_LEFT, spaceAfter=6)
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName=fonts.bold, fontSize=18, leading=22),
        "question": ParagraphStyle("Question", parent=body, textColor=colors.HexColor("#475569"), fontSize=10.5),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName=fonts.bold, fontSize=13, spaceBefore=10,
                             spaceAfter=6),
        "body": body,
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8, leading=10, spaceAfter=0),
        "cell_bold": ParagraphStyle("CellBold", parent=body, fontName=fonts.bold, fontSize=8, leading=10,
                                    spaceAfter=0),
        "reference": ParagraphStyle("Reference", parent=body, fontSize=9, leading=12, leftIndent=14,
                                    firstLineIndent=-14),
    }


def _table(rows: list[list[Paragraph]], widths: list[float]) -> Table:
    """A striped table with a bold header row that repeats on every page."""
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table


def report_to_pdf(report: ResearchReport) -> bytes:
    """Render the report as a PDF document and return its bytes."""
    fonts = pdf_fonts()
    styles = _styles(fonts)
    width = A4[0] - 4 * cm

    def paragraph(value: str, style: str = "body") -> Paragraph:
        return Paragraph(_text(value, fonts), styles[style])

    story: list[object] = [
        paragraph(report.title, "title"),
        paragraph(f"Research question: {report.question}", "question"),
        Spacer(1, 6),
        paragraph("Executive summary", "h2"),
        paragraph(report.executive_summary),
    ]
    for section in report.sections:
        story.append(paragraph(section.heading, "h2"))
        story.extend(paragraph(text) for text in section.paragraphs)

    if report.comparison_table:
        story.append(paragraph("Comparison of key papers", "h2"))
        header = [paragraph(h, "cell_bold") for h in ("Ref", "Paper", "Year", "Method", "Dataset", "Key result")]
        rows = [header] + [
            [paragraph(f"[{row.reference}]", "cell"), paragraph(row.title, "cell"), paragraph(str(row.year or "n.d."), "cell"),
             paragraph(row.method, "cell"), paragraph(row.dataset, "cell"), paragraph(row.key_result, "cell")]
            for row in report.comparison_table
        ]
        story.append(_table(rows, [w * width for w in (0.07, 0.25, 0.07, 0.21, 0.17, 0.23)]))

    if report.verification:
        counts = count_labels(report.verification)
        story.append(paragraph("Verification", "h2"))
        story.append(
            paragraph(
                f"{len(report.verification)} claims were checked against the source passages: "
                f"{counts['supported']} supported, {counts['partial']} partially supported, "
                f"{counts['unsupported']} unsupported."
            )
        )
        header = [paragraph(h, "cell_bold") for h in ("Claim", "Verdict", "Evidence")]
        rows = [header]
        for claim in report.verification:
            verdict = f'<font color="{LABEL_COLOURS[claim.label]}"><b>{claim.label.capitalize()}</b></font>'
            if claim.action != "kept":
                verdict += f"<br/>({claim.action})"
            evidence = claim.evidence[0] if claim.evidence else None
            evidence_text = f'"{evidence.text}" - {evidence.title}, {evidence.section}' if evidence else "No supporting passage"
            rows.append([paragraph(claim.text, "cell"), Paragraph(verdict, styles["cell"]), paragraph(evidence_text, "cell")])
        story.append(_table(rows, [w * width for w in (0.40, 0.14, 0.46)]))

    if report.references:
        story.append(paragraph("References", "h2"))
        for reference in report.references:
            authors = ", ".join(reference.authors[:3]) + (" et al." if len(reference.authors) > 3 else "")
            venue = f" {reference.venue}." if reference.venue else ""
            link = f" {reference.url}" if reference.url else ""
            story.append(
                paragraph(
                    f"[{reference.number}] {authors or 'Unknown authors'} ({reference.year or 'n.d.'}). "
                    f"{reference.title}.{venue}{link}",
                    "reference",
                )
            )

    def footer(canvas: Canvas, document: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont(fonts.regular, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(2 * cm, 1.2 * cm, "Generated by ResearchPilot")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {document.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=report.title,
        author="ResearchPilot",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
