"use client";

import { Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState, type FormEvent, type KeyboardEvent } from "react";

import { useAuth } from "@/components/AuthProvider";
import { useHistory } from "@/components/HistoryProvider";
import { Alert, Button } from "@/components/ui";

const MIN_LENGTH = 10;
const MAX_LENGTH = 1000;
const PAPER_COUNTS = [5, 8, 10, 15] as const;

export const EXAMPLE_QUESTIONS = [
  "What are recent methods for reducing hallucination in RAG systems?",
  "How do parameter-efficient fine-tuning methods like LoRA compare to full fine-tuning?",
  "What techniques improve the robustness of vision transformers to distribution shift?",
  "How are large language models evaluated for code generation?",
];

/** The question box on the dashboard: starts a research run and opens its live view. */
export function NewResearchForm() {
  const { api } = useAuth();
  const { refresh } = useHistory();
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [useCritic, setUseCritic] = useState(true);
  const [topN, setTopN] = useState<number>(10);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const trimmed = question.trim();

  async function submit() {
    if (trimmed.length < MIN_LENGTH) {
      setError(`Please write a complete research question (at least ${MIN_LENGTH} characters).`);
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const started = await api.startResearch({ question: trimmed, enable_critic: useCritic, top_n: topN });
      void refresh();
      router.push(`/research/${started.session.id}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start the research run");
      setSubmitting(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      void submit();
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="rounded-2xl border border-slate-300 bg-white shadow-sm focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-500/20 dark:border-slate-700 dark:bg-slate-900">
        <label htmlFor="question" className="sr-only">
          Research question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value.slice(0, MAX_LENGTH))}
          onKeyDown={handleKeyDown}
          rows={4}
          placeholder="e.g. What are recent methods for reducing hallucination in RAG systems?"
          className="block w-full resize-none rounded-t-2xl bg-transparent px-4 py-3 text-base text-slate-900 placeholder:text-slate-400 focus:outline-none dark:text-slate-100"
        />
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 px-4 py-2.5 dark:border-slate-800">
          <div className="flex flex-wrap items-center gap-4 text-sm text-slate-600 dark:text-slate-300">
            <label className="inline-flex cursor-pointer items-center gap-2" title="The Critic agent checks every claim against the sources">
              <input
                type="checkbox"
                checked={useCritic}
                onChange={(event) => setUseCritic(event.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Fact-check with the Critic
            </label>
            <label className="inline-flex items-center gap-2">
              Papers
              <select
                value={topN}
                onChange={(event) => setTopN(Number(event.target.value))}
                className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-700 dark:bg-slate-900"
              >
                {PAPER_COUNTS.map((count) => (
                  <option key={count} value={count}>
                    {count}
                  </option>
                ))}
              </select>
            </label>
            <span className="text-xs text-slate-400">
              {trimmed.length}/{MAX_LENGTH}
            </span>
          </div>
          <Button type="submit" loading={submitting} disabled={trimmed.length < MIN_LENGTH} icon={<Sparkles className="h-4 w-4" aria-hidden />}>
            Start research
          </Button>
        </div>
      </div>
      {error && <Alert tone="error">{error}</Alert>}
      <div className="flex flex-wrap gap-2">
        {EXAMPLE_QUESTIONS.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => setQuestion(example)}
            className="rounded-full border border-slate-200 px-3 py-1 text-left text-xs text-slate-600 hover:border-brand-400 hover:text-brand-700 dark:border-slate-700 dark:text-slate-300 dark:hover:border-brand-500 dark:hover:text-brand-300"
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}
