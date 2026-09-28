"""ResearchPilot evaluation: benchmark questions, LLM-as-judge metrics and the Critic ablation.

Run from the project root with the backend's virtual environment, e.g. on Windows::

    .\\backend\\.venv\\Scripts\\python -m eval.run_eval --limit 2 --yes

Importing this package puts ``backend/`` on ``sys.path`` so the ``app`` package
(the pipeline itself) can be imported.
"""

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
