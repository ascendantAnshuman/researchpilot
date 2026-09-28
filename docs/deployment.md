# Deploying ResearchPilot

The recommended setup is:

| Part | Host | Why |
|---|---|---|
| Frontend (Next.js) | **Vercel** | Built for Next.js; free tier is enough. |
| Backend (FastAPI + agents) | **Railway** (recommended) or **Render** | Runs the backend `Dockerfile`, supports a persistent volume and long-lived streaming (SSE) responses. |

## What the backend needs from a host

Check these against the plan you choose. They are the usual reasons a deployment fails.

- **Docker support.** The image is built from `backend/Dockerfile`.
- **Memory: at least 1 GB, ideally 2 GB.** PyTorch, the MiniLM embedding model and ChromaDB together use roughly 0.8–1 GB of RAM. Free tiers with 512 MB will be killed for running out of memory.
- **A persistent volume mounted at `/data`.** The SQLite database, the ChromaDB index and cached PDFs live there. Without a volume, every redeploy wipes all accounts and reports.
- **Exactly one instance and one worker.** Research runs and their live event streams are kept in the backend process's memory, so do not scale to several replicas.
- **Long-running responses.** A research run streams progress for several minutes. Both Railway and Render support this; the backend already sends `Cache-Control: no-cache` and `X-Accel-Buffering: no` so proxies do not buffer the stream.

Pricing and plan limits change often, so check the current pricing pages of Railway, Render and Vercel before choosing.

---

## Step 0 – Put the project on GitHub

From the `researchpilot/` folder:

```powershell
git init
git add .
git commit -m "ResearchPilot: multi-agent research copilot"
git branch -M main
git remote add origin https://github.com/<your-username>/researchpilot.git
git push -u origin main
```

`.env` is listed in `.gitignore`, so your keys are **not** uploaded. Check with `git status` before committing.

## Step 1 – Backend on Railway (recommended)

1. Go to <https://railway.app> → **New Project** → **Deploy from GitHub repo** → pick your repository.
2. Open the new service → **Settings** → **Root Directory**: `backend`. Railway finds `backend/Dockerfile` automatically.
3. **Variables** tab – add:

   | Variable | Value |
   |---|---|
   | `ENVIRONMENT` | `production` |
   | `JWT_SECRET_KEY` | a long random string: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | `GEMINI_API_KEY` | your free Gemini key (or `ANTHROPIC_API_KEY` if you use a Claude model) |
   | `LLM_MODEL` | `gemini-3.8-flash` (free) or a Claude model such as `claude-sonnet-5` |
   | `LLM_REQUESTS_PER_MINUTE` | `8` on the Gemini free tier |
   | `SEMANTIC_SCHOLAR_API_KEY` | your key (optional but recommended) |
   | `DATA_DIR` | `/data` |
   | `CORS_ORIGINS` | your Vercel URL from Step 3, e.g. `https://researchpilot.vercel.app` (fill in after Step 3) |
   | `MAX_CONCURRENT_RUNS` | `1` (keeps memory and API costs predictable) |

4. **Volumes**: right-click the service → **Attach Volume** → mount path `/data`.
5. **Settings → Networking → Generate Domain**. You get a URL such as `https://researchpilot-production.up.railway.app`.
6. Wait for the deployment. The first build takes about 5–10 minutes (it downloads PyTorch and the embedding model). Then open `https://<your-domain>/api/health`. You should see `"status": "ok"` and `"database": "ok"`.

Railway sets the `PORT` variable itself; the container listens on it automatically.

## Step 2 (alternative) – Backend on Render

**Option A: Blueprint.** The repository contains `render.yaml`. In Render choose **New → Blueprint**, select the repository, and fill in the secret values it asks for. Review the plan and disk size before confirming.

**Option B: by hand.**

1. **New → Web Service** → connect the repository.
2. **Language/Runtime:** Docker. **Root Directory:** `backend`.
3. **Instance type:** one with **at least 2 GB RAM**. The smallest instances are too small; see "What the backend needs" above.
4. **Advanced → Add Disk:** mount path `/data`, size 1–5 GB.
5. **Health Check Path:** `/api/health`.
6. Add the same environment variables as in the Railway table above.
7. Deploy, then open `https://<your-service>.onrender.com/api/health`.

## Step 3 – Frontend on Vercel

1. Go to <https://vercel.com> → **Add New… → Project** → import the repository.
2. **Root Directory:** `frontend`. The framework (Next.js) is detected automatically.
3. **Environment Variables:** `NEXT_PUBLIC_API_URL` = your backend URL, **without** a trailing slash, e.g. `https://researchpilot-production.up.railway.app`.
4. **Deploy.** You get a URL such as `https://researchpilot.vercel.app`.

`NEXT_PUBLIC_API_URL` is compiled into the JavaScript at build time. If you change it later, **redeploy** the frontend.

## Step 4 – Connect the two

1. On the backend (Railway/Render), set `CORS_ORIGINS` to the exact Vercel URL: `https://…`, no trailing slash. Several URLs can be separated by commas, e.g. your production URL and a preview URL.
2. Redeploy the backend.
3. Open the Vercel URL, create your account, and run a research question.
4. **Protect your API budget.** Once your account exists, set `ALLOW_REGISTRATION=false` on the backend and redeploy. Otherwise anyone who finds the site can register and spend your Anthropic credit.

---

## Running everything with Docker on your own machine

```powershell
Copy-Item .env.example .env      # then fill in GEMINI_API_KEY (or ANTHROPIC_API_KEY) and JWT_SECRET_KEY
docker compose up --build
```

Open <http://localhost:3000>. Data is kept in the Docker volume `researchpilot-data`. Useful commands:

| Command | What it does |
|---|---|
| `docker compose logs -f backend` | Follow the backend logs (agent progress, errors). |
| `docker compose down` | Stop everything. Your data is kept. |
| `docker compose down -v` | Stop and **delete** all data (accounts, reports, index). |

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Browser console shows a **CORS** error | `CORS_ORIGINS` on the backend does not exactly match the frontend URL (scheme, domain, no trailing slash). |
| "Cannot reach the ResearchPilot server" | `NEXT_PUBLIC_API_URL` was missing or wrong **when the frontend was built**. Fix it and redeploy the frontend. |
| Backend restarts during a run / "out of memory" | The instance has too little RAM. Use at least 1–2 GB. |
| Accounts and reports disappear after a redeploy | No persistent volume is mounted at `/data`. |
| Live progress never appears but the report finishes | Something between the browser and the backend buffers responses. Check for extra proxies or CDNs in front of the backend. |
| "The language model is not configured on the server" (503) | The key for `LLM_MODEL`'s provider (`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`) is not set on the backend. |
| Runs fail with "temporarily unavailable" | Anthropic rate limit or outage. New API accounts have low limits; set `MAX_CONCURRENT_RUNS=1` and `LLM_MAX_CONCURRENCY=1`. |
| Few papers, no citation counts | Semantic Scholar's shared public quota is exhausted (HTTP 429). Add `SEMANTIC_SCHOLAR_API_KEY`. |
