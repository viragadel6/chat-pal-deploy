# Chat Pal

Autonomous Hungarian-language agent application. The backend is a Python service that
drives an LLM through a cognitive agent loop (dual-process planning, reflection, skill
learning, consolidation), executes tools in a disposable Linux VM, controls a live cloud
browser and streams every step to the browser over SSE.

The repository contains the Python backend in `backend/`, the static frontend in `public/`
and the database schema in `supabase/migrations/`.

## Architecture

```
public/chat.html ──HTTP/SSE──▶ backend/app (FastAPI)
                                   ├── app/routers        HTTP surface
                                   ├── app/services       durable job orchestration
                                   ├── app/agent          agent loop, tools, provider,
                                   │                      sandbox, cognitive layer,
                                   │                      subagents, cloud browser, CDP
                                   ├── app/database       async PostgREST client
                                   └── app/utils          security, rate limit, SSE, validation
```

- **Frontend** – `public/chat.html`, `public/chat-work.js` and `public/chat-work.css` are
  served statically by the backend; `/` redirects to `/chat.html`.
- **Model provider** – OrcaRouter chat completions with SSE streaming, tool calls and
  reasoning deltas.
- **Tool runtime** – InstaVM provides a disposable Linux VM with shell, Python, Nix,
  file and port-forwarding tools; Browser Use Cloud provides the proxied live browser.
- **Learning layer** – reflections, distilled policies, synthesized skill DSLs and the
  operating manual are persisted in the `agent_*` tables and reused in later runs.
- **Durable jobs** – a chat turn becomes a `chat_jobs` row that a background worker
  executes with a 90 s lease and 20 s heartbeat, so the answer survives a browser reload.

## Requirements

- Python 3.11 or newer
- A Supabase project with the schema from `supabase/migrations/` applied
- `ORCAROUTER_API_KEY` for the model, `INSTAVM_API_KEY` for the sandbox
- Optional: `BROWSER_USE_API_KEY`, `GEMINI_API_KEY`, `FLYMYAI_API_KEY`

## Development

```sh
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`CHAT_DEPLOYMENT_MAX_REQUEST_MS` must be at least `21600000 + 120000 + 24000` milliseconds,
otherwise long-running tasks are rejected with `DEPLOYMENT_UNSUPPORTED`.

## Tests

```sh
cd backend
.venv/bin/python -m pytest -q
.venv/bin/ruff check app tests
```

The suite is hermetic: `tests/local_stack.py` starts a local OrcaRouter SSE server and a
local InstaVM bash server, `tests/fakes.py` provides an in-memory stateful database.

## Database

```sh
supabase db push
```

The migrations create the `kick_chat_worker` RPC, the `chat_browser_sessions` table and
the browser control handoff columns used by the Python backend.

## Docker

```sh
docker build -f backend/Dockerfile -t chat-pal-backend .
docker run --rm --env-file backend/.env -p 8000:8000 chat-pal-backend
```

The build context is the repository root so that the static frontend is baked into the
image as well.

## Documentation

`backend/README.md` lists every endpoint, the environment variables and the deployment
steps.
