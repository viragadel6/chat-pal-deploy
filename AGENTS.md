<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Decisions
- tsconfig: noPropertyAccessFromIndexSignature and noUncheckedIndexedAccess disabled — imported chat.ts (982 lines) uses dot access on Record types throughout; relaxing two flags was chosen over rewriting 144 call sites.
- Chat frontend is a static public/chat.html (395KB) that / redirects to; chat backend is src/routes/api/chat.ts (OrcaRouter + InstaVM).
- Cognitive agent layer (dual-process planning, lifecycle phases, reflection, skill learning, distillation, playbook, operator messages) lives in src/lib/cognitive.server.ts and is loaded dynamically by the chat route; learned knowledge persists in Lovable Cloud agent_* tables accessed only server-side, keeping model-loop code and learning logic separate.
- Forward provider reasoning and cognitive model output through reasoning_content SSE deltas, keeping cognitive output separate from the final answer so the existing thinking panel can display every emitted fragment.
- Image requests use a multipart server route that keeps the external provider key server-side and forwards the provider response stream; the existing image dialog parses the provider's image results and requires a real reference image for edit models.
- Background chat jobs run in the app's own /api/public/chat-worker route, started by a database HTTP call (pg_net) so the work survives browser closure; a 90s heartbeat lease lets reconnecting viewers restart a crashed job.
- Live browser viewing uses direct CDP screenshot capture and input on a separate persistent canvas; database-backed browser control handoff pauses model tool steps across worker invocations before enabling human input.
- Browser CDP commands use native WebSocket in a server-only module; the static chat loads a separate viewer script and stylesheet so transcript rendering cannot recreate the canvas.
- Work progress is emitted as a separate durable progress event, never as a preliminary model answer, so reconnects preserve actual execution status.
- The production runtime is the Python service in backend/ (FastAPI + httpx + websockets) started by `python -m backend.main`; it serves the same routes, SSE event contract, size limits, provider retry behaviour and tool set as the TypeScript sources, so public/chat.html and public/chat-work.js are unchanged.
- backend/agent.py owns the model loop (streamed deltas, tool-call reassembly, bounded retries, 600 steps, tool budget), backend/tools.py owns sandbox/provider tools (InstaVM, subagents, cloud browser, file read/write with paging) and backend/cognitive.py owns planning, verification, reflection, skill synthesis and playbook consolidation.
- Durable execution stays database-backed: backend/jobs.py claims chat_jobs leases, appends chat_job_events, follows them for SSE replay, and dispatches through the kick_chat_worker RPC with an in-process fallback only when the dispatch call itself cannot reach the database.
- Context pressure is handled by a byte-budget compaction that preserves the objective turn and replaces collapsed history with a system marker, mirroring the TypeScript ContextBudget so long runs keep the same observable behaviour.
- Long shell and Python tool commands run as detached jobs in /tmp/chat-job-<id> with pid/exit files, one-second polling and process-group termination on cancel, so a command outliving its HTTP wait does not lose its result.
- drizzle/migrations/0003_python_backend_schema.sql adds the tables and grants the Python service needs and re-declares claim_chat_browser_step without changing its signature.
