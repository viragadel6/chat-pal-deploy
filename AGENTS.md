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
- The application backend is a Python service in `backend/`; the previous TanStack Start server in `src/` was removed once the Python implementation covered the same HTTP and SSE contract.
- The frontend stays a static `public/chat.html` (plus `public/chat-work.js` and `public/chat-work.css`) that the FastAPI app serves; `/` redirects to `/chat.html`.
- The cognitive agent layer (dual-process planning, lifecycle phases, reflection, skill learning, distillation, playbook, operator messages) lives in `backend/app/agent/cognitive.py` and is loaded from the runtime module; learned knowledge persists in the `agent_*` tables accessed only server-side, keeping model-loop code and learning logic separate.
- The agent loop lives in `backend/app/agent/runtime.py`; it owns deliberation, reflection, context compaction, browser control handoff, verification and consolidation, and it emits the durable `progress` / `delta` / `agent_event` / `error` / `terminal` / `done` event sequence.
- Provider reasoning and cognitive model output travel through `reasoning_content` SSE deltas, keeping cognitive output separate from the final answer so the thinking panel can display every emitted fragment.
- Image requests use a multipart route that keeps the external provider key server-side and forwards the provider response; the existing image dialog parses the provider results and requires a real reference image for edit models.
- Background chat jobs run in the app's own `/api/public/chat-worker` route, started by a database HTTP call (`pg_net` through the `kick_chat_worker` RPC) so the work survives browser closure; a 90 s heartbeat lease lets reconnecting viewers restart a crashed job.
- Live browser viewing uses direct CDP screenshot capture and input on a separate persistent canvas; database-backed browser control handoff pauses model tool steps across worker invocations before enabling human input.
- Browser CDP commands use a native WebSocket client in `backend/app/agent/browser_cdp.py`; the static chat loads a separate viewer script and stylesheet so transcript rendering cannot recreate the canvas.
- Work progress is emitted as a separate durable progress event, never as a preliminary model answer, so reconnects preserve actual execution status.
- The TypeScript toolchain (bun, vite, drizzle, eslint, prettier) was removed together with the server sources; the database schema now lives in `supabase/migrations/` as plain SQL.
