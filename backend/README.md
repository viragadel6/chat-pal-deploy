# Chat Pal backend

A teljes Chat Pal alkalmazás Python backendje. Az HTTP/SSE szerződést teljes egészében
implementálja, így a repository `public/` mappájában lévő statikus frontend változtatás
nélkül használható.

## Követelmények

- Python 3.11 vagy újabb
- Supabase projekt (PostgREST + Auth), a `chat_conversations`, `chat_jobs`,
  `chat_job_events`, `chat_browser_sessions`, `agent_*` táblákkal és a
  `kick_chat_worker` RPC függvénnyel
- OrcaRouter API kulcs (`ORCAROUTER_API_KEY`)
- InstaVM API kulcs (`INSTAVM_API_KEY`) a sandbox eszközökhöz
- Opcionális: `BROWSER_USE_API_KEY`, `GEMINI_API_KEY`, `FLYMYAI_API_KEY`

## Telepítés

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

A `.env` fájlban minden titkos kulcsot és engedélyezési listát ki kell tölteni.
A `CHAT_DEPLOYMENT_MAX_REQUEST_MS` értékének legalább
`21600000 + 120000 + 24000` milliszekundumnak kell lennie, különben a futás
`DEPLOYMENT_UNSUPPORTED` hibával azonnal leáll.

## Futtatás

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

A szerver a `public/` mappát statikus fájlként szolgálja ki, a `/` útvonal a
`/chat.html` oldalra irányít, és a `GET /api/health` végpont állapotot ad.

## Tesztek

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check app tests
```

A tesztek hermetikusak: a `tests/local_stack.py` helyi OrcaRouter SSE és InstaVM bash
szervereket indít, a `tests/fakes.py` memóriában tartja az adatbázis állapotát.

## Végpontok

| Metódus | Útvonal | Leírás |
| --- | --- | --- |
| POST | `/api/chat` | Beszélgetés indítása, SSE folyam a válaszhoz |
| GET | `/api/chat/jobs/{job_id}/events` | Durable munka eseményeinek követése SSE folyammal |
| POST | `/api/chat/jobs/{job_id}/stop` | Futó munka leállítása |
| POST | `/api/chat/jobs/{job_id}/redirect` | Operátori irányítás átadása futó munkának |
| GET | `/api/conversations` | Beszélgetések és a legutóbbi munkák lekérése |
| POST | `/api/operator` | Operátori üzenet mentése egy feladathoz |
| GET | `/api/browser` | Böngészőnézet állapota, képkocka vagy munkamenetek |
| POST | `/api/browser` | Böngésző irányításának átvétele, visszaadása, bevitel |
| POST | `/api/images/edits` | Képszerkesztés a FlyMyAI szolgáltatással |
| POST | `/api/public/chat-worker` | Háttérmunka végrehajtása a munkakereső hívására |
| GET | `/api/health` | Élettartam állapota |

## Docker

A képfeldolgozás a repository gyökérkönyvtárából történik, hogy a statikus frontend is
a képbe kerüljön:

```bash
docker build -f backend/Dockerfile -t chat-pal-backend .
docker run --rm --env-file backend/.env -p 8000:8000 chat-pal-backend
```
