# AI Tutor backend

Greenfield, education-only FastAPI backend using Groq for tutoring and Whisper
transcription, SentenceTransformers plus FAISS for RAG, Edge TTS for speech, and
Silero VAD as an optional speech gate. The backend emits semantic whiteboard data;
it contains no frontend or notebook UI code.

## Architecture

- `app/api`: thin HTTP routes and error translation
- `app/core`: environment configuration and logging
- `app/rag`: PDF/TXT/MD loading, page-aware chunking, embeddings, persisted FAISS,
  metadata, duplicate detection, course filtering, and cached retrieval
- `app/llm`: one reusable Groq service, centralized prompts, academic classifier,
  structured planner, and tutor generation
- `app/teaching`: transport-independent session store and step-based state machine,
  follow-ups, quizzes, and interruption adaptation
- `app/speech`: replaceable Groq Whisper STT, Edge TTS, and lazy Silero VAD
- `app/schemas`: Pydantic API/domain contracts

FAISS uses normalized `all-MiniLM-L6-v2` vectors and inner product (cosine
similarity). An accepted result reports `source_type=course_material` with exact
document/page/chunk metadata. With no sufficiently relevant result it reports
`general_knowledge`; the tutor prompt explicitly prevents fabricated file claims.

## Setup

Python 3.11 is recommended. From `AI Tutor/backend`:

Browser microphone recordings are normalized to 16 kHz WAV before VAD and
transcription. Scanned pages and images use local OCR, while PDFs and slides are
rendered so their original visuals can be reused on the board. On macOS install
the system tools with `brew install ffmpeg tesseract poppler` and install
LibreOffice so `soffice` can render PPTX slides.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Optional configuration:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Tutoring, planning, guard, quiz model |
| `GROQ_WHISPER_MODEL` | `whisper-large-v3` | Speech transcription model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `RETRIEVAL_TOP_K` | `6` | Maximum grounded chunks |
| `RETRIEVAL_THRESHOLD` | `0.25` | Minimum cosine similarity |
| `TTS_VOICE_ENGLISH` | `en-US-AriaNeural` | English voice |
| `TTS_VOICE_URDU` | `ur-PK-AsadNeural` | Urdu/Roman Urdu voice |

Start the API:

```bash
uvicorn app.main:app --reload
```

OpenAPI is at `http://127.0.0.1:8000/docs` and health is at `/api/health`.

For the end-to-end student test flow, run the small Vite interface in
`../frontend` with `npm install && npm run dev -- --port 5173`, then open
`http://127.0.0.1:5173`. It automatically creates the internal course and session;
students never enter or see their identifiers.

## Example flow

Create a course:

```bash
curl -X POST http://127.0.0.1:8000/api/courses \
  -H 'Content-Type: application/json' \
  -d '{"name":"Physics 101","subject":"Physics"}'
```

Upload course material (replace `COURSE_ID`):

```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F course_id=COURSE_ID -F document_type=lecture \
  -F file=@lecture.pdf
```

Create and start a lesson:

```bash
curl -X POST http://127.0.0.1:8000/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"course_id":"COURSE_ID","topic":"Newtons second law","student_level":"secondary","language":"English"}'

curl -X POST http://127.0.0.1:8000/api/teaching/start \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION_ID"}'

curl -X POST http://127.0.0.1:8000/api/teaching/SESSION_ID/respond \
  -H 'Content-Type: application/json' \
  -d '{"content":"Wait, what exactly is mass?","is_interruption":true}'

curl -X POST http://127.0.0.1:8000/api/teaching/SESSION_ID/next
```

Chat, speech, follow-ups, and quiz:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain binary search","student_level":"college","language":"English"}'

curl -X POST http://127.0.0.1:8000/api/speech/transcribe -F file=@question.webm
curl -X POST http://127.0.0.1:8000/api/speech/synthesize \
  -F text='Force equals mass times acceleration.' -F language=English --output tutor.mp3
curl -X POST 'http://127.0.0.1:8000/api/teaching/SESSION_ID/followups?count=3'
curl -X POST 'http://127.0.0.1:8000/api/quiz/generate?session_id=SESSION_ID&count=5'
```

## Tests

External Groq calls and embedding downloads are replaced with fakes in tests:

```bash
pytest -q
```

The suite covers chunking, vector insertion and persistence, course filtering,
semantic academic guarding, session isolation, teaching/interruption transitions,
RAG fallback, and invalid upload types.

## Current limitations

- Sessions are process-local and disappear on restart; run one worker for this
  milestone. The repository interface is the intended replacement seam.
- Course/document catalogs are JSON files. FAISS writes are process-thread safe but
  not coordinated across multiple server processes.
- Text extraction does not OCR scanned PDFs.
- Groq's classifier is intentionally required for domain decisions; without a key,
  model-backed endpoints return 503 instead of weakening the guard to keywords.
- Browser microphone capture, audio cancellation, FFmpeg normalization, Silero VAD,
  and Groq Whisper transcription are implemented in the test UI. Audio is uploaded
  only after the student stops a recording.
- Edge TTS is a network service and returns a completed MP3 rather than streamed audio.

## Recommended real-time next step

Add a WebSocket transport around the existing `TeachingEngine`, without moving
business logic into the socket handler. Give each outbound text/audio event the
session `response_revision`. On incoming `speech_started`, the server increments the
revision and emits `cancel_audio`; the client immediately stops any older revision.
Then stream audio frames, gate them with Silero VAD, transcribe the completed utterance
with Groq Whisper, call `respond(..., is_interruption=True)`, and resume from the
engine's returned `next_action`. Later, replace completed-response Groq/TTS calls with
token/audio streaming behind the same service interfaces.
