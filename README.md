# AI Tutor

An AI tutoring system with course-material upload, RAG-assisted teaching,
whiteboard explanations, speech, interruptions, visuals, quizzes, and adaptive
lesson interaction.

## Project Structure

```text
backend/   FastAPI application, document ingestion, RAG, Groq tutoring, STT, and TTS
frontend/  Vite-based lesson test interface
```

## Backend Setup

Requirements include Python 3, Tesseract OCR, and Poppler (`pdftoppm`) for
scanned/image-based PDFs. LibreOffice is used when rendering PPTX slides.

```bash
cd backend

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GROQ_API_KEY to your own Groq API key.

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend API is available at `http://127.0.0.1:8000`. Swagger documentation
is available at `http://127.0.0.1:8000/docs`.

## Frontend Setup

In a separate terminal:

```bash
cd frontend
npm install
npm run dev -- --port 5173
```

Open `http://127.0.0.1:5173`.

The frontend automatically creates the internal course/session identifier. Students
can upload materials and start lessons without entering IDs manually.

## Environment

The backend reads configuration from `backend/.env`. Never commit this file.
The required secret is:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Additional defaults are documented in `backend/.env.example`.
