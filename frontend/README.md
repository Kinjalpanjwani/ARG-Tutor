# AI Tutor test UI

Small Vite interface for exercising the core tutor pipeline. It automatically
creates and retains an internal course, uploads up to three PDFs, reports indexing
status, creates lesson sessions, and renders generated teaching steps.

```bash
npm install
npm run dev -- --port 5173
```

The backend must be running at `http://127.0.0.1:8000`. To use another backend,
copy `.env.example` to `.env` and change `VITE_API_BASE_URL`.

Internal course and session identifiers are never displayed or requested from the
student. Swagger remains available independently for backend development.
