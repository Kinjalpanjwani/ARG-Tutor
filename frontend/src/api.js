const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";
const COURSE_STORAGE_KEY = "ai-tutor-test-course";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : null;
  if (!response.ok) {
    throw new Error(data?.detail || `Request failed (${response.status})`);
  }
  return data;
}

export async function checkHealth() {
  return request("/health");
}

export async function ensureCourse() {
  const existing = localStorage.getItem(COURSE_STORAGE_KEY);
  if (existing) {
    try {
      await request(`/courses/${existing}`);
      return existing;
    } catch {
      localStorage.removeItem(COURSE_STORAGE_KEY);
    }
  }
  const course = await request("/courses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "Test lesson workspace", subject: "Uploaded course material" }),
  });
  localStorage.setItem(COURSE_STORAGE_KEY, course.course_id);
  return course.course_id;
}

export function uploadMaterial(file, courseId, onPhase) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("course_id", courseId);
    form.append("document_type", "course_document");
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/documents/upload`);
    xhr.upload.addEventListener("loadstart", () => onPhase("uploading"));
    xhr.upload.addEventListener("load", () => onPhase("processing"));
    xhr.addEventListener("load", () => {
      let payload;
      try { payload = JSON.parse(xhr.responseText); } catch { payload = null; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
      else reject(new Error(payload?.detail || `Upload failed (${xhr.status})`));
    });
    xhr.addEventListener("error", () => reject(new Error("Cannot reach the backend")));
    xhr.send(form);
  });
}

export function removeMaterial(documentId, courseId) {
  return request(`/documents/${documentId}?course_id=${encodeURIComponent(courseId)}`, { method: "DELETE" });
}

export async function createLesson(courseId, topic) {
  return request("/lessons/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ course_id: courseId, topic, language: "English" }),
  });
}

export function nextLessonSection(sessionId) {
  return request(`/lessons/${sessionId}/next-section`, { method: "POST" });
}

export async function previewRetrieval(courseId, query) {
  return request("/documents/retrieve-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ course_id: courseId, query }),
  });
}

export async function interruptLesson(sessionId, question, currentChunkIndex) {
  return request(`/lessons/${sessionId}/interrupt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, current_chunk_index: currentChunkIndex }),
  });
}

export async function synthesizeSpeech(text, language = "English") {
  const form = new FormData();
  form.append("text", text);
  form.append("language", language);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 35000);
  let response;
  try {
    response = await fetch(`${API_BASE}/speech/synthesize`, {
      method: "POST", body: form, signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.detail || "Speech synthesis failed");
  }
  return response.blob();
}

export async function transcribeSpeech(blob) {
  const form = new FormData();
  const extension = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
  form.append("file", blob, `question.${extension}`);
  return request("/speech/transcribe", { method: "POST", body: form });
}

export function generateQuiz(sessionId, count) {
  return request(`/lessons/${sessionId}/quiz?count=${count}`, { method: "POST" });
}

export function gradeQuiz(sessionId, question, answer) {
  return request(`/lessons/${sessionId}/quiz/grade`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, answer }),
  });
}
