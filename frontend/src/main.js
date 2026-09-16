import "./styles.css";
import {
  checkHealth, createLesson, ensureCourse, interruptLesson, previewRetrieval,
  synthesizeSpeech, transcribeSpeech, uploadMaterial, removeMaterial,
  generateQuiz, gradeQuiz,
  nextLessonSection,
} from "./api.js";

const $ = (selector) => document.querySelector(selector);
const elements = {
  signal: $("#connectionSignal"), connection: $("#connectionText"), dropZone: $("#dropZone"),
  fileInput: $("#fileInput"), fileList: $("#fileList"), topic: $("#topicInput"),
  start: $("#startButton"), error: $("#formError"), empty: $("#emptyBoard"),
  content: $("#lessonContent"), lessonState: $("#lessonState"), backend: $("#debugBackend"),
  pdfs: $("#debugPdfs"), chunks: $("#debugChunks"), retrieval: $("#debugRetrieval"),
  generation: $("#debugGeneration"), pagination: $("#boardPagination"),
  previous: $("#previousPage"), next: $("#nextPage"), pageLabel: $("#pageLabel"),
  interruptForm: $("#interruptForm"), interruptInput: $("#interruptInput"),
  ask: $("#askButton"), mic: $("#micButton"), attach: $("#attachButton"), micHint: $("#micHint"),
  playback: $("#playbackControls"), pauseResume: $("#pauseResumeButton"), stop: $("#stopButton"),
  followups: $("#followups"), followupList: $("#followupList"), quizPanel: $("#quizPanel"),
  quizActions: $("#quizActions"), quizContent: $("#quizContent"),
};

const PAGE_CAPACITY = 7;
const state = {
  courseId: null, files: [], chunksIndexed: 0, generating: false,
  sessionId: null, lectureChunks: [], currentChunkIndex: -1,
  pages: [[]], currentPage: 0, runToken: 0, audio: null, audioResolve: null,
  interrupted: false, recorder: null, micStream: null, recordedParts: [],
  paused: false, stopped: false, followUpQuestions: [], quiz: null, quizIndex: 0, quizScore: 0,
  teachingActive: false,
  hasMoreSections: false, audioUnavailable: false,
  quizResults: [],
};

function setTutorStatus(label) { elements.lessonState.textContent = label; }
function setConnection(connected) {
  elements.signal.classList.toggle("connected", connected);
  elements.connection.textContent = connected ? "Tutor connected" : "Tutor offline";
  elements.backend.textContent = connected ? "Connected" : "Offline";
}
function updateStartState() {
  elements.start.disabled = !elements.topic.value.trim() || state.generating;
}

function renderFiles() {
  elements.fileList.replaceChildren(...state.files.map((item) => {
    const row = document.createElement("div");
    row.className = `file-row ${item.status}`;
    row.innerHTML = '<span class="file-glyph"></span><span class="file-name"></span><span class="file-status"></span><button class="file-remove" type="button" aria-label="Remove material">×</button>';
    row.querySelector(".file-glyph").textContent = item.file.name.split(".").pop().toUpperCase();
    row.querySelector(".file-name").textContent = item.file.name;
    row.querySelector(".file-status").textContent = item.status === "failed" ? item.error : item.status;
    const remove = row.querySelector(".file-remove");
    remove.disabled = item.status === "uploading" || item.status === "processing";
    remove.addEventListener("click", async () => {
      if (item.documentId) {
        remove.disabled = true;
        try {
          const result = await removeMaterial(item.documentId, state.courseId);
          state.chunksIndexed = Math.max(0, state.chunksIndexed - result.chunks_removed);
        } catch (error) { elements.error.textContent = error.message; remove.disabled = false; return; }
      }
      state.files = state.files.filter((candidate) => candidate !== item); renderFiles();
    });
    return row;
  }));
  elements.pdfs.textContent = String(state.files.filter((item) => item.status === "ready").length);
  elements.chunks.textContent = String(state.chunksIndexed);
  updateStartState();
}

async function handleFiles(fileList) {
  elements.error.textContent = "";
  const candidates = [...fileList];
  const allowed = ["pdf", "png", "jpg", "jpeg", "pptx", "txt", "md"];
  const accepted = candidates.filter((file) => allowed.includes(file.name.toLowerCase().split(".").pop()));
  if (accepted.length !== candidates.length) elements.error.textContent = "Some files were skipped. Add PDF, PPTX, PNG, JPG, TXT, or MD files.";
  for (const file of accepted) {
    const entry = { file, status: "uploading", error: "" };
    state.files.push(entry); renderFiles();
    try {
      const result = await uploadMaterial(file, state.courseId, (phase) => { entry.status = phase; renderFiles(); });
      entry.status = "ready"; entry.documentId = result.document.document_id; state.chunksIndexed += result.document.chunks_indexed;
    } catch (error) { entry.status = "failed"; entry.error = error.message; }
    renderFiles();
  }
}

function blockWeight(block) {
  if (["student_question", "tutor_answer"].includes(block.type)) return 2;
  return block.text.length > 170 ? 2 : 1;
}
function appendBlock(block) {
  let page = state.pages[state.pages.length - 1];
  const used = page.reduce((sum, item) => sum + blockWeight(item), 0);
  if (page.length && used + blockWeight(block) > PAGE_CAPACITY) { page = []; state.pages.push(page); }
  page.push(block); state.currentPage = state.pages.length - 1; renderBoard();
  const boardStyles = getComputedStyle($("#whiteboard"));
  const available = $("#whiteboard").clientHeight - parseFloat(boardStyles.paddingTop) - parseFloat(boardStyles.paddingBottom);
  if (page.length > 1 && elements.content.scrollHeight > available) {
    page.pop(); page = [block]; state.pages.push(page); state.currentPage = state.pages.length - 1; renderBoard();
  }
}
function setBlockStatus(id, status) {
  for (const page of state.pages) {
    const block = page.find((item) => item.id === id);
    if (block) block.status = status;
  }
  renderBoard();
}
function renderBoard() {
  const page = state.pages[state.currentPage] || [];
  elements.empty.hidden = page.length > 0;
  elements.content.replaceChildren(...page.map((block) => {
    const row = document.createElement("article");
    row.className = `board-block block-${block.type} ${block.status}`;
    row.dataset.blockId = block.id;
    const pointer = document.createElement("span");
    pointer.className = "focus-pointer"; pointer.textContent = block.status === "active" ? "▶" : "";
    const body = document.createElement("div");
    const label = document.createElement("span");
    label.className = "block-label"; label.textContent = block.type.replace("_", " ");
    const text = document.createElement(block.type === "heading" ? "h2" : "p");
    text.textContent = block.text;
    body.append(label);
    const activeDocumentIds = new Set(state.files.filter((item) => item.status === "ready").map((item) => item.documentId));
    if (block.type === "source_image" && block.source_url && activeDocumentIds.has(block.source_document_id)) {
      const img = document.createElement("img"); img.className = "source-visual";
      img.src = `http://127.0.0.1:8000${block.source_url}`; img.alt = block.text;
      body.append(img, text);
    } else if (["equation", "step_flow", "number_line", "graph", "diagram", "comparison"].includes(block.type)) {
      const visual = document.createElement("div"); visual.className = `teaching-visual visual-${block.type}`;
      const steps = block.text.split(/(?:\n|\s+[↓→↙↘]\s+)/).filter(Boolean);
      steps.forEach((step, index) => {
        const item = document.createElement("div"); item.className = "visual-step"; item.textContent = step; visual.append(item);
        if (index < steps.length - 1) { const arrow = document.createElement("span"); arrow.className = "visual-arrow"; arrow.textContent = "↓"; visual.append(arrow); }
      });
      body.append(visual);
    } else { body.append(text); }
    row.append(pointer, body); return row;
  }));
  elements.pagination.hidden = state.pages.length === 1 && !page.length;
  elements.pageLabel.textContent = `Page ${state.currentPage + 1} / ${state.pages.length}`;
  const navigationLocked = state.teachingActive && !state.paused && !state.stopped;
  elements.previous.disabled = navigationLocked || state.currentPage === 0;
  elements.next.disabled = navigationLocked || state.currentPage >= state.pages.length - 1;
}

function stopAudio() {
  if (state.audio) {
    state.audio.pause(); state.audio.currentTime = 0;
    URL.revokeObjectURL(state.audio.src); state.audio = null;
  }
  if (state.audioResolve) { state.audioResolve(); state.audioResolve = null; }
}
async function speak(text, runToken) {
  setTutorStatus("Speaking");
  const started = performance.now();
  let blob;
  try { blob = await synthesizeSpeech(text); }
  catch (error) {
    console.warn("[TTS] failure", error); state.audioUnavailable = true;
    elements.micHint.textContent = "Audio unavailable — continuing with text."; return;
  }
  console.info(`[PERF] tts: ${Math.round(performance.now() - started)}ms`);
  if (runToken !== state.runToken || state.interrupted) return;
  const audio = new Audio(URL.createObjectURL(blob)); state.audio = audio;
  await new Promise((resolve) => {
    state.audioResolve = resolve;
    audio.addEventListener("ended", resolve, { once: true });
    audio.addEventListener("error", () => { state.audioUnavailable = true; resolve(); }, { once: true });
    audio.play().catch((error) => { console.warn("[TTS] playback unavailable", error); state.audioUnavailable = true; elements.micHint.textContent = "Audio unavailable — continuing with text."; resolve(); });
  });
  URL.revokeObjectURL(audio.src); if (state.audio === audio) state.audio = null;
  state.audioResolve = null;
}

async function runLecture(fromIndex = 0) {
  const token = ++state.runToken; state.interrupted = false; state.stopped = false; state.teachingActive = true;
  elements.playback.hidden = false;
  let index = fromIndex;
  while (true) {
    if (index >= state.lectureChunks.length) {
      if (!state.hasMoreSections || token !== state.runToken || state.stopped) break;
      setTutorStatus("Preparing next section");
      try {
        const section = await nextLessonSection(state.sessionId);
        if (token !== state.runToken || state.stopped) return;
        state.lectureChunks.push(...section.chunks); state.followUpQuestions = section.follow_up_questions || state.followUpQuestions;
        state.hasMoreSections = section.has_more_sections;
      } catch (error) { elements.error.textContent = `Next section unavailable: ${error.message}`; break; }
      continue;
    }
    if (token !== state.runToken || state.interrupted || state.stopped) return;
    state.currentChunkIndex = index;
    let block = state.pages.flat().find((item) => item.id === state.lectureChunks[index].id);
    if (!block) { block = { ...state.lectureChunks[index], status: "active" }; appendBlock(block); }
    else { block.status = "active"; state.currentPage = state.pages.findIndex((page) => page.includes(block)); renderBoard(); }
    setTutorStatus("Speaking");
    await speak(block.type === "source_image" ? `Look at this source visual. ${block.text}` : block.text, token);
    if (token !== state.runToken || state.interrupted || state.stopped) return;
    setBlockStatus(block.id, "complete"); setTutorStatus("Teaching");
    index += 1;
  }
  if (token === state.runToken) {
    appendBlock({ id: crypto.randomUUID(), type: "note", text: "End of lesson. Ask a question any time.", status: "complete" });
    setTutorStatus("Lesson complete"); elements.generation.textContent = "Complete";
    state.teachingActive = false; elements.playback.hidden = true; renderBoard(); showLessonEnd();
  }
}

function showLessonEnd() {
  elements.followupList.replaceChildren(...state.followUpQuestions.map((question, index) => {
    const button = document.createElement("button"); button.type = "button";
    button.textContent = `${index + 1}. ${question}`;
    button.addEventListener("click", () => handleInterruption(`Give me the answer to question ${index + 1}: ${question}`));
    return button;
  }));
  elements.followups.hidden = !state.followUpQuestions.length;
  elements.quizPanel.hidden = false;
}

async function generateLesson() {
  const topic = elements.topic.value.trim(); if (!topic) return;
  state.generating = true; state.runToken += 1; stopAudio();
  state.pages = [[]]; state.currentPage = 0; state.currentChunkIndex = -1;
  state.paused = false; state.stopped = false; elements.followups.hidden = true; elements.quizPanel.hidden = true;
  elements.pauseResume.textContent = "⏸ Pause";
  elements.error.textContent = ""; elements.generation.textContent = "Retrieving";
  setTutorStatus("Preparing lesson"); elements.start.querySelector("span").textContent = "Building lesson…";
  updateStartState(); renderBoard();
  let lesson;
  const started = performance.now();
  try {
    const simpleFastPath = /^(?:the )?table of \d+\??$/i.test(topic);
    if (!simpleFastPath) {
      const retrievalStarted = performance.now();
      const preview = await previewRetrieval(state.courseId, topic);
      elements.retrieval.textContent = String(preview.result_count);
      console.info(`[PERF] retrieval: ${Math.round(performance.now() - retrievalStarted)}ms`);
    } else { elements.retrieval.textContent = "0"; }
    elements.generation.textContent = "Generating once";
    lesson = await createLesson(state.courseId, topic);
    state.sessionId = lesson.session_id; state.lectureChunks = lesson.chunks;
    state.hasMoreSections = Boolean(lesson.has_more_sections); state.audioUnavailable = false;
    state.followUpQuestions = lesson.follow_up_questions || [];
    elements.interruptForm.hidden = false; elements.generation.textContent = "Teaching";
    setTutorStatus("Teaching"); state.generating = false; updateStartState();
  } catch (error) {
    elements.generation.textContent = "Generation error"; setTutorStatus("Lesson generation failed"); elements.error.innerHTML = `${error.message} <button type="button" id="retryLesson">Retry</button>`;
    $("#retryLesson")?.addEventListener("click", generateLesson);
  } finally {
    state.generating = false; elements.start.querySelector("span").textContent = "Start lesson"; updateStartState();
  }
  if (lesson) { console.info(`[PERF] first-board-block: ${Math.round(performance.now() - started)}ms`); await runLecture(0); }
}

function pauseForInterruption() {
  if (!state.sessionId) return false;
  state.interrupted = true; state.runToken += 1; stopAudio();
  const currentId = state.lectureChunks[state.currentChunkIndex]?.id;
  if (currentId) {
    const current = state.pages.flat().find((block) => block.id === currentId);
    if (current) current.status = "complete";
  }
  setTutorStatus("Paused"); renderBoard(); return true;
}

function togglePause() {
  if (!state.sessionId || state.stopped) return;
  if (!state.paused) {
    state.paused = true; state.audio?.pause(); elements.pauseResume.textContent = "▶ Resume"; setTutorStatus("Paused"); renderBoard();
  } else {
    state.paused = false; elements.pauseResume.textContent = "⏸ Pause"; setTutorStatus("Speaking");
    renderBoard(); if (state.audio) state.audio.play().catch((error) => { elements.error.textContent = error.message; });
    else runLecture(Math.max(0, state.currentChunkIndex));
  }
}

function stopLesson() {
  state.stopped = true; state.paused = false; state.interrupted = false; state.runToken += 1;
  state.teachingActive = false;
  stopAudio();
  if (state.recorder?.state === "recording") state.recorder.stop();
  state.micStream?.getTracks().forEach((track) => track.stop());
  elements.pauseResume.textContent = "▶ Continue lesson"; elements.playback.hidden = false;
  setTutorStatus("Stopped"); elements.generation.textContent = "Stopped"; renderBoard();
}

function continueStoppedLesson() {
  if (!state.stopped) return false;
  state.stopped = false; elements.pauseResume.textContent = "⏸ Pause";
  runLecture(Math.max(0, state.currentChunkIndex)); return true;
}

async function handleInterruption(question, alreadyPaused = false) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion || (!alreadyPaused && !pauseForInterruption())) return;
  elements.ask.disabled = true; elements.mic.disabled = true;
  appendBlock({ id: crypto.randomUUID(), type: "student_question", text: cleanQuestion, status: "complete" });
  setTutorStatus("Answering question");
  try {
    const response = await interruptLesson(state.sessionId, cleanQuestion, Math.max(0, state.currentChunkIndex));
    appendBlock(response.answer);
    const answerToken = ++state.runToken; state.interrupted = false;
    await speak(response.answer.text, answerToken);
    setBlockStatus(response.answer.id, "complete");
    appendBlock({ ...response.continuation, status: "active" }); setTutorStatus("Teaching");
    setBlockStatus(response.continuation.id, "complete");
    elements.ask.disabled = false; elements.mic.disabled = false;
    await runLecture(response.resume_chunk_index);
  } catch (error) {
    elements.error.textContent = error.message;
    appendBlock({ id: crypto.randomUUID(), type: "note", text: "The tutor could not answer. Continuing the lesson.", status: "complete" });
    elements.ask.disabled = false; elements.mic.disabled = false;
    await runLecture(Math.max(0, state.currentChunkIndex + 1));
  } finally { elements.ask.disabled = false; elements.mic.disabled = false; }
}

async function toggleMicrophone() {
  if (state.recorder?.state === "recording") { state.recorder.stop(); return; }
  if (!pauseForInterruption()) return;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error("Voice recording is not supported by this browser.");
    console.info("[MIC] permission state: requesting");
    state.micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } }); state.recordedParts = [];
    const preferredTypes = ["audio/mp4", "audio/webm;codecs=opus", "audio/webm"];
    const mimeType = preferredTypes.find((type) => MediaRecorder.isTypeSupported(type));
    state.recorder = new MediaRecorder(state.micStream, mimeType ? { mimeType } : undefined);
    state.recorder.addEventListener("dataavailable", (event) => { if (event.data.size) state.recordedParts.push(event.data); });
    state.recorder.addEventListener("stop", async () => {
      state.micStream.getTracks().forEach((track) => track.stop());
      elements.mic.classList.remove("recording"); elements.mic.textContent = "🎤";
      setTutorStatus("Transcribing"); elements.mic.textContent = "…"; elements.mic.disabled = true;
      elements.micHint.textContent = "Transcribing…";
      try {
        const audio = new Blob(state.recordedParts, { type: state.recorder.mimeType || "audio/webm" });
        const result = await transcribeSpeech(audio);
        if (!result.speech_detected || !result.text) {
          elements.micHint.textContent = "No speech detected. Continuing the lesson.";
          await runLecture(Math.max(0, state.currentChunkIndex + 1)); return;
        }
        elements.micHint.textContent = `Heard: ${result.text}`; elements.mic.disabled = false; elements.mic.textContent = "🎤"; await handleInterruption(result.text, true);
      } catch (error) {
        elements.mic.disabled = false; elements.mic.textContent = "🎤"; elements.micHint.textContent = error.message; await runLecture(Math.max(0, state.currentChunkIndex + 1));
      }
    }, { once: true });
    state.recorder.start(); elements.mic.classList.add("recording"); elements.mic.textContent = "🔴";
    setTutorStatus("Listening…"); elements.micHint.textContent = "Listening… press the red microphone when finished.";
  } catch (error) {
    const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
    console.warn("[MIC] permission failure", error);
    elements.micHint.textContent = denied ? "Microphone unavailable" : `Microphone unavailable: ${error.message}`;
    await runLecture(Math.max(0, state.currentChunkIndex + 1));
  }
}

elements.fileInput.addEventListener("change", (event) => handleFiles(event.target.files));
elements.topic.addEventListener("input", updateStartState); elements.start.addEventListener("click", generateLesson);
elements.interruptForm.addEventListener("submit", (event) => {
  event.preventDefault(); const question = elements.interruptInput.value; elements.interruptInput.value = ""; handleInterruption(question);
});
elements.mic.addEventListener("click", toggleMicrophone);
elements.attach.addEventListener("click", () => elements.fileInput.click());
elements.pauseResume.addEventListener("click", () => { if (!continueStoppedLesson()) togglePause(); });
elements.stop.addEventListener("click", stopLesson);
elements.quizActions.addEventListener("click", async (event) => {
  const count = Number(event.target.dataset.count); if (!count || !state.sessionId) return;
  elements.quizContent.textContent = "Creating lesson quiz…";
  try { state.quiz = await generateQuiz(state.sessionId, count); state.quizIndex = 0; state.quizScore = 0; state.quizResults = []; renderQuizQuestion(); }
  catch (error) { elements.quizContent.textContent = error.message; }
});

function renderQuizQuestion() {
  const questions = state.quiz?.questions || [];
  if (state.quizIndex >= questions.length) {
    const needsReview = state.quizResults.filter((item) => item.result !== "correct").map((item) => item.question);
    const strong = state.quizResults.filter((item) => item.result === "correct").map((item) => item.question);
    elements.quizContent.innerHTML = `<h3>Score: ${state.quizScore} / ${questions.length}</h3><p><strong>Needs review:</strong> ${needsReview.length ? needsReview.join(" · ") : "None"}</p><p><strong>Strong:</strong> ${strong.length ? strong.join(" · ") : "Keep practicing"}</p><button type="button" id="reviewMistakes">Review mistakes</button><button type="button" id="retryQuiz">Try another quiz</button><button type="button" id="returnLesson">Return to lesson</button>`;
    $("#reviewMistakes").addEventListener("click", () => { elements.quizContent.innerHTML = needsReview.length ? `<h3>Review these concepts</h3><p>${needsReview.join("</p><p>")}</p>` : "<p>No mistakes to review.</p>"; });
    $("#retryQuiz").addEventListener("click", () => { state.quizIndex = 0; state.quizScore = 0; state.quizResults = []; renderQuizQuestion(); });
    $("#returnLesson").addEventListener("click", () => { elements.quizContent.replaceChildren(); }); return;
  }
  const question = questions[state.quizIndex]; elements.quizContent.replaceChildren();
  const heading = document.createElement("h3"); heading.textContent = `Question ${state.quizIndex + 1} of ${questions.length}`;
  const prompt = document.createElement("p"); prompt.textContent = question.question;
  const form = document.createElement("form"); form.className = "quiz-question";
  if (question.options.length) question.options.forEach((option) => {
    const label = document.createElement("label"); const input = document.createElement("input");
    input.type = "radio"; input.name = "answer"; input.value = option; label.append(input, document.createTextNode(option)); form.append(label);
  });
  else { const input = document.createElement("textarea"); input.name = "answer"; input.required = true; form.append(input); }
  const submit = document.createElement("button"); submit.type = "submit"; submit.textContent = "Submit"; form.append(submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault(); const answer = new FormData(form).get("answer"); if (!answer) return;
    submit.disabled = true;
    try {
      const grade = await gradeQuiz(state.sessionId, question, String(answer));
      if (grade.result === "correct") state.quizScore += 1;
      state.quizResults.push({ question: question.question, result: grade.result });
      const result = document.createElement("p"); result.className = `quiz-result ${grade.result}`; result.textContent = `${grade.result.replace("_", " ")}: ${grade.explanation}`;
      const next = document.createElement("button"); next.type = "button"; next.textContent = "Next question"; next.addEventListener("click", () => { state.quizIndex += 1; renderQuizQuestion(); });
      form.append(result, next);
    } catch (error) { elements.error.textContent = error.message; submit.disabled = false; }
  });
  elements.quizContent.append(heading, prompt, form);
}
elements.previous.addEventListener("click", () => { state.currentPage -= 1; renderBoard(); });
elements.next.addEventListener("click", () => { state.currentPage += 1; renderBoard(); });
["dragenter", "dragover"].forEach((type) => elements.dropZone.addEventListener(type, (event) => { event.preventDefault(); elements.dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((type) => elements.dropZone.addEventListener(type, (event) => { event.preventDefault(); elements.dropZone.classList.remove("dragging"); }));
elements.dropZone.addEventListener("drop", (event) => handleFiles(event.dataTransfer.files));

async function initialize() {
  try { await checkHealth(); state.courseId = await ensureCourse(); setConnection(true); }
  catch (error) { setConnection(false); elements.error.textContent = `Start the backend to continue. ${error.message}`; }
}
initialize();
