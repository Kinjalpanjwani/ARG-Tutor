from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from app.core.config import Settings
from app.llm.groq_client import GroqService
from app.llm.guardrails import AcademicGuard
from app.llm.planner import TeachingPlanner
from app.llm.tutor import TutorService
from app.rag.chunker import DocumentChunker
from app.rag.embeddings import SentenceTransformerEmbeddings
from app.rag.ingestion import IngestionService
from app.rag.loader import DocumentLoader
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.speech.stt import SpeechToTextService
from app.speech.tts import TextToSpeechService
from app.speech.vad import VoiceActivityDetector
from app.storage import CourseRepository, DocumentRepository
from app.teaching.engine import TeachingEngine
from app.teaching.followups import FollowupService
from app.teaching.quiz import QuizService
from app.teaching.session import SessionStore
from app.teaching.notebook_lesson import NotebookLessonService


@dataclass
class Container:
    runtime_dir: Path
    settings: Settings
    courses: CourseRepository
    documents: DocumentRepository
    ingestion: IngestionService
    retriever: Retriever
    guard: AcademicGuard
    tutor: TutorService
    sessions: SessionStore
    teaching: TeachingEngine
    followups: FollowupService
    quizzes: QuizService
    stt: SpeechToTextService
    tts: TextToSpeechService
    vad: VoiceActivityDetector
    notebook_lessons: NotebookLessonService


def build_container(settings: Settings) -> Container:
    runtime_dir = Path(mkdtemp(prefix="ai-tutor-"))
    uploads_dir = runtime_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    courses = CourseRepository(settings.data_dir / "courses.json")
    documents = DocumentRepository(runtime_dir / "documents.json")
    embeddings = SentenceTransformerEmbeddings(settings.embedding_model, settings.embedding_dimension)
    store = VectorStore(runtime_dir / "vector_store", settings.embedding_dimension)
    retriever = Retriever(embeddings, store, settings.retrieval_top_k, settings.retrieval_threshold)
    groq = GroqService(settings.groq_api_key, settings.groq_model, settings.groq_whisper_model)
    guard = AcademicGuard(groq)
    tutor = TutorService(groq)
    sessions = SessionStore()
    ingestion = IngestionService(
        DocumentLoader(runtime_dir / "visuals"), DocumentChunker(settings.chunk_size, settings.chunk_overlap),
        embeddings, store, documents, retriever,
    )
    teaching = TeachingEngine(sessions, guard, TeachingPlanner(groq), tutor, retriever)
    vad = VoiceActivityDetector()
    return Container(
        runtime_dir=runtime_dir, settings=settings, courses=courses, documents=documents,
        ingestion=ingestion, retriever=retriever, guard=guard, tutor=tutor,
        sessions=sessions, teaching=teaching,
        followups=FollowupService(sessions, tutor),
        quizzes=QuizService(sessions, tutor, retriever),
        stt=SpeechToTextService(groq, vad),
        tts=TextToSpeechService(runtime_dir / "audio", settings.tts_voice_english, settings.tts_voice_urdu),
        vad=vad,
        notebook_lessons=NotebookLessonService(groq, guard, retriever),
    )
