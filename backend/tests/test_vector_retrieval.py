from app.rag.retriever import Retriever
from app.rag.vector_store import ChunkRecord, VectorStore
from tests.fakes import FakeEmbeddings


def record(course: str, text: str, index: int) -> ChunkRecord:
    return ChunkRecord(
        page_content=text, document_id=f"d{index}", course_id=course,
        document_name=f"doc{index}.txt", document_type="notes", page=1,
        chunk_id=f"c{index}",
    )


def test_vector_insertion_retrieval_and_course_filter(tmp_path) -> None:
    embeddings = FakeEmbeddings()
    store = VectorStore(tmp_path, dimension=4)
    records = [record("physics", "force mass acceleration", 1), record("history", "world war causes", 2)]
    store.add(records, embeddings.embed_texts([item.page_content for item in records]))
    retriever = Retriever(embeddings, store, top_k=2, threshold=-1)

    result = retriever.retrieve("force mass acceleration", course_id="physics")
    assert result.records
    assert all(item.course_id == "physics" for item in result.records)
    assert result.source_type == "course_material"

    reloaded = VectorStore(tmp_path, dimension=4)
    assert reloaded.index.ntotal == 2


def test_empty_store_returns_general_knowledge(tmp_path) -> None:
    embeddings = FakeEmbeddings()
    result = Retriever(embeddings, VectorStore(tmp_path, 4)).retrieve("calculus")
    assert result.context == ""
    assert result.source_type == "general_knowledge"
