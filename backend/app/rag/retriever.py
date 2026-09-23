from collections import OrderedDict
from dataclasses import dataclass
import re

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import ChunkRecord, VectorStore
from app.schemas.common import Source


@dataclass(frozen=True)
class RetrievalResult:
    context: str
    sources: list[Source]
    records: list[ChunkRecord]

    @property
    def source_type(self) -> str:
        return "course_material" if self.records else "general_knowledge"


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingService,
        store: VectorStore,
        top_k: int = 6,
        threshold: float = 0.25,
        cache_size: int = 128,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.top_k = top_k
        self.threshold = threshold
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str | None], RetrievalResult] = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def retrieve(self, query: str, course_id: str | None = None) -> RetrievalResult:
        key = (query.strip().casefold(), course_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        attachment_patterns = (
            r"\bexplain (?:this|the) (?:pdf|document|image|file|presentation)\b",
            r"\b(?:teach me|summarize) (?:this|the) (?:document|file|pdf)\b",
            r"\bgo through (?:this|the|these)\b",
            r"\bexplain (?:this|these|the) slides?\b",
            r"\bexplain (?:every|the) chapters?\b",
            r"\bexplain (?:every|each|the) pages?\b",
            r"\bpage[ -]by[ -]page\b",
            r"\b(?:show|display|inspect|look at) (?:me )?(?:the )?(?:original|source) (?:page|pdf|document|image|diagram|figure)\b",
            r"\b(?:teach|explain|solve) (?:me )?(?:this|the) equations?\b",
            r"\b(?:teach|explain|cover|go through)\b.*\b(?:all|everything|them)\b",
            r"\b(?:these|uploaded)\s+(?:pdfs?|documents?|files?|chapters?|materials?|slides?)\b",
        )
        attachment_phrases = {"explain this", "what is this", "teach me this"}
        attachment_reference = key[0] in attachment_phrases or any(re.search(pattern, key[0]) for pattern in attachment_patterns)
        if attachment_reference and course_id:
            candidates = [record for record in reversed(self.store.records) if record.course_id == course_id]
            if "pdf" in key[0]:
                pdf_records = [record for record in candidates if record.document_name.casefold().endswith(".pdf")]
                candidates = pdf_records or candidates
            elif "slide" in key[0] or "presentation" in key[0]:
                slide_records = [record for record in candidates if record.document_name.casefold().endswith(".pptx")]
                candidates = slide_records or candidates
            visual = [record for record in candidates if record.visual_url]
            ranked = visual or candidates
            multi_document_request = bool(re.search(
                r"\b(?:all|every|each|these|uploaded|them|everything)\b", key[0]
            ))
            if multi_document_request:
                # Give the lesson builder evidence from every referenced upload instead
                # of silently anchoring a multi-document request to the newest file.
                chosen = []
                seen_documents: set[str] = set()
                for record in ranked:
                    if record.document_id not in seen_documents:
                        chosen.append(record)
                        seen_documents.add(record.document_id)
                    if len(chosen) == self.top_k:
                        break
                if len(chosen) < self.top_k:
                    chosen_ids = {record.chunk_id for record in chosen}
                    chosen.extend(record for record in ranked if record.chunk_id not in chosen_ids)
                    chosen = chosen[:self.top_k]
            else:
                latest_document_id = ranked[0].document_id if ranked else None
                chosen = [record for record in ranked if record.document_id == latest_document_id][:self.top_k]
            # Limit to at most 4 chunks for context size
            if len(chosen) > 4:
                chosen = chosen[:4]
            # Build context string
            context_str = "\n\n".join(f"[{record.document_name}, p.{record.page}] {record.page_content}" for record in chosen)
            # Truncate to 2500 characters
            if len(context_str) > 2500:
                context_str = context_str[:2500]
            result = RetrievalResult(
                context=context_str,
                sources=[Source(document_id=record.document_id, document_name=record.document_name, page=record.page, chunk_id=record.chunk_id, score=1.0, visual_url=record.visual_url) for record in chosen],
                records=chosen,
            )
            self._cache[key] = result
            return result
        fetch_k = self.top_k * 4 if course_id else self.top_k * 2
        raw = self.store.search(self.embeddings.embed_query(query), fetch_k)
        selected: list[tuple[ChunkRecord, float]] = []
        seen: set[str] = set()
        stop_words = {"the", "and", "this", "that", "what", "how", "why", "teach", "explain", "about", "from", "with", "into", "does", "me"}
        query_terms = {term for term in re.findall(r"[a-z0-9]+", query.casefold()) if len(term) > 2 and term not in stop_words}
        for record, score in raw:
            if course_id and record.course_id != course_id:
                continue
            fingerprint = record.page_content[:100].casefold()
            record_terms = set(re.findall(r"[a-z0-9]+", record.page_content.casefold()))
            has_lexical_anchor = bool(query_terms & record_terms)
            if score < self.threshold or (not has_lexical_anchor and score < 0.5) or fingerprint in seen:
                continue
            seen.add(fingerprint)
            selected.append((record, score))
            if len(selected) == self.top_k:
                break
        result = RetrievalResult(
            context="\n\n".join(
                f"[{record.document_name}, p.{record.page}] {record.page_content}"
                for record, _ in selected
            ),
            sources=[
                Source(
                    document_id=record.document_id,
                    document_name=record.document_name,
                    page=record.page,
                    chunk_id=record.chunk_id,
                    score=round(score, 3),
                    visual_url=record.visual_url,
                )
                for record, score in selected
            ],
            records=[record for record, _ in selected],
        )
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result
