"""
RAG Pipeline
Orchestrates: Document chunking → OpenAI embeddings → Endee storage → Retrieval → LLM answer
"""

import logging
from typing import List, Dict, Any

from openai import OpenAI

from app.endee_store import EndeeVectorStore
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert insurance advisor AI. Answer the user's question 
based ONLY on the provided policy document excerpts. Be concise, accurate, and helpful.
If the answer is not in the context, say "I don't have enough information in the provided 
policy documents to answer that." Never make up coverage details."""


class RAGPipeline:
    def __init__(self):
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.vector_store = EndeeVectorStore()

    def initialize_index(self):
        self.vector_store.initialize()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate OpenAI embeddings for a list of texts."""
        response = self.openai.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        size = settings.chunk_size
        overlap = settings.chunk_overlap
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i: i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks

    def index_documents(self, documents: List[str]) -> int:
        """Chunk, embed, and store documents into Endee."""
        all_chunks, all_meta = [], []
        for doc_idx, doc in enumerate(documents):
            chunks = self._chunk_text(doc)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_meta.append({"text": chunk, "source": f"document_{doc_idx + 1}"})

        logger.info(f"Embedding {len(all_chunks)} chunks...")
        # Batch in groups of 100 (OpenAI limit)
        total = 0
        for i in range(0, len(all_chunks), 100):
            batch_texts = all_chunks[i: i + 100]
            batch_meta = all_meta[i: i + 100]
            embeddings = self._embed(batch_texts)
            total += self.vector_store.upsert(embeddings, batch_meta)

        return total

    def answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Full RAG flow: embed question → retrieve → generate answer."""
        # 1. Embed the question
        q_embedding = self._embed([question])[0]

        # 2. Retrieve top-k relevant chunks from Endee
        retrieved = self.vector_store.query(q_embedding, top_k=top_k)

        if not retrieved:
            return {
                "question": question,
                "answer": "No relevant policy information found. Please index documents first.",
                "sources": [],
            }

        # 3. Build context from retrieved chunks
        context = "\n\n".join(
            [f"[Source: {r['source']} | Similarity: {r['similarity']}]\n{r['text']}"
             for r in retrieved]
        )

        # 4. Generate answer via GPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        completion = self.openai.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        answer = completion.choices[0].message.content.strip()

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
        }

    def clear_index(self):
        self.vector_store.clear()
