"""
tutor.py — RAG chain for the COMP-4400 chatbot.

Handles:
  - Loading the ChromaDB vector store (built by ingest.py)
  - Embedding queries and retrieving relevant chunks
  - Building prompts with conversation history
  - Calling the LLM (Groq or OpenAI) and returning (answer, sources)

Set LLM_PROVIDER in .env to "groq" (default, free) or "openai".

Typical usage (from app.py or the REPL):

    from tutor import Tutor
    bot = Tutor()
    answer, sources = bot.ask("What is beta reduction?")
    print(answer)
    for s in sources:
        print(s)
"""

import os
import pathlib
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ── Provider selection ─────────────────────────────────────────────────────────
# Set LLM_PROVIDER=groq or LLM_PROVIDER=openai in your .env
PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}

if PROVIDER == "groq":
    from groq import Groq as _LLMClient
    _API_KEY = os.environ.get("GROQ_API_KEY")
    _KEY_NAME = "GROQ_API_KEY"
else:
    from openai import OpenAI as _LLMClient
    _API_KEY = os.environ.get("OPENAI_API_KEY")
    _KEY_NAME = "OPENAI_API_KEY"

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "comp4400"
TOP_K = 4          # number of chunks to retrieve
HISTORY_TURNS = 4  # recent conversation turns included in prompt

SYSTEM_PROMPT = """\
You are a helpful and knowledgeable tutor for COMP-4400 (Principles of \
Programming Languages) at the University of Windsor. Your role is to help \
students understand course topics such as lambda calculus, Scheme, Prolog, \
MapReduce, axiomatic semantics, code optimization, garbage collection, \
aspect-oriented programming, and large language models.

Answer questions based only on the provided context excerpts from the course \
materials. If the context does not contain enough information to answer the \
question, say so clearly rather than guessing. Be concise, use examples where \
helpful, and cite the source material when relevant.\
"""


class Tutor:
    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODELS[PROVIDER]
        if not _API_KEY:
            raise EnvironmentError(
                f"{_KEY_NAME} is not set. "
                f"Copy .env.example to .env and add your key."
            )
        self.client = _LLMClient(api_key=_API_KEY)
        print(f"Using provider: {PROVIDER} | model: {self.model}")
        self._load_vectorstore()
        self.history: list[dict] = []  # list of {role, content} dicts

    def _load_vectorstore(self):
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {VECTORSTORE_DIR}. "
                "Run `python ingest.py` first."
            )
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
        print(f"Loaded vector store: {self.collection.count()} chunks.")

    def _retrieve(self, query: str) -> list[dict]:
        """Return top-k chunks most similar to query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "distance": round(dist, 4),
            })
        return chunks

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[{i}] (source: {c['source']}, page {c['page']})\n{c['text']}"
            )
        return "\n\n".join(parts)

    def ask(self, question: str) -> tuple[str, list[str]]:
        """
        Ask a question. Returns (answer_text, list_of_source_strings).
        Maintains conversation history internally.
        """
        chunks = self._retrieve(question)
        context = self._build_context(chunks)

        # User message that includes the retrieved context
        user_content = (
            f"Context from course materials:\n{context}\n\nQuestion: {question}"
        )

        # Keep only the last HISTORY_TURNS exchanges to avoid token bloat
        recent_history = self.history[-(HISTORY_TURNS * 2):]

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + recent_history
            + [{"role": "user", "content": user_content}]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

        # Update history with clean question (no context blob) for readability
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        sources = [
            f"{c['source']}, page {c['page']} (similarity: {1 - c['distance']:.2f})"
            for c in chunks
        ]
        return answer, sources

    def reset(self):
        """Clear conversation history."""
        self.history = []


# ── Quick CLI for testing ──────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = Tutor()
    print("COMP-4400 Tutor ready. Type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        if q.lower() == "reset":
            bot.reset()
            print("History cleared.\n")
            continue
        answer, sources = bot.ask(q)
        print(f"\nTutor: {answer}\n")
        print("Sources:")
        for s in sources:
            print(f"  • {s}")
        print()
