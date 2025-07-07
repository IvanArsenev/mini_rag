import logging
from typing import List
from elasticsearch import Elasticsearch
import chardet
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from config import OLLAMA_HOST

logger = logging.getLogger(__name__)

embeddings = OllamaEmbeddings(
    model="llama3",
    base_url=OLLAMA_HOST,
)
llm = Ollama(
    model="llama3",
    base_url=OLLAMA_HOST,
)


class ElasticModule:
    """Elasticsearch wrapper for per-user document storage and embedding."""

    def __init__(self, host: str):
        """Initialize Elasticsearch client."""
        self.es = Elasticsearch(host)

    def check_user_db(self, user_id: str) -> None:
        """Ensure the user index exists in Elasticsearch."""
        if self.es.indices.exists(index=user_id):
            logger.info("Index '%s' already exists.", user_id)
        else:
            body = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 4096
                        }
                    }
                }
            }
            self.es.indices.create(index=user_id, body=body)
            logger.info("Index '%s' was created.", user_id)

    def clear_index(self, user_id: str) -> None:
        """Delete user index."""
        self.es.indices.delete(index=user_id, ignore=[400, 404])
        logger.info("Index '%s' was deleted.", user_id)

    def add_document(self, user_id: str, content: str, embedding: List[float]) -> None:
        """Add document to Elasticsearch index."""
        self.es.index(
            index=user_id,
            body={
                "content": content,
                "embedding": embedding
            }
        )

    async def get_embedding(self, text: str) -> List[float]:
        """Request embedding for a given text from external service."""
        try:
            vector = embeddings.embed_query(text)
            if not vector or not isinstance(vector, list) or not all(isinstance(v, (float, int)) for v in vector):
                logger.error(f"Invalid embedding returned: {vector}")
                raise ValueError("Embedding must be a list of floats")
            return vector
        except Exception as e:
            logger.error(f"Error while getting embedding: {e}")
            raise

    async def add_text_file(self, user_id: str, file_path: str, chunk_size: int = 100) -> None:
        """Chunk a text file and add all parts with embeddings to the index."""
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'

        text = raw_data.decode(encoding)

        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        logger.info("Loaded %d chunks from file '%s'.", len(chunks), file_path)

        for chunk in chunks:
            embedding = await self.get_embedding(chunk)
            self.add_document(user_id, chunk, embedding)

    def search_documents_text(self, user_id: str, query: str, top_k: int = 5) -> List[str]:
        """Search documents by text query."""
        try:
            response = self.es.search(
                index=user_id,
                body={
                    "size": top_k,
                    "query": {
                        "match": {
                            "content": query
                        }
                    }
                }
            )
        except Exception as e:
            logger.error(f"Ошибка при поиске по тексту: {e}")
            return []

        hits = response["hits"]["hits"]
        return [hit["_source"]["content"] for hit in hits]

    def search_documents_vector(self, user_id: str, embedding: List[float], top_k: int = 5) -> List[str]:
        """Search documents by embedding vector."""
        try:
            response = self.es.search(
                index=user_id,
                body={
                    "size": top_k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": embedding}
                            }
                        }
                    }
                }
            )
        except Exception as e:
            logger.error(f"Ошибка при векторном поиске: {e}")
            return []

        hits = response["hits"]["hits"]
        return [hit["_source"]["content"] for hit in hits]
