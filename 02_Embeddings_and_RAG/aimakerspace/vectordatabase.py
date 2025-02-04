from typing import List, Tuple, Callable, Union, TypeVar, Dict, Optional, Any
import numpy as np
from collections import defaultdict
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

# Type variable for numpy arrays
ArrayType = TypeVar('ArrayType', bound=np.ndarray)

class DocumentRecord:
    """Class to store a document's vector and metadata."""
    def __init__(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        self.vector = vector
        self.metadata = metadata or {}

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class VectorDatabase:
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.documents: Dict[str, DocumentRecord] = {}
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert a vector and its metadata into the database with the given key.

        Args:
            key: Unique identifier for the document
            vector: The embedding vector
            metadata: Optional dictionary containing additional information about the document
        """
        self.documents[key] = DocumentRecord(vector, metadata)

    def update_metadata(self, key: str, metadata: Dict[str, Any], merge: bool = True) -> bool:
        """
        Update metadata for an existing document.

        Args:
            key: The document identifier
            metadata: New metadata dictionary
            merge: If True, merge with existing metadata; if False, replace entirely

        Returns:
            bool: True if update was successful, False if key not found
        """
        if key not in self.documents:
            return False

        if merge:
            self.documents[key].metadata.update(metadata)
        else:
            self.documents[key].metadata = metadata
        return True

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        distance_measure: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for the k nearest vectors using the given distance measure.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            distance_measure: Function to compute distance/similarity
            filter_func: Optional function to filter results based on metadata

        Returns:
            List of tuples containing (key, similarity_score, metadata)
        """
        scores = []
        for key, doc in self.documents.items():
            if filter_func is None or filter_func(doc.metadata):
                similarity = distance_measure(query_vector, doc.vector)
                scores.append((key, similarity, doc.metadata))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        return_as_text: bool = False
    ) -> Union[List[str], List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Search using a text query instead of a vector.

        Args:
            query_text: Text to search for
            k: Number of results to return
            distance_measure: Function to compute distance/similarity
            filter_func: Optional function to filter results based on metadata
            return_as_text: If True, return only the keys of matching documents

        Returns:
            Either list of keys or list of (key, score, metadata) tuples
        """
        query_vector = np.array(self.embedding_model.get_embedding(query_text))
        results = self.search(query_vector, k, distance_measure, filter_func)
        return [result[0] for result in results] if return_as_text else results

    def retrieve(self, key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a vector and its metadata from the database using its key.

        Args:
            key: Document identifier

        Returns:
            Tuple of (vector, metadata) if found, None if not found
        """
        doc = self.documents.get(key)
        return (doc.vector, doc.metadata) if doc else None

    async def abuild_from_list(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> "VectorDatabase":
        """
        Asynchronously build the database from a list of texts and optional metadata.

        Args:
            texts: List of text documents to embed
            metadata_list: Optional list of metadata dictionaries, one per text

        Returns:
            Self for method chaining
        """
        embeddings = await self.embedding_model.async_get_embeddings(texts)

        if metadata_list is None:
            metadata_list = [{} for _ in texts]

        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.insert(text, np.array(embedding), metadata)

        return self

if __name__ == "__main__":
    # Example usage with metadata
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    # Example metadata for each document
    metadata_list = [
        {"category": "food", "timestamp": "2024-01-30", "source": "user_1"},
        {"category": "food", "timestamp": "2024-01-30", "source": "user_2"},
        {"category": "pets", "timestamp": "2024-01-29", "source": "user_3"},
        {"category": "pets", "timestamp": "2024-01-28", "source": "user_4"},
        {"category": "pets", "timestamp": "2024-01-27", "source": "user_5"},
    ]

    # Initialize and build database
    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text, metadata_list))

    # Example search with metadata filter
    def filter_food_category(metadata: Dict[str, Any]) -> bool:
        return metadata.get("category") == "food"

    # Search with filter
    results = vector_db.search_by_text(
        "I think fruit is awesome!",
        k=2,
        filter_func=filter_food_category
    )

    print("Search results (filtered by food category):")
    for key, score, metadata in results:
        print(f"Text: {key}")
        print(f"Similarity: {score:.3f}")
        print(f"Metadata: {metadata}")
        print()

    # Demonstrate metadata update
    vector_db.update_metadata(
        list_of_text[0],
        {"rating": 5, "review_count": 10},
        merge=True
    )

    # Retrieve updated document
    retrieved_doc = vector_db.retrieve(list_of_text[0])
    if retrieved_doc:
        vector, metadata = retrieved_doc
        print("Retrieved document metadata:", metadata)
