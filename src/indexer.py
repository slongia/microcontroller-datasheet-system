from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


class DatasheetIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []  # Store (filename, text) for retrieval

    def embed_text(self, text):
        """Generate embedding for a text."""
        return self.model.encode(text, convert_to_numpy=True)

    def add_to_index(self, datasheets):
        """Add datasheets to FAISS index."""
        for filename, text in datasheets:
            embedding = self.embed_text(text)
            self.index.add(np.array([embedding]))
            self.metadata.append((filename, text))

    def save_index(
        self, index_path="data/index.faiss", metadata_path="data/metadata.pkl"
    ):
        """Save index and metadata."""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(
        self, index_path="data/index.faiss", metadata_path="data/metadata.pkl"
    ):
        """Load index and metadata."""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query, k=1):
        """Search for top-k relevant datasheets."""
        query_embedding = self.embed_text(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        results = [
            (self.metadata[i], distances[0][j]) for j, i in enumerate(indices[0])
        ]
        return results


if __name__ == "__main__":
    from pdf_processor import process_directory

    datasheets = process_directory("data/datasheets")
    indexer = DatasheetIndexer()
    indexer.add_to_index(datasheets)
    indexer.save_index()
    results = indexer.search("What is the clock speed of STM32F303?", k=1)
    for (filename, text), distance in results:
        print(f"Found in {filename}: {text[:100]}... (Distance: {distance})")
