from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import EmbeddingProvider

class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using the SentenceTransformers library"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SentenceTransformers provider
        
        Args:
            config: Configuration dictionary with model_name
        """
        self.config = config or {}
        self.model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformer model
        
        Args:
            texts: List of text strings to encode
            **kwargs: Additional arguments passed to model.encode()
            
        Returns:
            Numpy array of embeddings
        """
        # Apply any default kwargs for SentenceTransformer
        kwargs.setdefault("show_progress_bar", True)
        
        # Generate embeddings
        return self.model.encode(texts, **kwargs)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def name(self) -> str:
        """Get provider name"""
        return f"sentence_transformer_{self.model_name}"