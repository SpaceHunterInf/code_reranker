from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import numpy as np

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the embedding provider with configuration"""
        pass
    
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to encode
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dimension)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name identifier for this provider
        
        Returns:
            String name of the provider
        """
        pass