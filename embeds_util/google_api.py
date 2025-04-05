from typing import List, Dict, Optional, Union
import os
import numpy as np
from .base import EmbeddingProvider
from google import genai
from google.genai import types

class GoogleEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Google's embedding API"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Google embedding provider
        
        Args:
            config: Configuration dictionary with api_key and embedding_model
        """
        self.config = config or {}
        self.embedding_model = self.config.get("embedding_model", "gemini-embedding-exp-03-07")
        self.api_key = self.config.get("api_key")
        self.embedding_config = types.EmbedContentConfig(task_type=self.config.get("task_type","CODE_RETRIEVAL_QUERY"))
        self.batch_size = self.config.get("batch_size", 100)  # Process texts in batches
        
        # Initialize Google genai client
        self._init_client()
    
    def _init_client(self):
        """Initialize Google genai client"""
        try:
            # Check for API key
            if not self.api_key and not self.config.get("api_key") and "GOOGLE_API_KEY" not in os.environ:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY environment "
                    "variable or provide 'api_key' in config."
                )
                
            # Use API key from config or environment variable
            api_key = self.api_key or self.config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
            
            # Initialize the client
            self.client = genai.Client(api_key=api_key)
            
        except ImportError:
            raise ImportError(
                "Google genai SDK is not installed. "
                "Install it with: pip install google-generativeai"
            )
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Generate embeddings using Google's embedding API
        
        Args:
            texts: List of text strings to encode
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of embeddings
        """
        if not hasattr(self, "client"):
            self._init_client()
        
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            if kwargs.get("show_progress_bar", False):
                print(f"Processing batch {i//self.batch_size + 1}/{len(texts)//self.batch_size + 1}")
            
            # Call the Google API
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=batch
            )
            
            # Extract embeddings from response
            for embedding in response.embeddings:
                all_embeddings.append(np.array(embedding.values))
        
        # Stack all embeddings into a single numpy array
        return np.vstack(all_embeddings)
    
    @property
    def name(self) -> str:
        """Get provider name"""
        return f"google_{self.embedding_model}"