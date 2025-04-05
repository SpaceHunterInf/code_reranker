from typing import Dict, Optional
from .base import EmbeddingProvider
from .sentence_transformer import SentenceTransformerProvider
from .google_api import GoogleEmbeddingProvider

def create_embedding_provider(config: Optional[Dict] = None) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider
    
    Args:
        provider_type: Type of provider ('sentence_transformer' or 'google')
        config: Configuration for the provider
        
    Returns:
        An initialized EmbeddingProvider instance
    """
    config = config or {}
    
    provider_type = config.get("embedding_provider", "sentence_transformer")
    if provider_type == "sentence_transformer":
        return SentenceTransformerProvider(config)
    elif provider_type == "google":
        return GoogleEmbeddingProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")