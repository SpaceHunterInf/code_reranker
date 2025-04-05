from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union

class LLMHelper(ABC):
    """Abstract base class for language model helpers"""
    
    @abstractmethod
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the helper with configuration"""
        pass
    
    @abstractmethod
    def get_expansion(self, query: str) -> List[str]:
        """
        Expand a query into multiple related queries
        
        Args:
            query: The original query string
            
        Returns:
            List of expanded query strings
        """
        pass
    
    @abstractmethod
    def get_summary(self, content: str, max_length: Optional[int] = None) -> str:
        """
        Generate a concise summary of the provided content
        
        Args:
            content: The text content to summarize
            max_length: Optional maximum length for the summary
            
        Returns:
            Summary string
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name identifier for this helper
        
        Returns:
            String name of the helper
        """
        pass