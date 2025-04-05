from typing import Dict, Optional
from .base import LLMHelper
from .vllm_helper import VLLMHelper

def create_llm_helper(helper_type: str, config: Optional[Dict] = None) -> LLMHelper:
    """
    Factory function to create an LLM helper
    
    Args:
        helper_type: Type of helper ('vllm' or 'google')
        config: Configuration for the helper
        
    Returns:
        An initialized LLMHelper instance
    """
    config = config or {}
    
    if helper_type == "vllm":
        return VLLMHelper(config)
    elif helper_type == "google":
        raise NotImplementedError("Google helper is not implemented yet.")
    else:
        raise ValueError(f"Unknown helper type: {helper_type}")