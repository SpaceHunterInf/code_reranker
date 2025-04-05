from typing import Optional, Dict, List, Union
import time
from .base import LLMHelper
from .prompts import expansion_prompt, summary_prompt
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class VLLMHelper(LLMHelper):
    """Helper implementation using vLLM for local inference"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the vLLM helper
        
        Args:
            config: Configuration dictionary with model_name, etc.
        """
        self.config = config or {}
        self.dtype = self.config.get("dtype", "float16")
        self.model_name = self.config.get("helper_model", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.max_tokens = self.config.get("helper_model_max_tokens", 512)
        self.temperature = self.config.get("helper_model_temperature", 0)  # Low temperature for more focused outputs
        self.chat_mode = self.config.get("chat_mode", False)
        self.client = None
        
        if self.chat_mode:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.expansion_prompt = expansion_prompt
        
        self.summary_prompt = summary_prompt
        
        # Initialize vLLM client
        self._init_client()
    
    def _init_client(self):
        """Initialize the vLLM client"""
            
        # Initialize the model
        print(f"Loading vLLM model: {self.model_name}")
        self.llm = LLM(model=self.model_name, dtype=self.dtype, gpu_memory_utilization=0.5)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


    def get_expansion(self, query: str, count: int = 3) -> List[str]:
        """
        Expand a query into multiple related queries using vLLM
        
        Args:
            query: The original query string
            count: Number of expanded queries to generate
            
        Returns:
            List of expanded query strings
        """
        if not hasattr(self, "llm"):
            self._init_client()
            
        # Format the prompt
        
        prompt = self.expansion_prompt.format(query=query, count=count)
        
        # Generate output
        outputs = self.llm.generate([prompt], self.sampling_params)
        result = outputs[0].outputs[0].text.strip()
        
        # Parse the result into separate queries
        expanded_queries = [q.strip() for q in result.split('\n') if q.strip()]
        
        # Add the original query as the first in the list
        if query not in expanded_queries:
            expanded_queries = [query] + expanded_queries[:count-1]
            
        return expanded_queries[:count]  # Ensure we only return the requested count
    
    def get_summary(self, content: str, max_length: Optional[int] = None) -> str:
        """
        Generate a summary of the provided content using vLLM
        
        Args:
            content: The text content to summarize
            max_length: Maximum length for the summary in words
            
        Returns:
            Summary string
        """
        if not hasattr(self, "llm"):
            self._init_client()
            
        # Set default max length if not provided
        if max_length is None:
            max_length = self.config.get("summary_max_length", 50)
            
        # Format the prompt
        if self.chat_mode:
            messages = [
                {"role": "system", "content": self.summary_prompt.format(max_length=max_length)},
                {"role": "user", "content": content},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            prompt = self.summary_prompt.format(max_length=max_length) + '\n Content: ' + content + '\n\nSummary:'
        
        # Generate output
        outputs = self.llm.generate([prompt], self.sampling_params)
        summary = outputs[0].outputs[0].text.strip()
        
        return summary
    
    @property
    def name(self) -> str:
        """Get helper name"""
        return f"vllm_{self.model_name.replace('/', '_')}"