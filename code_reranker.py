import os
import subprocess
import shutil
import tempfile
import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Set, Union, Optional
from pathlib import Path
# Import the embedding provider factory
from embeds_util import create_embedding_provider, EmbeddingProvider
from helper_util import create_llm_helper

class CodeReranker:
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the code reranker with configuration.
        
        Args:
            config_path: Path to JSON configuration file
            config: Configuration dictionary (overrides config_path)
        """
        # Default configuration
        self.config = {
            "embedding_provider": "sentence_transformer",
            "embedding_model": "all-MiniLM-L6-v2",
            "repo_dir": None,
            "update": False,
            "github_url": None,
            "save_dir": "save",
            "query_expansion": False,
            "summary": False,
            "helper_type": None
        }
        
        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
                
        # Override with provided config dict
        if config:
            self.config.update(config)
            
        # Extract repo name from github_url if not specified
        if self.config["github_url"] and not self.config.get("repo_name"):
            self.config["repo_name"] = self._get_repo_name(self.config["github_url"])
            
        # Setup embedding provider
        self._setup_embedding_provider()
        
        self.index = None
        self.file_paths = []
        self.repo_dir = None
        
        if self.config['summary'] or self.config['query_expansion']:
            if not self.config.get("helper_type"):
                raise ValueError("Helper type must be specified for summary or query expansion.")
            else:
                self.llm_helper = create_llm_helper(self.config['helper_type'], self.config)
        
    def _setup_embedding_provider(self):
        """Setup the embedding provider based on configuration"""
        
        self.embedding_provider = create_embedding_provider(self.config)
        
    def _get_repo_name(self, github_url: str) -> str:
        """Extract repository name from GitHub URL"""
        return github_url.rstrip('/').split('/')[-1]
        
    def _setup_directories(self) -> None:
        """Setup necessary directories for saving data"""
        repo_name = self.config["repo_name"]
        
        # Setup code directory
        if not self.config.get("repo_dir"):
            self.config["repo_dir"] = os.path.join(self.config["save_dir"], repo_name, "code")
            
        # Ensure repo directory exists
        os.makedirs(self.config["repo_dir"], exist_ok=True)
        
        # Setup cache directory using the provider name for isolation
        provider_name = self.embedding_provider.name
        self.cache_dir = os.path.join(
            self.config["save_dir"], 
            repo_name, 
            "cache", 
            provider_name
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def build_index(self, github_url: Optional[str] = None) -> None:
        """
        Build search index from a GitHub repository
        
        Args:
            github_url: URL of the GitHub repository (overrides config)
        """
        if github_url:
            self.config["github_url"] = github_url
            self.config["repo_name"] = self._get_repo_name(github_url)
            
        if not self.config["github_url"]:
            raise ValueError("GitHub URL must be provided")
            
        # Setup directories
        self._setup_directories()
        self.repo_dir = self.config["repo_dir"]
        
        # Check if we need to update or can load from cache
        index_path = os.path.join(self.cache_dir, "faiss_index.bin")
        paths_path = os.path.join(self.cache_dir, "file_paths.pkl")
        
        if (not self.config["update"] and 
            os.path.exists(index_path) and 
            os.path.exists(paths_path) and
            os.path.exists(self.repo_dir) and
            os.listdir(self.repo_dir)):
            # Load from cache
            print("Loading index from cache...")
            self._load_index()
        else:
            # Clone repo if needed
            if not os.path.exists(self.repo_dir) or not os.listdir(self.repo_dir) or self.config["update"]:
                self._clone_repository()
            
            # Collect code files
            self.file_paths = self._collect_code_files()
            print(f"Found {len(self.file_paths)} code files")
            
            # Generate embeddings
            embeddings = self._generate_embeddings()
            
            # Build the FAISS index
            self._build_faiss_index(embeddings)
            print("Index built successfully")
            
            # Save index and paths
            self._save_index()
    
    def _clone_repository(self) -> None:
        """Clone the repository to the designated directory"""
        if os.path.exists(self.repo_dir) and os.listdir(self.repo_dir):
            print(f"Removing existing repository at {self.repo_dir}...")
            shutil.rmtree(self.repo_dir)
            os.makedirs(self.repo_dir, exist_ok=True)
            
        print(f"Cloning repository to {self.repo_dir}...")
        subprocess.run(
            ["git", "clone", "--depth=1", self.config["github_url"], self.repo_dir], 
            check=True, capture_output=True
        )
    
    def _collect_code_files(self) -> List[str]:
        """Collect all code files from the repository"""
        file_paths = []
        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', 
                          '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.vue', '.md', 
                          '.json', '.yml', '.yaml', '.xml'}
        
        for root, _, files in os.walk(self.repo_dir):
            # Skip hidden directories
            if '/.' in root:
                continue
                
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in code_extensions:
                    abs_path = os.path.join(root, file)
                    # Store paths relative to repo directory
                    rel_path = os.path.relpath(abs_path, self.repo_dir)
                    file_paths.append(rel_path)
        
        return file_paths
    
    def _generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for all code files"""
        contents = []
        
        for file_path in self.file_paths:
            abs_path = os.path.join(self.repo_dir, file_path)
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    # Include file path in content for better context
                    content = f"File: {file_path}\n\n{f.read()}"
                    contents.append(content)
            except (UnicodeDecodeError, IOError) as e:
                print(f"Error reading {file_path}: {e}")
                # Add empty content for failed files to maintain index alignment
                contents.append(f"File: {file_path}")
        
        print("Generating embeddings...")
        # Use the provider to generate embeddings
        embeddings = self.embedding_provider.encode(contents, show_progress_bar=True)
        
        # Save the embeddings
        embeddings_path = os.path.join(self.cache_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        
        return embeddings
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build a FAISS index from the embeddings"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
    
    def _save_index(self) -> None:
        """Save the FAISS index and file paths to disk"""
        if self.index is None:
            raise ValueError("Index not built yet.")
            
        # Save FAISS index
        index_path = os.path.join(self.cache_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save file paths
        paths_path = os.path.join(self.cache_dir, "file_paths.pkl")
        with open(paths_path, 'wb') as f:
            pickle.dump(self.file_paths, f)
            
        print(f"Index saved to {self.cache_dir}")
    
    def _load_index(self) -> None:
        """Load the FAISS index and file paths from disk"""
        index_path = os.path.join(self.cache_dir, "faiss_index.bin")
        paths_path = os.path.join(self.cache_dir, "file_paths.pkl")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load file paths
        with open(paths_path, 'rb') as f:
            self.file_paths = pickle.load(f)
            
        print(f"Loaded index with {len(self.file_paths)} files")
    
    def query(self, question: str, k: int = 10) -> List[str]:
        """
        Query the index with a natural language question
        
        Args:
            question: Natural language query
            k: Number of results to return
            
        Returns:
            List of relevant file paths
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        # Generate embedding for the question using the provider
        if self.config["query_expansion"] and self.llm_helper:
            questions = self.llm_helper.get_expansion(question)
            print(f"Expanded queries: {questions}")
        else:
            questions = [question]
            
        all_results = []
        all_indices = set()
        
        
        # Process each question
        # TODO better model for reranking
        for q in questions:
            # Generate embedding for the question
            q_embedding = self.embedding_provider.encode([q])
            
            # Search the index
            _, indices = self.index.search(q_embedding.astype(np.float32), k)
            
            # Collect unique results
            for idx in indices[0]:
                if 0 <= idx < len(self.file_paths) and idx not in all_indices:
                    all_results.append(self.file_paths[idx])
                    all_indices.add(idx)
                    
                    # Stop when we have k results
                    if len(all_results) >= k:
                        break
                        
            # Stop when we have k results
            if len(all_results) >= k:
                break
                
        # Return the top k results (or fewer if not enough found)
        
        if self.config["summary"] and self.llm_helper:
            # Generate summaries for the results
            file_summaries = {}
            for result in all_results[:k]:
                summary = self.llm_helper.get_summary(result)
                file_summaries[result] = summary
                print(f"File: {result}\nSummary: {summary}\n")
        
        return all_results[:k]
    
    def cleanup(self) -> None:
        """Clean up temporary repository directory"""
        # Only clean up if it's a temp directory, not our save directory
        if self.repo_dir and os.path.exists(self.repo_dir) and self.repo_dir != self.config["repo_dir"]:
            shutil.rmtree(self.repo_dir)
            self.repo_dir = None