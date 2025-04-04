import os
import subprocess
import shutil
import tempfile
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Set, Union, Optional
from pathlib import Path

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
            "embedding_model": "all-MiniLM-L6-v2",
            "repo_dir": None,
            "update": False,
            "github_url": None,
            "save_dir": "save"
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
            
        # Setup model
        self.model = SentenceTransformer(self.config["embedding_model"])
        self.index = None
        self.file_paths = []
        self.repo_dir = None
        
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
        
        # Setup cache directory
        self.cache_dir = os.path.join(
            self.config["save_dir"], 
            repo_name, 
            "cache", 
            self.config["embedding_model"]
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
        embeddings = self.model.encode(contents, show_progress_bar=True)
        
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
            
        # Generate embedding for the question
        question_embedding = self.model.encode([question])
        
        # Search the index
        _, indices = self.index.search(question_embedding.astype(np.float32), k)
        
        # Return file paths for the top results
        results = [self.file_paths[idx] for idx in indices[0] if 0 <= idx < len(self.file_paths)]
        return results
    
    def cleanup(self) -> None:
        """Clean up temporary repository directory"""
        # Only clean up if it's a temp directory, not our save directory
        if self.repo_dir and os.path.exists(self.repo_dir) and self.repo_dir != self.config["repo_dir"]:
            shutil.rmtree(self.repo_dir)
            self.repo_dir = None