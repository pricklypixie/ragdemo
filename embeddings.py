#!/usr/bin/env python3
"""
Embeddings Library for RAG Applications

This module provides a flexible interface for generating embeddings using
different embedding models and backends.
"""

import os
import sys
import time
import json
import importlib
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple

# CPU vs Metal on MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["MPS_FALLBACK_POLICY"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA

# Set threading options
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Default models for each embedding type
DEFAULT_MODELS = {
	"sentence_transformers": "all-MiniLM-L6-v2",
	"openai": "text-embedding-3-small",
	# Add other embedding types and their defaults here
}

# Embedding dimension for each model
MODEL_DIMENSIONS = {
	# Sentence Transformers models
	"all-MiniLM-L6-v2": 384,
	"all-mpnet-base-v2": 768,
	"multi-qa-distilbert-dot-v1": 768,
	"multi-qa-mpnet-base-dot-v1": 786,
	"paraphrase-multilingual-mpnet-base-v2": 768,
	"nomic-ai/nomic-embed-text-v1": 8192,
	# OpenAI models
	"text-embedding-3-small": 1536,
	"text-embedding-3-large": 3072,
	"text-embedding-ada-002": 1536,
	# Add other models as needed
}


class EmbeddingConfig:
	"""Configuration for embedding generation."""
	
	def __init__(self, 
				 embedding_type: str = "sentence_transformers",
				 model_name: Optional[str] = None,
				 api_key: Optional[str] = None,
				 additional_params: Optional[Dict[str, Any]] = None):
		"""
		Initialize embedding configuration.
		
		Args:
			embedding_type: Type of embedding to use (e.g., "sentence_transformers", "openai")
			model_name: Name of the model to use (if None, uses default for the type)
			api_key: API key for services that require it
			additional_params: Additional parameters for the embedding model
		"""
		self.embedding_type = embedding_type
		
		# Use default model if none specified
		if model_name is None and embedding_type in DEFAULT_MODELS:
			self.model_name = DEFAULT_MODELS[embedding_type]
		else:
			self.model_name = model_name
		
		self.api_key = api_key
		self.additional_params = additional_params or {}
		
		# Validate configuration
		self._validate()
	
	def _validate(self):
		"""Validate the configuration."""
		if not self.embedding_type:
			raise ValueError("Embedding type must be specified")
		
		if not self.model_name:
			raise ValueError(f"Model name must be specified for embedding type {self.embedding_type}")
		
		# Check if API key is required but not provided
		if self.embedding_type == "openai" and not self.api_key:
			# Try to get from environment
			self.api_key = os.environ.get("OPENAI_API_KEY")
			if not self.api_key:
				raise ValueError("API key is required for OpenAI embeddings. Set OPENAI_API_KEY environment variable.")
	
	@classmethod
	def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmbeddingConfig':
		"""Create configuration from a dictionary."""
		return cls(
			embedding_type=config_dict.get("embedding_type", "sentence_transformers"),
			model_name=config_dict.get("model_name"),
			api_key=config_dict.get("api_key"),
			additional_params=config_dict.get("additional_params")
		)
	
	@classmethod
	def from_json_file(cls, config_path: str) -> 'EmbeddingConfig':
		"""Load configuration from a JSON file."""
		try:
			with open(config_path, 'r') as f:
				config_dict = json.load(f)
			return cls.from_dict(config_dict)
		except Exception as e:
			print(f"Error loading embedding configuration from {config_path}: {e}")
			raise
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert configuration to a dictionary."""
		return {
			"embedding_type": self.embedding_type,
			"model_name": self.model_name,
			"api_key": self.api_key,
			"additional_params": self.additional_params
		}
	
	def save_to_file(self, config_path: str) -> None:
		"""Save configuration to a JSON file."""
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		with open(config_path, 'w') as f:
			# Don't save API key to file for security
			config_dict = self.to_dict()
			config_dict["api_key"] = None
			json.dump(config_dict, f, indent=2)
	
	def get_dimension(self) -> int:
		"""Get the dimension of the embedding model."""
		if self.model_name in MODEL_DIMENSIONS:
			return MODEL_DIMENSIONS[self.model_name]
		else:
			# Default size if unknown
			return 768


class EmbeddingFactory:
	"""Factory for creating embedding providers."""
	
	@staticmethod
	def create_provider(config: EmbeddingConfig, debug: bool = False) -> 'BaseEmbeddingProvider':
		"""
		Create and return an embedding provider based on configuration.
		
		Args:
			config: Embedding configuration
			debug: Whether to enable debug logging
			
		Returns:
			An embedding provider instance
		"""
		if config.embedding_type == "sentence_transformers":
			return SentenceTransformersProvider(config, debug)
		elif config.embedding_type == "openai":
			return OpenAIEmbeddingProvider(config, debug)
		else:
			raise ValueError(f"Unsupported embedding type: {config.embedding_type}")


class BaseEmbeddingProvider:
	"""Base class for embedding providers."""
	
	def __init__(self, config: EmbeddingConfig, debug: bool = False):
		"""
		Initialize the embedding provider.
		
		Args:
			config: Embedding configuration
			debug: Whether to enable debug logging
		"""
		self.config = config
		self.debug = debug
		self.model = None
	
	def debug_log(self, message: str) -> None:
		"""Print debug message if debug mode is enabled."""
		if self.debug:
			print(f"[DEBUG] {message}")
	
	def load_model(self) -> None:
		"""Load the embedding model."""
		raise NotImplementedError("Subclasses must implement load_model")
	
	def create_embedding(self, text: str) -> List[float]:
		"""
		Create an embedding for the given text.
		
		Args:
			text: Text to embed
			
		Returns:
			List of floating point values representing the embedding
		"""
		raise NotImplementedError("Subclasses must implement create_embedding")
	
	def get_embedding_dimension(self) -> int:
		"""Get the dimension of the embedding model."""
		return self.config.get_dimension()
	
	def batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
		"""
		Create embeddings for multiple texts.
		
		Args:
			texts: List of texts to embed
			
		Returns:
			List of embeddings
		"""
		# Default implementation - override for more efficient batching
		return [self.create_embedding(text) for text in texts]


class SentenceTransformersProvider(BaseEmbeddingProvider):
	"""Provider for sentence-transformers embeddings."""
	
	def load_model(self) -> None:
		"""Load the sentence-transformers model."""
		if self.model is not None:
			return
		
		try:
			import torch
			from sentence_transformers import SentenceTransformer
			
			self.debug_log(f"PyTorch version: {torch.__version__}")
			self.debug_log(f"Loading Sentence Transformer model: {self.config.model_name} on MPS")
			
			# Force CPU or Metal usage
			self.model = SentenceTransformer(self.config.model_name, device="mps", trust_remote_code=True)
			
			self.debug_log("Model loaded successfully")
		except ImportError as e:
			print(f"Error: Required package not installed - {e}")
			print("Please install with: pip install sentence-transformers torch")
			raise
		except Exception as e:
			print(f"Error loading model: {e}")
			if self.debug:
				print(traceback.format_exc())
			raise
	
	def create_embedding(self, text: str) -> List[float]:
		"""Create an embedding using sentence-transformers."""
		if self.model is None:
			self.load_model()
		
		self.debug_log(f"Generating embedding for text of length {len(text)}")
		
		try:
			import torch
			with torch.no_grad():
				# Process a single item, no batching
				embedding = self.model.encode(
					text,
					convert_to_numpy=True,
					show_progress_bar=False,
					batch_size=1
				).tolist()
			
			self.debug_log(f"Generated embedding with dimension {len(embedding)}")
			return embedding
		except Exception as e:
			print(f"Error generating embedding: {e}")
			if self.debug:
				print(traceback.format_exc())
			return []
	
	def batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
		"""Create embeddings for multiple texts efficiently."""
		if self.model is None:
			self.load_model()
		
		if not texts:
			return []
		
		self.debug_log(f"Batch generating embeddings for {len(texts)} texts")
		
		try:
			import torch
			with torch.no_grad():
				embeddings = self.model.encode(
					texts,
					convert_to_numpy=True,
					show_progress_bar=False,
					batch_size=8  # Use small batch size to balance speed and memory
				).tolist()
			
			# Handle the case where a single embedding is returned
			if len(texts) == 1 and not isinstance(embeddings[0], list):
				embeddings = [embeddings]
				
			self.debug_log(f"Generated {len(embeddings)} embeddings")
			return embeddings
		except Exception as e:
			print(f"Error batch generating embeddings: {e}")
			if self.debug:
				print(traceback.format_exc())
			# Fall back to individual processing
			return super().batch_create_embeddings(texts)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
	"""Provider for OpenAI embeddings."""
	
	def load_model(self) -> None:
		"""Set up the OpenAI client."""
		if self.model is not None:
			return
		
		try:
			# Try to import OpenAI
			try:
				import openai
				self.debug_log("Using OpenAI Python package")
				
				# Initialize the client
				self.model = openai.OpenAI(api_key=self.config.api_key)
				
			except (ImportError, AttributeError):
				# Fall back to older API
				self.debug_log("OpenAI package not found or incompatible, using fallback")
				import openai as openai_legacy
				openai_legacy.api_key = self.config.api_key
				self.model = openai_legacy
				self.legacy_api = True
				
			self.debug_log(f"OpenAI client initialized for model: {self.config.model_name}")
			
		except ImportError as e:
			print(f"Error: OpenAI package not installed - {e}")
			print("Please install with: pip install openai")
			raise
		except Exception as e:
			print(f"Error initializing OpenAI client: {e}")
			if self.debug:
				print(traceback.format_exc())
			raise
	
	def create_embedding(self, text: str) -> List[float]:
		"""Create an embedding using OpenAI's embedding API."""
		if self.model is None:
			self.load_model()
		
		self.debug_log(f"Generating OpenAI embedding for text of length {len(text)}")
		
		try:
			start_time = time.time()
			
			# Check if using legacy or new API
			if hasattr(self, 'legacy_api') and self.legacy_api:
				# Using older OpenAI API
				response = self.model.Embedding.create(
					model=self.config.model_name,
					input=text
				)
				embedding = response["data"][0]["embedding"]
			else:
				# Using newer OpenAI client
				response = self.model.embeddings.create(
					model=self.config.model_name,
					input=text
				)
				embedding = response.data[0].embedding
			
			elapsed_time = time.time() - start_time
			
			self.debug_log(f"Generated embedding with dimension {len(embedding)} in {elapsed_time:.2f}s")
			return embedding
		except Exception as e:
			print(f"Error generating OpenAI embedding: {e}")
			if self.debug:
				print(traceback.format_exc())
			return []


def load_project_config(project_dir: str, document_dir: str, default_config: EmbeddingConfig) -> EmbeddingConfig:
	"""
	Load project-specific embedding configuration or use default.
	
	Args:
		project_dir: Project directory or name
		document_dir: Base documents directory
		default_config: Default configuration to use if no project config exists
		
	Returns:
		EmbeddingConfig for the project
	"""
	if project_dir == "master":
		# For master project, look in the document_dir
		config_path = os.path.join(document_dir, "embedding_config.json")
	else:
		# For other projects, look in the project subdirectory
		config_path = os.path.join(document_dir, project_dir, "embedding_config.json")
	
	if os.path.exists(config_path):
		try:
			return EmbeddingConfig.from_json_file(config_path)
		except Exception as e:
			print(f"Error loading project config, using default: {e}")
			return default_config
	else:
		# No project-specific config found, use default
		return default_config


def get_embedding_provider(project_dir: str = "master", 
						   document_dir: str = "documents",
						   config: Optional[EmbeddingConfig] = None,
						   debug: bool = False) -> BaseEmbeddingProvider:
	"""
	Get the appropriate embedding provider for a project.
	
	Args:
		project_dir: Project directory or name
		document_dir: Base documents directory
		config: Configuration to use (if None, loads from project or default)
		debug: Whether to enable debug logging
		
	Returns:
		An initialized embedding provider
	"""
	# Use provided config or load from project
	if config is None:
		default_config = EmbeddingConfig()
		config = load_project_config(project_dir, document_dir, default_config)
	
	if debug:
		print(f"[DEBUG] Using embedding type: {config.embedding_type}, model: {config.model_name}")
	
	# Create and return the provider
	return EmbeddingFactory.create_provider(config, debug)


# Simple test function
def test_embedding():
	"""Test the embedding module with default settings."""
	print("Testing Embedding Module")
	
	# Create a default configuration
	config = EmbeddingConfig()
	print(f"Using embedding type: {config.embedding_type}, model: {config.model_name}")
	
	# Create provider
	provider = EmbeddingFactory.create_provider(config, debug=True)
	
	# Test embedding generation
	test_text = "This is a test of the embedding system."
	embedding = provider.create_embedding(test_text)
	
	if embedding:
		print(f"Successfully generated embedding with dimension {len(embedding)}")
		print(f"Sample values: {embedding[:3]}...")
	else:
		print("Failed to generate embedding")


if __name__ == "__main__":
	# Run test if executed directly
	test_embedding()