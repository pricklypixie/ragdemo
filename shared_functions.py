#!/usr/bin/env python3
"""
Shared Functions for RAG Applications

This module provides common functions used by rag_query.py, document_indexer.py,
and other components of the RAG system.
"""

import os
import sys
import json
import time
import signal
import traceback
import pickle
import glob
import subprocess
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path

import anthropic
try:
	from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
	pass

# Import our embedding library
from embeddings import EmbeddingConfig, get_embedding_provider, load_project_config

# Constants (should match those in rag_query.py and document_indexer.py)
# Claude model constants
CLAUDE_HAIKU = "claude-3-5-haiku-20241022"  # Default Claude model
CLAUDE_SONNET = "claude-3-sonnet-20240229"
CLAUDE_OPUS = "claude-3-opus-20240229"
CLAUDE_HAIKU_LEGACY = "claude-3-haiku-20240307"
CLAUDE_SONNET_LEGACY = "claude-3-sonnet-20240229"

# OpenAI constants
LLM_OPENAI = "openai"  # New LLM type for OpenAI
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # Default OpenAI model

# Other OpenAI models
OPENAI_O3_MINI = "o3-mini"
OPENAI_GPT3_TURBO = "gpt-3.5-turbo"
OPENAI_GPT4_TURBO = "gpt-4-turbo"
OPENAI_GPT4 = "gpt-4"
OPENAI_GPT4O = "gpt-4o"
OPENAI_GPT4O_MINI = "gpt-4o-mini"

DEFAULT_CLAUDE_MODEL = CLAUDE_HAIKU
MAX_TOKENS = 8096
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_TYPE = "sentence_transformers"
TOP_K_DOCUMENTS = 3
API_TIMEOUT = 120  # Timeout for API calls in seconds
MASTER_PROJECT = "master"  # Name for the master index
PROMPTS_DIR = "prompts"  # Directory to save prompt logs
DEFAULT_CHARS_PER_DIMENSION = 4

# LLM types
LLM_CLAUDE = "claude"
LLM_LOCAL = "local"
LLM_HF = "hf"

# Default models
DEFAULT_LLM_TYPE = LLM_LOCAL
DEFAULT_LOCAL_MODEL = "mistral-7b-instruct-v0"
DEFAULT_HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class APITimeoutError(Exception):
	"""Exception raised when an API call times out."""
	pass

class Document:
	"""Represents a document with text content and metadata."""
	
	def __init__(self, 
				 content: str, 
				 metadata: Dict[str, Any],
				 embedding: Optional[List[float]] = None):
		self.content = content
		self.metadata = metadata
		self.embedding = embedding

class EmbeddingProviderCache:
	"""Caches embedding providers to avoid reloading models."""
	
	def __init__(self, debug=False):
		"""Initialize the cache."""
		self.providers = {}  # {(project, embedding_type, model_name): provider}
		self.debug = debug
		# Add a counter for cache hits/misses for debugging
		self.hits = 0
		self.misses = 0
	
	def get_provider(self, project, document_dir, config=None):
		"""
		Get an embedding provider from the cache or create a new one.
		
		Args:
			project: Project name
			document_dir: Base document directory
			config: Optional EmbeddingConfig (loads from project if not provided)
			
		Returns:
			An embedding provider
		"""
		# If no config provided, load from project
		if config is None:
			config_path = get_project_config_path(project, document_dir)
			if os.path.exists(config_path):
				try:
					config = EmbeddingConfig.from_json_file(config_path)
					if self.debug:
						print(f"[DEBUG] Loaded embedding config from: {config_path}")
				except Exception as e:
					if self.debug:
						print(f"[DEBUG] Error loading config from {config_path}: {e}")
					# Use defaults
					config = EmbeddingConfig(
						embedding_type=DEFAULT_EMBEDDING_TYPE,
						model_name=DEFAULT_EMBEDDING_MODEL
					)
			else:
				# Use defaults
				config = EmbeddingConfig(
					embedding_type=DEFAULT_EMBEDDING_TYPE,
					model_name=DEFAULT_EMBEDDING_MODEL
				)
		
		# Create a cache key from the project and config
		cache_key = (project, config.embedding_type, config.model_name)
		
		# Check if we already have this provider in the cache
		if cache_key in self.providers:
			if self.debug:
				self.hits += 1
				print(f"[DEBUG] Using cached embedding provider for {cache_key} (hits: {self.hits}, misses: {self.misses})")
			return self.providers[cache_key]
		
		# Create a new provider
		if self.debug:
			self.misses += 1
			print(f"[DEBUG] Creating new embedding provider for {cache_key} (hits: {self.hits}, misses: {self.misses})")
		
		# Create the provider
		provider = get_embedding_provider(
			project_dir=project,
			document_dir=document_dir,
			config=config,
			debug=self.debug
		)
		
		# Explicitly force loading the model now to ensure it's ready
		# This can be slow the first time but then it's cached
		if self.debug:
			print(f"[DEBUG] Pre-loading model for {config.embedding_type}/{config.model_name}")
		
		# For SentenceTransformers, we need to explicitly call load_model
		if hasattr(provider, 'load_model'):
			provider.load_model()
		# For other providers that lazy-load, try creating a simple embedding to force loading
		else:
			provider.create_embedding("Test loading the model")
		
		# Store in cache
		self.providers[cache_key] = provider
		return provider
	
	def clear_cache(self):
		"""Clear the provider cache."""
		old_count = len(self.providers)
		self.providers.clear()
		self.hits = 0
		self.misses = 0
		if self.debug:
			print(f"[DEBUG] Cleared embedding provider cache ({old_count} providers removed)")

def timeout_handler(signum, frame):
	"""Signal handler for timeouts."""
	raise APITimeoutError("API call timed out")

def get_project_config_path(project: str, document_dir: str, use_legacy: bool = False) -> str:
	"""
	Get the path to the project's configuration file.
	
	Args:
		project: Project name
		document_dir: Base document directory
		use_legacy: If True, returns path to legacy embedding_config.json, 
				  otherwise returns path to project_config.json
	
	Returns:
		Path to the configuration file
	"""
	if project == MASTER_PROJECT:
		# For master project, look in the document_dir
		if use_legacy:
			return os.path.join(document_dir, "embedding_config.json")
		else:
			return os.path.join(document_dir, "project_config.json")
	else:
		# For other projects, look in the project subdirectory
		if use_legacy:
			return os.path.join(document_dir, project, "embedding_config.json")
		else:
			return os.path.join(document_dir, project, "project_config.json")

def load_project_config_file(project: str, document_dir: str) -> Dict[str, Any]:
	"""
	Load the complete project configuration from file.
	
	Args:
		project: Project name
		document_dir: Base documents directory
		
	Returns:
		Complete project configuration dictionary
	"""
	# Try new project_config.json first
	config_path = get_project_config_path(project, document_dir, use_legacy=False)
	
	if os.path.exists(config_path):
		try:
			with open(config_path, 'r') as f:
				return json.load(f)
		except Exception as e:
			print(f"Error loading project config: {e}")
	
	# If new format not found, check for legacy embedding_config.json
	legacy_path = get_project_config_path(project, document_dir, use_legacy=True)
	
	if os.path.exists(legacy_path):
		try:
			# Load legacy embedding config
			with open(legacy_path, 'r') as f:
				embedding_config = json.load(f)
			
			# Build a new format project config
			return {
				"indexing": embedding_config,
				"rag": {
					"llm_type": DEFAULT_LLM_TYPE,
					"llm_model": DEFAULT_LOCAL_MODEL,
					"rag_mode": "chunk",
					"rag_count": TOP_K_DOCUMENTS
				}
			}
		except Exception as e:
			print(f"Error loading legacy embedding config: {e}")
	
	# Return default config if nothing found
	return {
		"indexing": {
			"embedding_type": DEFAULT_EMBEDDING_TYPE,
			"model_name": DEFAULT_EMBEDDING_MODEL,
			"api_key": None,
			"additional_params": {}
		},
		"rag": {
			"llm_type": DEFAULT_LLM_TYPE,
			"llm_model": DEFAULT_LOCAL_MODEL,
			"rag_mode": "chunk",
			"rag_count": TOP_K_DOCUMENTS
		}
	}

def get_project_embedding_config(project: str, document_dir: str, debug: bool = False) -> EmbeddingConfig:
	"""
	Get the embedding configuration for a project from the project config.
	
	Args:
		project: Project name
		document_dir: Base documents directory
		debug: Whether to enable debug output
		
	Returns:
		EmbeddingConfig object
	"""
	project_config = load_project_config_file(project, document_dir)
	
	# Extract indexing configuration
	indexing_config = project_config.get("indexing", {})
	
	if debug:
		print(f"[DEBUG] Loaded project config for {project}")
		print(f"[DEBUG] Indexing config: {indexing_config}")
	
	# Convert to EmbeddingConfig object
	return EmbeddingConfig.from_dict(indexing_config)

def save_embedding_config(project: str, document_dir: str, config: EmbeddingConfig) -> None:
	"""
	Save embedding configuration as part of the project configuration.
	
	Args:
		project: Project name
		document_dir: Base documents directory
		config: EmbeddingConfig to save
	"""
	# Get the path to the project config
	config_path = get_project_config_path(project, document_dir, use_legacy=False)
	
	# Load existing config or create new one
	if os.path.exists(config_path):
		try:
			with open(config_path, 'r') as f:
				project_config = json.load(f)
		except Exception:
			# Start with default if can't load existing
			project_config = {
				"indexing": {},
				"rag": {
					"llm_type": DEFAULT_LLM_TYPE,
					"llm_model": DEFAULT_LOCAL_MODEL,
					"rag_mode": "chunk",
					"rag_count": TOP_K_DOCUMENTS
				}
			}
	else:
		# Start with default config
		project_config = {
			"indexing": {},
			"rag": {
				"llm_type": DEFAULT_LLM_TYPE,
				"llm_model": DEFAULT_LOCAL_MODEL,
				"rag_mode": "chunk",
				"rag_count": TOP_K_DOCUMENTS
			}
		}
	
	# Update the indexing section
	project_config["indexing"] = config.to_indexing_dict()
	
	# Ensure we don't save API key to file
	if "api_key" in project_config["indexing"]:
		project_config["indexing"]["api_key"] = None
	
	# Create directory if needed
	os.makedirs(os.path.dirname(config_path), exist_ok=True)
	
	# Save the updated config
	with open(config_path, 'w') as f:
		json.dump(project_config, f, indent=2)
	
	print(f"Saved project configuration to {config_path}")

def get_index_path(index_dir: str, project: str) -> Tuple[str, str]:
	"""Get the index path and backup directory for a project."""
	if project == MASTER_PROJECT:
		index_path = os.path.join(index_dir, "document_index.pkl")
	else:
		# Project subdirectory in the index directory
		project_dir = os.path.join(index_dir, project)
		index_path = os.path.join(project_dir, "document_index.pkl")
	
	# Backup directory
	backup_dir = os.path.join(os.path.dirname(index_path), "backups")
	
	return index_path, backup_dir

def load_index(index_path: str, backup_dir: str, debug: bool = False) -> List[Document]:
	"""Load the document index from disk."""
	try:
		with open(index_path, 'rb') as f:
			documents = pickle.load(f)
		print(f"Loaded {len(documents)} documents from index: {index_path}")
		return documents
	except Exception as e:
		print(f"Error loading index: {e}")
		# Try to load from backup if main index fails
		backup_files = sorted(glob.glob(os.path.join(backup_dir, "*.pkl")), reverse=True)
		if backup_files:
			print(f"Attempting to load from latest backup: {backup_files[0]}")
			try:
				with open(backup_files[0], 'rb') as f:
					documents = pickle.load(f)
				print(f"Loaded {len(documents)} documents from backup")
				return documents
			except Exception as backup_error:
				print(f"Error loading backup: {backup_error}")
		
		return []

def discover_projects(document_dir: str, index_dir: str = None, use_sqlite: bool = False) -> List[str]:
	"""
	Discover all projects (subdirectories) in the document directory.
	
	Args:
		document_dir: Base documents directory
		index_dir: Base index directory (required if use_sqlite=True)
		use_sqlite: Whether to use SQLite storage
		
	Returns:
		List of project names
	"""
	if use_sqlite:
		if not index_dir:
			print("Error: Index directory is required for SQLite storage")
			return [MASTER_PROJECT]
			
		try:
			from sqlite_storage import discover_projects as discover_sqlite_projects
			projects = discover_sqlite_projects(index_dir)
			return projects
		except ImportError:
			print(f"Error: sqlite_storage module not found. Please make sure it's installed.")
			return [MASTER_PROJECT]
		except Exception as e:
			print(f"Error discovering projects from SQLite: {e}")
			return [MASTER_PROJECT]
	else:
		# Original implementation for document directories
		projects = [MASTER_PROJECT]  # Master project is always included
		
		try:
			# Get all subdirectories in the document directory
			for item in os.listdir(document_dir):
				item_path = os.path.join(document_dir, item)
				if os.path.isdir(item_path):
					projects.append(item)
			
			return projects
		except Exception as e:
			print(f"Error discovering projects: {e}")
			return [MASTER_PROJECT]  # Return at least the master project

def clear_index(project: str, index_dir: str, use_sqlite: bool = False, debug: bool = False) -> bool:
	"""
	Clear (erase) the index for a specific project.
	
	Args:
		project: The project name to clear
		index_dir: Base index directory
		use_sqlite: Whether to use SQLite storage
		debug: Enable debug logging
		
	Returns:
		True if successful, False otherwise
	"""
	if use_sqlite:
		try:
			from sqlite_storage import clear_project_index
			success = clear_project_index(index_dir, project, debug)
			return success
		except ImportError:
			print(f"Error: sqlite_storage module not found. Please make sure it's installed.")
			return False
		except Exception as e:
			print(f"Error clearing SQLite index: {e}")
			if debug:
				print(traceback.format_exc())
			return False
	else:
		# Original pickle implementation
		index_path, backup_dir = get_index_path(index_dir, project)
		
		if not os.path.exists(index_path):
			print(f"Index file not found: {index_path}")
			return False
		
		try:
			# Create a backup before deletion
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			backup_file = os.path.join(backup_dir, f"pre_clear_backup_{timestamp}.pkl")
			
			os.makedirs(backup_dir, exist_ok=True)
			
			if debug:
				print(f"[DEBUG] Creating backup of current index at: {backup_file}")
			
			# Copy the current index to a backup
			with open(index_path, 'rb') as src, open(backup_file, 'wb') as dst:
				dst.write(src.read())
			
			# Create an empty index file (with no documents)
			empty_documents = []
			with open(index_path, 'wb') as f:
				pickle.dump(empty_documents, f)
			
			print(f"Successfully cleared index for project: {project}")
			print(f"Backup created at: {backup_file}")
			return True
			
		except Exception as e:
			print(f"Error clearing index: {e}")
			if debug:
				print(traceback.format_exc())
			return False

def index_project(project: str, document_dir: str, index_dir: str, 
				  debug: bool = False, auto_adjust_chunks: bool = True,
				  chars_per_dimension: int = DEFAULT_CHARS_PER_DIMENSION, 
				  use_sqlite: bool = False) -> bool:
	"""
	Index a project using document_indexer.py.
	Uses the project's embedding configuration if available.
	
	Returns True if indexing was successful, False otherwise.
	"""
	# First, check if we need to create a configuration file
	config_path = get_project_config_path(project, document_dir)
	
	if not os.path.exists(config_path):
		# Create a default configuration and save it
		default_config = EmbeddingConfig(
			embedding_type=DEFAULT_EMBEDDING_TYPE,
			model_name=DEFAULT_EMBEDDING_MODEL
		)
		save_embedding_config(project, document_dir, default_config)
		print(f"Created default embedding configuration at {config_path}")
	
	# Get the embedding config for the project
	config = get_project_embedding_config(project, document_dir, debug)
	
	# Check for document_indexer.py
	if not os.path.exists("document_indexer.py"):
		print(f"document_indexer.py not found")
		return False
	
	# Build the command
	cmd = [
		sys.executable,  # Current Python interpreter
		"document_indexer.py",
		"--document-dir", document_dir,
		"--index-dir", index_dir,
		"--embedding-type", config.embedding_type,
		"--embedding-model", config.model_name
	]
	
	# Important change: Only pass --project if it's not MASTER_PROJECT
	# This ensures we only index the specified project
	if project != MASTER_PROJECT:
		cmd.extend(["--project", project])
	
	# Add auto-adjust-chunks flag if requested
	if auto_adjust_chunks:
		cmd.append("--auto-adjust-chunks")
		cmd.extend(["--chars-per-dimension", str(chars_per_dimension)])
	
	# Add SQLite flag if requested
	if use_sqlite:
		cmd.append("--use-sqlite")
	
	if debug:
		cmd.append("--debug")
		print(f"[DEBUG] Running indexer with command: {' '.join(cmd)}")
	
	# Run the indexer
	print(f"Indexing project '{project}'...")
	try:
		result = subprocess.run(cmd, check=True)
		print(f"Indexing complete for project '{project}'")
		return True
	except subprocess.CalledProcessError as e:
		print(f"Error indexing project: {e}")
		return False

def save_prompt_to_json(prompt: str, query: str, project: str, 
						 relevant_docs: List[Document], 
						 prompts_dir: str = PROMPTS_DIR, 
						 model: str = None) -> str:
	"""
	Save the prompt and related information to a JSON file.
	Returns the path to the saved file.
	"""
	# Create the prompts directory if it doesn't exist
	os.makedirs(prompts_dir, exist_ok=True)
	
	# Generate timestamp for the filename
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	file_path = os.path.join(prompts_dir, f"{timestamp}.json")
	
	# Use the provided model or the default
	model_used = model or DEFAULT_CLAUDE_MODEL
	
	# Prepare the document context information
	doc_contexts = []
	for i, doc in enumerate(relevant_docs):
		doc_contexts.append({
			"index": i + 1,
			"file_path": doc.metadata.get('file_path', 'Unknown'),
			"project": doc.metadata.get('project', MASTER_PROJECT),
			"chunk_index": doc.metadata.get('chunk_index', 0),
			"total_chunks": doc.metadata.get('total_chunks', 0),
			"embedding_model": doc.metadata.get('embedding_model', 'Unknown'),
			"embedding_type": doc.metadata.get('embedding_type', 'Unknown'),
			"content_length": len(doc.content),
			"content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
		})
	
	# Create the data structure to save
	data = {
		"timestamp": timestamp,
		"query": query,
		"project": project,
		"model": model_used,
		"max_tokens": MAX_TOKENS,
		"num_relevant_docs": len(relevant_docs),
		"document_contexts": doc_contexts,
		"prompt": prompt
	}
	
	# Save to file
	try:
		with open(file_path, 'w') as f:
			json.dump(data, f, indent=2)
		return file_path
	except Exception as e:
		print(f"Failed to save prompt to {file_path}: {e}")
		return ""

def get_full_document_content(file_path, document_dir, debug=False):
	"""
	Retrieve the full content of a document from its file path.
	
	Args:
		file_path: Path to the document file
		document_dir: Base document directory
		debug: Whether to enable debug output
		
	Returns:
		The full content of the document as a string
	"""
	try:
		# Ensure the path is absolute
		if not os.path.isabs(file_path):
			# Try to resolve relative to document_dir
			full_path = os.path.join(document_dir, file_path)
		else:
			full_path = file_path
		
		# Check if the file exists
		if not os.path.exists(full_path):
			if debug:
				print(f"[DEBUG] File not found: {full_path}")
			return f"[File not found: {file_path}]"
			
		# Read the file content
		with open(full_path, 'r', encoding='utf-8') as f:
			content = f.read()
			
		if debug:
			print(f"[DEBUG] Loaded full document: {file_path} ({len(content)} characters)")
			
		return content
	except Exception as e:
		if debug:
			print(f"[DEBUG] Error loading document {file_path}: {e}")
		return f"[Error loading file: {file_path} - {str(e)}]"

def search_documents(query: str, documents: List[Document], project: str, 
					document_dir: str, embedding_config: Optional[EmbeddingConfig] = None,
					top_k: int = TOP_K_DOCUMENTS, debug: bool = False,
					provider_cache: Optional[EmbeddingProviderCache] = None,
					rag_mode: str = "chunk", use_sqlite: bool = False,
					index_dir: str = DEFAULT_INDEX_DIR) -> List[Document]:
	"""
	Search for documents relevant to the query, handling different RAG modes.
	
	Args:
		query: The user's query
		documents: All available documents (ignored when use_sqlite=True)
		project: Current project name
		document_dir: Base document directory
		embedding_config: Optional embedding configuration
		top_k: Number of results to return
		debug: Enable debug output
		provider_cache: Cache for embedding providers
		rag_mode: RAG mode ("chunk", "file", or "none")
		use_sqlite: Whether to use SQLite storage (True) or pickle files (False)
		index_dir: Base index directory (needed for SQLite)
		
	Returns:
		List of relevant documents based on the RAG mode
	"""
	if rag_mode.lower() == "none":
		if debug:
			print(f"[DEBUG] RAG mode is 'none', returning empty document list")
		return []
	
	# Get embedding provider using the cache (or create if not provided)
	temp_cache = False
	if provider_cache is None:
		provider_cache = EmbeddingProviderCache(debug=debug)
		temp_cache = True
	
	provider = provider_cache.get_provider(
		project=project,
		document_dir=document_dir,
		config=embedding_config
	)
	
	# Create query embedding
	if debug:
		print(f"[DEBUG] Creating query embedding...")
	
	start_time = time.time()
	query_embedding = provider.create_embedding(query)
	embedding_time = time.time() - start_time
	
	if debug:
		print(f"[DEBUG] Created query embedding in {embedding_time:.2f} seconds")
		print(f"Created embedding in {embedding_time:.2f}s. Searching documents...")
	
	# Perform the search using the appropriate method
	start_time = time.time()
	if use_sqlite:
		top_results = search_with_sqlite(
			query_embedding, index_dir, project, top_k, debug, rag_mode
		)
	else:
		top_results = search_in_memory(
			query, query_embedding, documents, project, document_dir, 
			embedding_config, top_k, debug, provider_cache, rag_mode
		)
	
	search_time = time.time() - start_time
	if debug:
		print(f"[DEBUG] Search completed in {search_time:.2f} seconds")
	
	# Clean up temporary cache if created
	if temp_cache:
		provider_cache.clear_cache()
	
	# Present search results to the user (both for SQLite and in-memory)
	display_search_results(top_results, project, rag_mode, debug)
	
	return top_results

def search_with_sqlite(query_embedding: List[float], index_dir: str, project: str, 
					  top_k: int, debug: bool, rag_mode: str) -> List[Document]:
	"""Perform document search using SQLite with vector search."""
	try:
		from sqlite_storage import search_similar_documents
		
		# Use SQLite to search for similar documents
		results = search_similar_documents(
			query_embedding, 
			index_dir, 
			project, 
			top_k=top_k,
			debug=debug,
			rag_mode=rag_mode
		)
		
		if debug:
			print(f"[DEBUG] Found {len(results)} relevant documents using SQLite vector search")
			
		return results
			
	except ImportError as e:
		print(f"Error importing sqlite_storage: {e}")
		print(f"Falling back to in-memory search")
		# Return empty list as this will cause the main function to fall back to in-memory
		return []

def search_in_memory(query: str, query_embedding: List[float], documents: List[Document], 
					project: str, document_dir: str, embedding_config: Optional[EmbeddingConfig],
					top_k: int, debug: bool, provider_cache: EmbeddingProviderCache, 
					rag_mode: str) -> List[Document]:
	"""Perform document search using in-memory documents and embeddings."""
	if not documents:
		print("No documents in index")
		return []
	
	if debug:
		print(f"[DEBUG] Searching for: '{query}'")
		print(f"[DEBUG] Using top_k value: {top_k}")
		print(f"[DEBUG] RAG mode: {rag_mode}")
		print(f"[DEBUG] Current project: {project}")
	
	# Try to import tqdm for progress bar
	try:
		from tqdm import tqdm
		has_tqdm = True
	except ImportError:
		has_tqdm = False
		if not debug:
			print("For progress bars, install tqdm: pip install tqdm")
	
	# Group documents by embedding model/type
	document_groups = {}
	for doc in documents:
		# Filter by project if not in master project
		if project != MASTER_PROJECT and doc.metadata.get('project', MASTER_PROJECT) != project:
			continue
			
		embedding_key = (
			doc.metadata.get('embedding_type', 'sentence_transformers'),
			doc.metadata.get('embedding_model', 'all-MiniLM-L6-v2')
		)
		if embedding_key not in document_groups:
			document_groups[embedding_key] = []
		document_groups[embedding_key].append(doc)
	
	# Get base embedding type and model
	base_embedding_type = embedding_config.embedding_type if embedding_config else "sentence_transformers"
	base_embedding_model = embedding_config.model_name if embedding_config else "all-MiniLM-L6-v2"
	base_key = (base_embedding_type, base_embedding_model)
	
	all_results = []
	
	# First, try to search documents with matching embedding type/model
	if base_key in document_groups:
		base_docs = document_groups[base_key]
		
		# Create iterator with progress bar if tqdm is available
		if has_tqdm and not debug and len(base_docs) > 10:
			base_docs_iter = tqdm(base_docs, desc="Searching documents", unit="doc")
		else:
			base_docs_iter = base_docs
		
		# Calculate similarities
		for doc in base_docs_iter:
			if doc.embedding:
				# Calculate cosine similarity
				sim = cosine_similarity([query_embedding], [doc.embedding])[0][0]
				all_results.append((doc, sim))
		
		if debug:
			print(f"[DEBUG] Calculated similarity for {len(all_results)} documents with matching embedding model")
	
	# If we don't have enough results and there are other embedding types
	if len(all_results) < top_k and len(document_groups) > 1:
		print(f"Searching documents with different embedding models...")
		
		# For each different embedding type, we need a new provider
		for key, docs in document_groups.items():
			if key == base_key:
				continue  # Skip the base key we already processed
			
			embedding_type, model_name = key
			if debug:
				print(f"[DEBUG] Searching {len(docs)} documents with embedding type: {embedding_type}, model: {model_name}")
			
			# Create a temporary config for this embedding type
			temp_config = EmbeddingConfig(embedding_type=embedding_type, model_name=model_name)
			
			# Get a provider for this config from the cache
			temp_provider = provider_cache.get_provider(
				project=project,
				document_dir=document_dir,
				config=temp_config
			)
			
			# Create query embedding with this provider
			temp_query_embedding = temp_provider.create_embedding(query)
			
			# Calculate similarities with progress bar
			if has_tqdm and not debug and len(docs) > 10:
				docs_iter = tqdm(docs, desc=f"Searching {model_name}", unit="doc")
			else:
				docs_iter = docs
			
			for doc in docs_iter:
				if doc.embedding:
					sim = cosine_similarity([temp_query_embedding], [doc.embedding])[0][0]
					all_results.append((doc, sim))
	
	# Sort all results by similarity score
	sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
	
	# Handle different RAG modes
	if rag_mode.lower() == "file":
		docs = get_file_mode_results(sorted_results, top_k)
		# Add similarity to metadata
		for i, doc in enumerate(docs):
			# Find the original similarity for this document
			for original_doc, sim in sorted_results:
				if (doc.metadata.get('file_path') == original_doc.metadata.get('file_path') and
					doc.metadata.get('chunk_index') == original_doc.metadata.get('chunk_index')):
					doc.metadata['similarity'] = sim
					break
		return docs
	else:
		# Default "chunk" mode: just get top_k chunks
		docs = []
		for doc, sim in sorted_results[:top_k]:
			# Add similarity to metadata
			doc.metadata['similarity'] = sim
			docs.append(doc)
		return docs

def get_file_mode_results(sorted_results: List[Tuple[Document, float]], top_k: int) -> List[Document]:
	"""Extract all chunks from the top_k distinct files."""
	# Get best score for each file
	file_scores = {}  # {file_path: best_score}
	for doc, sim in sorted_results:
		file_path = doc.metadata.get('file_path', 'unknown')
		if file_path not in file_scores or sim > file_scores[file_path]:
			file_scores[file_path] = sim
	
	# Sort files by their best score
	top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
	top_file_paths = {file_path for file_path, _ in top_files}
	
	# Collect all chunks from the top files, preserving original order within each file
	file_chunks = {}  # {file_path: [(doc, score, chunk_index)]}
	for doc, sim in sorted_results:
		file_path = doc.metadata.get('file_path', 'unknown')
		if file_path in top_file_paths:
			chunk_index = doc.metadata.get('chunk_index', 0)
			if file_path not in file_chunks:
				file_chunks[file_path] = []
			file_chunks[file_path].append((doc, sim, chunk_index))
	
	# Sort chunks within each file by their original order
	results_from_top_files = []
	for file_path in top_file_paths:
		if file_path in file_chunks:
			# Sort by chunk index
			sorted_chunks = sorted(file_chunks[file_path], key=lambda x: x[2])
			# Add to final results
			for doc, sim, _ in sorted_chunks:
				doc.metadata['similarity'] = sim
				results_from_top_files.append(doc)
	
	return results_from_top_files

def display_search_results(results: List[Document], project: str, rag_mode: str, debug: bool) -> None:
	"""Display the search results to the user in a nicely formatted way."""
	if not results:
		print("No relevant documents found in the index.")
		return
	
	# In debug mode, print details about the relevant documents
	if debug:
		print(f"[DEBUG] Found {len(results)} relevant documents:")
		for i, doc in enumerate(results):
			# Get similarity score if available
			sim = doc.metadata.get('similarity', 0)
			proj = doc.metadata.get('project', MASTER_PROJECT)
			file_path = doc.metadata.get('file_path', 'unknown')
			file_name = doc.metadata.get('file_name', 'unknown')
			chunk_index = doc.metadata.get('chunk_index', 0)
			total_chunks = doc.metadata.get('total_chunks', 0)
			emb_model = doc.metadata.get('embedding_model', 'unknown')
			
			print(f"[DEBUG] Document {i+1}:")
			print(f"[DEBUG]   Score: {sim:.4f}")
			print(f"[DEBUG]   Project: {proj}")
			print(f"[DEBUG]   File: {file_path}")
			print(f"[DEBUG]   Chunk: {chunk_index+1}/{total_chunks}")
			print(f"[DEBUG]   Embedding Model: {emb_model}")
			print(f"[DEBUG]   Content Preview: {doc.content[:100]}...")
			print()
		return
	
	# For regular users, show a neat summary of discovered documents
	# Count results by file to show a summary
	file_counts = {}
	for doc in results:
		file_path = doc.metadata.get('file_path', 'unknown')
		file_name = os.path.basename(file_path)
		# Get the similarity score (might be in metadata for SQLite results)
		sim = doc.metadata.get('similarity', 0)
		
		if file_path not in file_counts:
			file_counts[file_path] = {
				'count': 0, 
				'score': sim, 
				'project': doc.metadata.get('project', MASTER_PROJECT),
				'file_name': file_name
			}
		file_counts[file_path]['count'] += 1
		# Keep track of best score for this file
		if sim > file_counts[file_path]['score']:
			file_counts[file_path]['score'] = sim

	# If in file mode, emphasize we're showing entire files
	if rag_mode.lower() == "file":
		print(f"\nFound {len(file_counts)} relevant files:")
	else:
		print(f"\nFound {len(results)} relevant chunks from {len(file_counts)} files:")
	
	# Show files sorted by best score
	sorted_files = sorted(file_counts.items(), key=lambda x: x[1]['score'], reverse=True)
	for i, (file_path, info) in enumerate(sorted_files):
		# Get file name from the path
		file_name = info['file_name']
		
		# Get project if different from current
		doc_project = info['project']
		project_info = f" (project: {doc_project})" if doc_project != project and doc_project != MASTER_PROJECT else ""
		
		# Format the similarity score as percentage
		score_percent = int(info['score'] * 100)
		
		# Show chunk count if in chunk mode
		count_info = f", {info['count']} chunks" if rag_mode.lower() == "chunk" and info['count'] > 1 else ""
		
		# Print a clean summary line
		print(f"  {i+1}. {file_name} - {score_percent}% match{project_info}{count_info}")

def get_model_name_for_llm_type(llm_type: str) -> str:
	"""Get the default model name for a given LLM type."""
	if llm_type == LLM_LOCAL:
		return DEFAULT_LOCAL_MODEL
	elif llm_type == LLM_HF:
		return DEFAULT_HF_MODEL
	elif llm_type == LLM_CLAUDE:
		return DEFAULT_CLAUDE_MODEL
	elif llm_type == LLM_OPENAI:
		return DEFAULT_OPENAI_MODEL
	else:
		return "unknown"

def ask_claude(query: str, relevant_docs: List[Document], api_key: str, project: str, 
			  debug: bool = False, prompts_dir: str = PROMPTS_DIR, 
			  rag_mode: str = "chunk", document_dir: str = DEFAULT_DOCUMENT_DIR,
			  model: str = DEFAULT_CLAUDE_MODEL,
			  system_prompt: str = None) -> str:
	"""Process a user query and return Claude's response, using the specified RAG mode."""
	try:
		client = anthropic.Anthropic(api_key=api_key)
		
		# Use the provided system prompt or a default one
		system_message = system_prompt or "You are a helpful, accurate assistant that helps users find information in their documents."
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		if rag_mode.lower() == "none":
			# RAG mode 'none': Don't use any document context
			prompt = f"""
			User has asked: {query}
			
			Please answer this question based on your general knowledge.
			You should NOT reference any specific documents in your answer.
			"""
			
			if debug:
				print(f"[DEBUG] Using RAG mode 'none' - no document context provided to Claude")
				
		elif not relevant_docs:
			# If no relevant documents found, just ask Claude directly
			prompt = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			Please note that I couldn't find any relevant documents in my knowledge base to help answer this question.
			Please answer based on your general knowledge, and mention that no specific documents were found.
			"""
		elif rag_mode.lower() == "file":
			# RAG mode 'file': Use entire documents instead of chunks
			if debug:
				print(f"[DEBUG] Using RAG mode 'file' - retrieving full documents for {len(relevant_docs)} sources")
				
			# Get distinct document file paths
			distinct_file_paths = set()
			for doc in relevant_docs:
				file_path = doc.metadata.get('file_path', '')
				if file_path and file_path not in distinct_file_paths:
					distinct_file_paths.add(file_path)
			
			# Build context from full documents
			context_pieces = []
			for i, file_path in enumerate(distinct_file_paths):
				# Get full document content
				full_content = get_full_document_content(file_path, document_dir, debug)
				
				# Add document info
				file_name = os.path.basename(file_path)
				context_pieces.append(f"Document {i+1} (Source: {file_path}):\n{full_content}")
			
			context = "\n\n".join(context_pieces)
			
			# Prepare prompt with full document context
			prompt = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			I've retrieved the following complete documents that might help answer this question:
			
			{context}
			
			Please answer the user's question based on the information in these documents.
			If the documents don't contain the necessary information, please say so and answer based on your general knowledge.
			In your answer, cite which documents you used.
			"""
		else:
			# Default RAG mode 'chunk': Use document chunks (original behavior)
			# Build context from relevant document chunks
			context_pieces = []
			for i, doc in enumerate(relevant_docs):
				# Get document metadata
				source = f"{doc.metadata.get('file_path', 'Unknown document')}"
				doc_project = doc.metadata.get('project', MASTER_PROJECT)
				
				# Include chunk info in the source
				chunk_info = f"Chunk {doc.metadata.get('chunk_index', 'unknown')+1}/{doc.metadata.get('total_chunks', 'unknown')}"
				
				# Add paragraph count and project if available
				extra_info = []
				if 'paragraphs' in doc.metadata:
					extra_info.append(f"{doc.metadata.get('paragraphs')} paragraphs")
				if doc_project != MASTER_PROJECT:
					extra_info.append(f"project: {doc_project}")
				
				source_with_info = f"{source} ({chunk_info}, {', '.join(extra_info)})"
				context_pieces.append(f"Document {i+1} (Source: {source_with_info}):\n{doc.content}")
			
			context = "\n\n".join(context_pieces)
			
			# Prepare prompt with context
			prompt = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			I've retrieved the following document chunks that might help answer this question:
			
			{context}
			
			Please answer the user's question based on the information in these documents.
			If the documents don't contain the necessary information, please say so and answer based on your general knowledge.
			In your answer, cite which documents you used.
			"""
		
		if debug:
			print(f"[DEBUG] Sending prompt to Claude model: {model} using RAG mode: {rag_mode}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt, query, project, relevant_docs, prompts_dir)
			if log_path:
				print(f"[DEBUG] Saved prompt to {log_path}")
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		# Get response from Claude
		start_time = time.time()
		response = client.messages.create(
			model=model,
			max_tokens=MAX_TOKENS,
			system=system_message,  # Use system prompt here
			messages=[
				{"role": "user", "content": prompt}
			]
		)
		
		# Cancel the alarm
		signal.alarm(0)
		
		elapsed_time = time.time() - start_time
		if debug:
			print(f"[DEBUG] Received response from Claude ({model}) in {elapsed_time:.2f} seconds")
		
		return response.content[0].text
		
	except APITimeoutError:
		return "I'm sorry, but the request to Claude timed out. Please try again with a simpler question or check your internet connection."
	except Exception as e:
		if debug:
			print(traceback.format_exc())
		return f"I'm sorry, but an error occurred while processing your request: {str(e)}"
	finally:
		# Make sure to cancel the alarm
		signal.alarm(0)

def ask_openai(query: str, relevant_docs: List[Document], api_key: str, project: str, 
			  debug: bool = False, prompts_dir: str = PROMPTS_DIR, 
			  rag_mode: str = "chunk", document_dir: str = DEFAULT_DOCUMENT_DIR,
			  model: str = DEFAULT_OPENAI_MODEL,
			  system_prompt: str = None) -> str:
	"""
	Process a user query and return OpenAI's response, using the specified RAG mode.
	
	Args:
		query: The user's query
		relevant_docs: List of relevant documents
		api_key: OpenAI API key
		project: Current project name
		debug: Enable debug output
		prompts_dir: Directory to save prompts for debugging
		rag_mode: RAG mode to use (chunk, file, or none)
		document_dir: Base document directory
		model: OpenAI model to use
		
	Returns:
		The response from OpenAI
	"""
	try:
		# Import OpenAI here to avoid dependency issues if not used
		from openai import OpenAI
		
		# Create client with API key
		client = OpenAI(api_key=api_key)
		
		# Use the provided system prompt or a default one
		system_message = system_prompt or "You are a helpful, accurate assistant that helps users find information in their documents."
		
		if rag_mode.lower() == "none":
			# RAG mode 'none': Don't use any document context
			prompt = f"""
			User has asked: {query}
			
			Please answer this question based on your general knowledge.
			You should NOT reference any specific documents in your answer.
			"""
			
			if debug:
				print(f"[DEBUG] Using RAG mode 'none' - no document context provided to OpenAI")
				
		elif not relevant_docs:
			# If no relevant documents found, just ask OpenAI directly
			prompt = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			Please note that I couldn't find any relevant documents in my knowledge base to help answer this question.
			Please answer based on your general knowledge, and mention that no specific documents were found.
			"""
		elif rag_mode.lower() == "file":
			# RAG mode 'file': Use entire documents instead of chunks
			if debug:
				print(f"[DEBUG] Using RAG mode 'file' - retrieving full documents for {len(relevant_docs)} sources")
				
			# Get distinct document file paths
			distinct_file_paths = set()
			for doc in relevant_docs:
				file_path = doc.metadata.get('file_path', '')
				if file_path and file_path not in distinct_file_paths:
					distinct_file_paths.add(file_path)
			
			# Build context from full documents
			context_pieces = []
			for i, file_path in enumerate(distinct_file_paths):
				# Get full document content
				full_content = get_full_document_content(file_path, document_dir, debug)
				
				# Add document info
				file_name = os.path.basename(file_path)
				context_pieces.append(f"Document {i+1} (Source: {file_path}):\n{full_content}")
			
			context = "\n\n".join(context_pieces)
			
			# Prepare prompt with full document context
			prompt = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			I've retrieved the following complete documents that might help answer this question:
			
			{context}
			
			Please answer the user's question based on the information in these documents.
			If the documents don't contain the necessary information, please say so and answer based on your general knowledge.
			In your answer, cite which documents you used.
			"""
		else:
			# Default RAG mode 'chunk': Use document chunks (original behavior)
			# Build context from relevant document chunks
			context_pieces = []
			for i, doc in enumerate(relevant_docs):
				# Get document metadata
				source = f"{doc.metadata.get('file_path', 'Unknown document')}"
				doc_project = doc.metadata.get('project', MASTER_PROJECT)
				
				# Include chunk info in the source
				chunk_info = f"Chunk {doc.metadata.get('chunk_index', 'unknown')+1}/{doc.metadata.get('total_chunks', 'unknown')}"
				
				# Add paragraph count and project if available
				extra_info = []
				if 'paragraphs' in doc.metadata:
					extra_info.append(f"{doc.metadata.get('paragraphs')} paragraphs")
				if doc_project != MASTER_PROJECT:
					extra_info.append(f"project: {doc_project}")
				
				source_with_info = f"{source} ({chunk_info}, {', '.join(extra_info)})"
				context_pieces.append(f"Document {i+1} (Source: {source_with_info}):\n{doc.content}")
			
			context = "\n\n".join(context_pieces)
			
			# Prepare prompt with context
			prompt = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			I've retrieved the following document chunks that might help answer this question:
			
			{context}
			
			Please answer the user's question based on the information in these documents.
			If the documents don't contain the necessary information, please say so and answer based on your general knowledge.
			In your answer, cite which documents you used.
			"""
		
		if debug:
			print(f"[DEBUG] Sending prompt to OpenAI model: {model} using RAG mode: {rag_mode}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt, query, project, relevant_docs, prompts_dir, model)
			if log_path:
				print(f"[DEBUG] Saved prompt to {log_path}")
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		# Get response from OpenAI
		start_time = time.time()
		response = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": system_message},  # Use system prompt here
				{"role": "user", "content": prompt}
			],
			max_completion_tokens=MAX_TOKENS
		)
		
		# Cancel the alarm
		signal.alarm(0)
		
		elapsed_time = time.time() - start_time
		if debug:
			print(f"[DEBUG] Received response from OpenAI ({model}) in {elapsed_time:.2f} seconds")
		
		# Extract the response text
		return response.choices[0].message.content
		
	except APITimeoutError:
		return "I'm sorry, but the request to OpenAI timed out. Please try again with a simpler question or check your internet connection."
	except Exception as e:
		if debug:
			print(traceback.format_exc())
		return f"I'm sorry, but an error occurred while processing your request with OpenAI: {str(e)}"
	finally:
		# Make sure to cancel the alarm
		signal.alarm(0)

def ask_local_llm(query: str, relevant_docs: List[Document], project: str, local_model: str = "gpt4all", 
				 debug: bool = False, prompts_dir: str = PROMPTS_DIR, rag_mode: str = "chunk",
				 document_dir: str = DEFAULT_DOCUMENT_DIR,
				 system_prompt: str = None) -> str:
	"""Process a user query using Simon Willison's LLM library."""
	try:
		# Use provided system prompt or a default
		system_message = system_prompt or "You are a helpful assistant that provides accurate information based on the provided documents."

		# Prepare prompt based on RAG mode
		if rag_mode.lower() == "none":
			# No RAG context mode
			prompt_text = f"""
TASK:

Please answer or address the following:

{query}

Answer using your general knowledge. Do not reference any specific documents.
"""
		elif not relevant_docs:
			# If no relevant documents found
			prompt_text = f"""
TASK:

Please read very carefully the documents with the <DOCUMENTS> tag and use them to answer or address the following:

{query}

If you cannot address the question or task using the documents below, do not rely on your general knowledge.

DOCUMENTS:

<documents>
No relevant documents were found in the database for project: {project}.
</documents>

Since no relevant documents were found, please let the user know you don't have specific information on this topic.
"""
		elif rag_mode.lower() == "file":
			# RAG mode 'file': Use complete documents
			distinct_file_paths = set()
			for doc in relevant_docs:
				file_path = doc.metadata.get('file_path', '')
				if file_path and file_path not in distinct_file_paths:
					distinct_file_paths.add(file_path)
			
			# Build context from full documents
			context_pieces = []
			for i, file_path in enumerate(distinct_file_paths):
				# Get full document content
				full_content = get_full_document_content(file_path, document_dir, debug)
				
				# Format the document with its ID and source
				doc_text = f"""<document id={i+1}, source={file_path}>
{full_content}
</document>"""
				context_pieces.append(doc_text)
			
			# Join all document contexts
			documents_context = "\n\n".join(context_pieces)
			
			# Prepare prompt with complete documents
			prompt_text = f"""
TASK:

Please read very carefully the complete documents with the <DOCUMENTS> tag and use them to answer or address the following:

{query}

If you cannot address the question or task using the documents below, do not rely on your general knowledge.

DOCUMENTS:

<documents>
{documents_context}
</documents>

REFERENCING:

Provide a list of the documents you used, and how you made use of them at the end of your answer.
"""
		else:
			# Default RAG mode 'chunk': Use document chunks
			# Build context from relevant documents
			context_pieces = []
			for i, doc in enumerate(relevant_docs):
				# Get document metadata
				file_path = doc.metadata.get('file_path', 'Unknown document')
				
				# Format the document with its ID and source
				doc_text = f"""<document id={i+1}, source={file_path}>
{doc.content}
</document>"""
				context_pieces.append(doc_text)
			
			# Join all document contexts
			documents_context = "\n\n".join(context_pieces)
			
			# Prepare prompt with the new structure
			prompt_text = f"""
TASK:

Please read very carefully the documents with the <DOCUMENTS> tag and use them to answer or address the following:

{query}

If you cannot address the question or task using the documents below, do not rely on your general knowledge.

DOCUMENTS:

<documents>
{documents_context}
</documents>

REFERENCING:

Provide a list of the documents you used, and how you made use of them at the end of your answer.
"""
		
		if debug:
			print(f"[DEBUG] Using Simon Willison's llm with model: {local_model} and RAG mode: {rag_mode}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt_text, query, project, relevant_docs, prompts_dir)
			if log_path:
				print(f"[DEBUG] Saved prompt to {log_path}")
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		try:
			if debug:
				print(f"[DEBUG] Importing llm library...")
				
			import llm
			
			start_time = time.time()
			
			# Try different methods to get models
			model_names = []
			
			# Method 1: Try using llm.get_models()
			try:
				models = llm.get_models()
				if models:
					model_names = [getattr(m, 'model_id', str(m)) for m in models]
					if debug:
						print(f"[DEBUG] Found {len(model_names)} models via llm.get_models()")
			except Exception as e:
				if debug:
					print(f"[DEBUG] Error with llm.get_models(): {e}")
			
			# Method 2: Try using llm.list_models() if available
			if not model_names:
				try:
					if hasattr(llm, 'list_models'):
						model_list = llm.list_models()
						model_names = [str(m) for m in model_list]
						if debug:
							print(f"[DEBUG] Found {len(model_names)} models via llm.list_models()")
				except Exception as e:
					if debug:
						print(f"[DEBUG] Error with llm.list_models(): {e}")
			
			# Method 3: Check if llm-gpt4all plugin is installed and list its models
			if not model_names:
				try:
					# Try to import the specific gpt4all plugin
					import llm_gpt4all
					if hasattr(llm_gpt4all, 'get_models'):
						plugin_models = llm_gpt4all.get_models()
						model_names = [m.model_id for m in plugin_models]
						if debug:
							print(f"[DEBUG] Found {len(model_names)} models via llm_gpt4all plugin")
				except ImportError:
					if debug:
						print(f"[DEBUG] llm_gpt4all plugin not found")
			
			if debug:
				print(f"[DEBUG] Available models: {', '.join(model_names) if model_names else 'None found'}")
			
			# Use the specified model if available, otherwise use the first available model
			if model_names:
				if local_model in model_names:
					try:
						model = llm.get_model(local_model)
						if debug:
							print(f"[DEBUG] Using model: {local_model}")
					except Exception as model_err:
						if debug:
							print(f"[DEBUG] Error loading specified model: {model_err}")
						# Try using the first available model
						try:
							model = llm.get_model(model_names[0])
							if debug:
								print(f"[DEBUG] Using first available model: {model_names[0]}")
						except:
							raise ValueError(f"Could not load any models")
				else:
					try:
						# Use the first available model
						model = llm.get_model(model_names[0])
						if debug:
							print(f"[DEBUG] Model '{local_model}' not found, using '{model_names[0]}' instead")
					except Exception as e:
						if debug:
							print(f"[DEBUG] Error loading first model: {e}")
						raise ValueError(f"Model '{local_model}' not found and couldn't load alternatives")
				
				# Generate response
				try:
					if debug:
						print(f"[DEBUG] Generating response with {model}")
					
					# Different versions of llm have slightly different APIs
					try:
						response = model.prompt(prompt_text, system=system_message)  # Use system prompt here
						result = str(response)
					except AttributeError:
						# Try alternate API style - if it doesn't support system prompts, integrate it into the prompt_text
						if system_message:
							prompt_text = f"SYSTEM INSTRUCTION: {system_message}\n\n{prompt_text}"
						response = model.complete(prompt_text)
						result = response.text()
					
					elapsed_time = time.time() - start_time
					if debug:
						print(f"[DEBUG] Generated response in {elapsed_time:.2f} seconds")
						print(f"[DEBUG] {prompt_text}")
					
					# Cancel the alarm
					signal.alarm(0)
					
					return result
				except Exception as e:
					if debug:
						print(f"[DEBUG] Error generating response: {e}")
					raise
			else:
				if debug:
					print(f"[DEBUG] No models available through llm library")
				raise ValueError("No local models available. Please install one using 'llm install'")
				
		except (ImportError, ModuleNotFoundError) as e:
			if debug:
				print(f"[DEBUG] llm library not found: {e}")
			raise ImportError("Simon Willison's llm library is not installed. Please install it with: pip install llm")
			
	except APITimeoutError:
		return "I'm sorry, but the local LLM timed out. Please try a simpler question or a lighter model."
	except ImportError as ie:
		return str(ie)
	except Exception as e:
		if debug:
			print(traceback.format_exc())
		return f"I'm sorry, but an error occurred while processing your request with the local LLM: {str(e)}"
	finally:
		# Make sure to cancel the alarm
		signal.alarm(0)

def ask_local_hf(query: str, relevant_docs: List[Document], project: str, 
				local_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
				debug: bool = False, prompts_dir: str = PROMPTS_DIR, 
				rag_mode: str = "chunk", document_dir: str = DEFAULT_DOCUMENT_DIR,
				system_prompt: str = None) -> str:
	"""Process a user query using Hugging Face transformers."""
	try:
		# Prepare prompt with system message
		system_message = system_prompt or "You are a helpful assistant that provides accurate information based on the provided documents."
		prompt_text = f"SYSTEM: {system_message}\n\n"

		# Prepare prompt with the same format as for Claude
		if not relevant_docs:
			# If no relevant documents found
			prompt_text = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			Please note that I couldn't find any relevant documents in my knowledge base to help answer this question.
			Please answer based on your general knowledge, and mention that no specific documents were found.
			"""
		else:
			# Build context from relevant documents
			context_pieces = []
			for i, doc in enumerate(relevant_docs):
				# Get document metadata
				source = f"{doc.metadata.get('file_path', 'Unknown document')}"
				doc_project = doc.metadata.get('project', MASTER_PROJECT)
				
				# Include chunk info in the source
				chunk_info = f"Chunk {doc.metadata.get('chunk_index', 'unknown')+1}/{doc.metadata.get('total_chunks', 'unknown')}"
				
				# Add paragraph count and project if available
				extra_info = []
				if 'paragraphs' in doc.metadata:
					extra_info.append(f"{doc.metadata.get('paragraphs')} paragraphs")
				if doc_project != MASTER_PROJECT:
					extra_info.append(f"project: {doc_project}")
				
				source_with_info = f"{source} ({chunk_info}, {', '.join(extra_info)})"
				context_pieces.append(f"Document {i+1} (Source: {source_with_info}):\n{doc.content}")
			
			context = "\n\n".join(context_pieces)
			
			# Prepare prompt with context
			prompt_text = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			I've retrieved the following documents that might help answer this question:
			
			{context}
			
			Please answer the user's question based on the information in these documents.
			If the documents don't contain the necessary information, please say so and answer based on your general knowledge.
			In your answer, cite which documents you used.
			"""
		
		if debug:
			print(f"[DEBUG] Using Hugging Face transformers with model: {local_model}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt_text, query, project, relevant_docs, prompts_dir)
			if log_path:
				print(f"[DEBUG] Saved prompt to {log_path}")
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		try:
			if debug:
				print(f"[DEBUG] Importing transformers...")
			
			from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
			import torch
			
			if debug:
				print(f"[DEBUG] Successfully imported transformers!")
				print(f"[DEBUG] Torch version: {torch.__version__}")
				print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
				print(f"[DEBUG] MPS available: {torch.backends.mps.is_available()}")
			
			start_time = time.time()
			
			# Map some common model shorthand names to their Hugging Face equivalents
			model_map = {
				"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
				"gpt4all": "nomic-ai/gpt4all-j",
				"gpt-j": "EleutherAI/gpt-j-6b",
				"gpt-neo": "EleutherAI/gpt-neo-1.3B",
				"llama2": "meta-llama/Llama-2-7b-chat-hf",
				"mistral": "mistralai/Mistral-7B-v0.1"
			}
			
			# Check if we need to map the model name
			if local_model in model_map:
				hf_model_name = model_map[local_model]
				if debug:
					print(f"[DEBUG] Mapped {local_model} to Hugging Face model: {hf_model_name}")
			else:
				hf_model_name = local_model
			
			if debug:
				print(f"[DEBUG] Loading model: {hf_model_name}")
				
			# Initialize tokenizer 
			tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
			
			# Check device availability for better resource management
			if torch.cuda.is_available():
				device = "cuda"
				dtype = torch.float16
			else:
				device = "cpu"
				dtype = torch.float32
				
			if debug:
				print(f"[DEBUG] Using device: {device}")
				
			# Initialize model with appropriate settings for the available hardware
			model = AutoModelForCausalLM.from_pretrained(
				hf_model_name,
				device_map="auto",
				torch_dtype=dtype,
				low_cpu_mem_usage=True
			)
			
			if debug:
				print(f"[DEBUG] Model loaded successfully")
				
			# Create text generation pipeline
			generator = pipeline(
				"text-generation",
				model=model,
				tokenizer=tokenizer,
				max_length=1024,
				do_sample=True,
				temperature=0.7,
				top_p=0.95
			)
			
			# Generate response
			if debug:
				print(f"[DEBUG] Generating response...")
				
			outputs = generator(prompt_text, max_new_tokens=512, return_full_text=False)
			result = outputs[0]['generated_text']
			
			elapsed_time = time.time() - start_time
			if debug:
				print(f"[DEBUG] Generated response with {hf_model_name} in {elapsed_time:.2f} seconds")
			
			# Cancel the alarm
			signal.alarm(0)
			
			return result
			
		except (ImportError, ModuleNotFoundError) as e:
			if debug:
				print(f"[DEBUG] Transformers library not found: {e}")
			raise ImportError("Hugging Face transformers library is not installed. Please install it with: pip install transformers torch")
			
	except APITimeoutError:
		return "I'm sorry, but the Hugging Face model timed out. Please try a simpler question or a lighter model."
	except ImportError as ie:
		return str(ie)
	except Exception as e:
		if debug:
			print(traceback.format_exc())
		return f"I'm sorry, but an error occurred while processing your request with the Hugging Face model: {str(e)}"
	finally:
		# Make sure to cancel the alarm
		signal.alarm(0)

def get_response(query: str, relevant_docs: List[Document], api_key: str, project: str, 
				llm_type: str = LLM_CLAUDE, model_name: str = None,
				debug: bool = False, prompts_dir: str = PROMPTS_DIR,
				rag_mode: str = "chunk", document_dir: str = DEFAULT_DOCUMENT_DIR,
				system_prompt: str = None) -> str:
	"""
	Get a response using the selected LLM.
	"""
	if llm_type.lower() == LLM_CLAUDE:
		claude_model = model_name or DEFAULT_CLAUDE_MODEL
		if debug:
			print(f"[DEBUG] Using Claude API model: {claude_model} for response with RAG mode: {rag_mode}")
			if system_prompt:
				print(f"[DEBUG] Using system prompt: \"{system_prompt}\"")
		return ask_claude(query, relevant_docs, api_key, project, debug, prompts_dir, rag_mode, document_dir, claude_model, system_prompt)
	
	elif llm_type.lower() == LLM_OPENAI:
		openai_model = model_name or DEFAULT_OPENAI_MODEL
		if debug:
			print(f"[DEBUG] Using OpenAI API model: {openai_model} for response with RAG mode: {rag_mode}")
			if system_prompt:
				print(f"[DEBUG] Using system prompt: \"{system_prompt}\"")
		# For OpenAI, use the OPENAI_API_KEY environment variable if not provided
		openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
		if not openai_api_key:
			return "OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or provide it via --api-key."
		return ask_openai(query, relevant_docs, openai_api_key, project, debug, prompts_dir, rag_mode, document_dir, openai_model, system_prompt)
	
	elif llm_type.lower() == LLM_LOCAL:
		local_model = model_name or DEFAULT_LOCAL_MODEL
		if debug:
			print(f"[DEBUG] Using Simon Willison's llm with model: {local_model} and RAG mode: {rag_mode}")
			if system_prompt:
				print(f"[DEBUG] Using system prompt: \"{system_prompt}\"")
		try:
			return ask_local_llm(query, relevant_docs, project, local_model, debug, prompts_dir, rag_mode, document_dir, system_prompt)
		except Exception as e:
			if debug:
				print(f"[DEBUG] Error with local llm: {e}, falling back to Hugging Face")
			# If Simon's llm fails, fall back to Hugging Face
			return ask_local_hf(query, relevant_docs, project, DEFAULT_HF_MODEL, debug, prompts_dir, rag_mode, document_dir, system_prompt)
	
	elif llm_type.lower() == LLM_HF:
		hf_model = model_name or DEFAULT_HF_MODEL
		if debug:
			print(f"[DEBUG] Using Hugging Face transformers with model: {hf_model} and RAG mode: {rag_mode}")
			if system_prompt:
				print(f"[DEBUG] Using system prompt: \"{system_prompt}\"")
		return ask_local_hf(query, relevant_docs, project, hf_model, debug, prompts_dir, rag_mode, document_dir, system_prompt)
	
	else:
		print(f"Unknown LLM type: {llm_type}. Using Claude as fallback.")
		return ask_claude(query, relevant_docs, api_key, project, debug, prompts_dir, rag_mode, document_dir, DEFAULT_CLAUDE_MODEL, system_prompt)

def is_command(text: str) -> bool:
	"""Check if the input is a command rather than a question."""
	command_prefixes = [
		"help", "project ", "projects", "config", 
		"index", "index clear",
		"history", "history clear", "history save", 
		"rag mode ", "rag count ",
		"system prompt", "system prompt show", "system prompt clear",
		"defaults save", "defaults read",
		"exit", "quit", "llm ", "models"
	]
	return any(text.lower() == cmd or text.lower().startswith(cmd) for cmd in command_prefixes)