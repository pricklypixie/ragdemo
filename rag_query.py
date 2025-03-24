#!/usr/bin/env python3
"""
RAG Query Application with Project Support and Colorful Interface

This application:
1. Loads document indexes created by the document_indexer
2. Supports querying specific projects or the master index
3. Retrieves relevant documents based on the query
4. Sends the query and context to Claude for answering
5. Supports on-demand indexing of projects
6. Features a colorful terminal interface
7. Logs prompts to JSON files when in debug mode
"""

import os
import sys
import json
import argparse
import glob
import pickle
import numpy as np
import time
import signal
import traceback
import subprocess
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import atexit
import json

try:
	import readline  # For Unix/Linux/Mac
except ImportError:
	try:
		import pyreadline3 as readline  # For Windows
	except ImportError:
		# Readline not available
		pass



import llm

# Force CPU usage instead of Metal on MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_FALLBACK_POLICY"] = "0" 

# # Default models
# DEFAULT_LLM = "local"  # Default to Claude
# LOCAL_MODEL = "orca-2-7b"  # Default local model, can be changed via CLI



import anthropic
from sklearn.metrics.pairwise import cosine_similarity

# Import colorama for terminal colors
try:
	from colorama import init, Fore, Style
	# Initialize colorama
	init(autoreset=True)
	COLORS_AVAILABLE = True
except ImportError:
	print("For a colorful interface, install colorama: pip install colorama")
	# Create dummy color constants
	class DummyFore:
		def __getattr__(self, name):
			return ""
	class DummyStyle:
		def __getattr__(self, name):
			return ""
	Fore = DummyFore()
	Style = DummyStyle()
	COLORS_AVAILABLE = False

# Import our embedding library
from embeddings import EmbeddingConfig, get_embedding_provider, load_project_config

# Constants
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


DEFAULT_CLAUDE_MODEL = CLAUDE_HAIKU  # Change default to Haiku
MAX_TOKENS = 8096
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_TYPE = "sentence_transformers"
TOP_K_DOCUMENTS = 3
API_TIMEOUT = 120  # Timeout for API calls in seconds
MASTER_PROJECT = "master"  # Name for the master index
PROMPTS_DIR = "prompts"  # Directory to save prompt logs


# Make sure this is the same here and in document_indexer.py
DEFAULT_CHARS_PER_DIMENSION = 4

# For the different models
# LLM types
LLM_CLAUDE = "claude"
LLM_LOCAL = "local"
LLM_HF = "hf"

# Default models
DEFAULT_LLM_TYPE = LLM_LOCAL
DEFAULT_LOCAL_MODEL = "mistral-7b-instruct-v0"
DEFAULT_HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# Color scheme
QUERY_COLOR = Fore.GREEN
ANSWER_COLOR = Fore.CYAN
DEBUG_COLOR = Fore.YELLOW
ERROR_COLOR = Fore.RED
SYSTEM_COLOR = Fore.MAGENTA
HIGHLIGHT_COLOR = Fore.WHITE + Style.BRIGHT
RESET_COLOR = Style.RESET_ALL


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


class CommandHistory:
		"""Manages command history for interactive mode."""
		
		def __init__(self, history_dir="history", max_size=1000):
			"""
			Initialize command history manager.
			
			Args:
				history_dir: Directory to store history files
				max_size: Maximum number of commands to store in history
			"""
			self.history_dir = history_dir
			self.max_size = max_size
			self.entries = []  # List of (command, is_query) tuples
			
			# Create history directory if it doesn't exist
			os.makedirs(history_dir, exist_ok=True)
			
			# Set up readline
			try:
				# Set up readline history
				self.history_file = os.path.join(os.path.expanduser("~"), ".rag_query_history")
				
				# Read existing history if it exists
				try:
					readline.read_history_file(self.history_file)
				except FileNotFoundError:
					pass
					
				# Save history on exit
				atexit.register(readline.write_history_file, self.history_file)
				
				# Set history length
				readline.set_history_length(max_size)
				
				self.readline_supported = True
			except (ImportError, AttributeError):
				self.readline_supported = False
		
		def add(self, command, is_query=True):
			"""
			Add a command to history.
			
			Args:
				command: The command or query string
				is_query: Whether this is a query (True) or a command (False)
			"""
			self.entries.append((command, is_query))
			if len(self.entries) > self.max_size:
				self.entries.pop(0)  # Remove oldest entry if we exceed max size
		
		def clear(self):
			"""Clear the command history."""
			self.entries = []
			
			# Clear readline history if supported
			if self.readline_supported:
				for i in range(readline.get_current_history_length()):
					readline.remove_history_item(0)
		
		def save(self, filename=None):
			"""
			Save history to a JSON file.
			
			Args:
				filename: Optional filename, if None uses timestamp
				
			Returns:
				Path to the saved file
			"""
			if not filename:
				timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
				filename = f"history_{timestamp}.json"
			
			filepath = os.path.join(self.history_dir, filename)
			
			# Convert history to a serializable format
			history_data = {
				"timestamp": datetime.now().isoformat(),
				"count": len(self.entries),
				"entries": [
					{
						"command": cmd,
						"type": "query" if is_query else "command"
					}
					for cmd, is_query in self.entries
				]
			}
			
			# Save to file
			try:
				with open(filepath, 'w') as f:
					json.dump(history_data, f, indent=2)
				return filepath
			except Exception as e:
				print_error(f"Failed to save history: {e}")
				return None
		
		def get_entries(self):
			"""Get all history entries."""
			return self.entries
		
		def get_last_n(self, n=10):
			"""Get the last n history entries."""
			return self.entries[-n:] if n <= len(self.entries) else self.entries




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
						print_debug(f"Loaded embedding config from: {config_path}")
				except Exception as e:
					if self.debug:
						print_debug(f"Error loading config from {config_path}: {e}")
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
				print_debug(f"Using cached embedding provider for {cache_key} (hits: {self.hits}, misses: {self.misses})")
			return self.providers[cache_key]
		
		# Create a new provider
		if self.debug:
			self.misses += 1
			print_debug(f"Creating new embedding provider for {cache_key} (hits: {self.hits}, misses: {self.misses})")
		
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
			print_debug(f"Pre-loading model for {config.embedding_type}/{config.model_name}")
		
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
			print_debug(f"Cleared embedding provider cache ({old_count} providers removed)")



def print_debug(message: str) -> None:
	"""Print debug message in debug color."""
	print(f"{DEBUG_COLOR}[DEBUG] {message}{RESET_COLOR}")


def print_error(message: str) -> None:
	"""Print error message in error color."""
	print(f"{ERROR_COLOR}Error: {message}{RESET_COLOR}")


def print_system(message: str) -> None:
	"""Print system message in system color."""
	print(f"{SYSTEM_COLOR}{message}{RESET_COLOR}")


def save_prompt_to_json(prompt: str, query: str, project: str, relevant_docs: List[Document], prompts_dir: str = PROMPTS_DIR, model: str = None) -> str:
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
		print_error(f"Failed to save prompt to {file_path}: {e}")
		return ""
		
		
		
				
def timeout_handler(signum, frame):
	"""Signal handler for timeouts."""
	raise APITimeoutError("API call timed out")


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




# Add these functions to rag_query.py, replacing the existing functions for project config

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
			print_error(f"Error loading project config: {e}")
	
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
			print_error(f"Error loading legacy embedding config: {e}")
	
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
		print_debug(f"Loaded project config for {project}")
		print_debug(f"Indexing config: {indexing_config}")
	
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
	
	print_system(f"Saved project configuration to {config_path}")














def discover_projects(index_dir: str) -> List[str]:
	"""Discover all indexed projects in the index directory."""
	projects = []
	
	# Check for master index
	master_path = os.path.join(index_dir, "document_index.pkl")
	if os.path.exists(master_path):
		projects.append(MASTER_PROJECT)
	
	# Look for project subdirectories with indexes
	try:
		for item in os.listdir(index_dir):
			item_path = os.path.join(index_dir, item)
			if os.path.isdir(item_path) and item != "backups":
				# Check if this directory has an index file
				if os.path.exists(os.path.join(item_path, "document_index.pkl")):
					projects.append(item)
	except Exception as e:
		print_error(f"Error discovering projects: {e}")
	
	return projects


def load_index(index_path: str, backup_dir: str, debug: bool = False) -> List[Document]:
	"""Load the document index from disk."""
	try:
		with open(index_path, 'rb') as f:
			documents = pickle.load(f)
		print_system(f"Loaded {len(documents)} documents from index: {index_path}")
		return documents
	except Exception as e:
		print_error(f"Error loading index: {e}")
		# Try to load from backup if main index fails
		backup_files = sorted(glob.glob(os.path.join(backup_dir, "*.pkl")), reverse=True)
		if backup_files:
			print_system(f"Attempting to load from latest backup: {backup_files[0]}")
			try:
				with open(backup_files[0], 'rb') as f:
					documents = pickle.load(f)
				print_system(f"Loaded {len(documents)} documents from backup")
				return documents
			except Exception as backup_error:
				print_error(f"Error loading backup: {backup_error}")
		
		return []

def clear_index(project: str, index_dir: str, debug: bool = False) -> bool:
		"""
		Clear (erase) the index for a specific project.
		
		Args:
			project: The project name to clear
			index_dir: Base index directory
			debug: Enable debug logging
			
		Returns:
			True if successful, False otherwise
		"""
		index_path, backup_dir = get_index_path(index_dir, project)
		
		if not os.path.exists(index_path):
			print_error(f"Index file not found: {index_path}")
			return False
		
		try:
			# Create a backup before deletion
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			backup_file = os.path.join(backup_dir, f"pre_clear_backup_{timestamp}.pkl")
			
			os.makedirs(backup_dir, exist_ok=True)
			
			if debug:
				print_debug(f"Creating backup of current index at: {backup_file}")
			
			# Copy the current index to a backup
			with open(index_path, 'rb') as src, open(backup_file, 'wb') as dst:
				dst.write(src.read())
			
			# Create an empty index file (with no documents)
			empty_documents = []
			with open(index_path, 'wb') as f:
				pickle.dump(empty_documents, f)
			
			print_system(f"Successfully cleared index for project: {HIGHLIGHT_COLOR}{project}{RESET_COLOR}")
			print_system(f"Backup created at: {backup_file}")
			return True
			
		except Exception as e:
			print_error(f"Error clearing index: {e}")
			if debug:
				print(traceback.format_exc())
			return False
			
			
def get_project_embedding_config(project: str, document_dir: str, debug: bool = False) -> EmbeddingConfig:
	"""Get the embedding configuration for a project, loading from project config if available."""
	config_path = get_project_config_path(project, document_dir)
	
	if os.path.exists(config_path):
		try:
			config = EmbeddingConfig.from_json_file(config_path)
			if debug:
				print_debug(f"Loaded project embedding config from: {config_path}")
				print_debug(f"Embedding type: {config.embedding_type}")
				print_debug(f"Embedding model: {config.model_name}")
				print_debug(f"Embedding dimensions: {config.dimensions}")

			return config
		except Exception as e:
			print_error(f"Error loading project config, using defaults: {e}")
	else:
		if debug:
			print_debug(f"No project config found at {config_path}, using defaults")
	
	# Use defaults if no config or loading failed
	return EmbeddingConfig(
		embedding_type=DEFAULT_EMBEDDING_TYPE,
		model_name=DEFAULT_EMBEDDING_MODEL
	)




def index_project(project: str, document_dir: str, index_dir: str, 
			 debug: bool = False, auto_adjust_chunks: bool = True,
			 chars_per_dimension: int = 4) -> bool:
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
		print_system(f"Created default embedding configuration at {config_path}")
	
	# Get the embedding config for the project
	config = get_project_embedding_config(project, document_dir, debug)
	
	# Check for document_indexer.py
	if not os.path.exists("document_indexer.py"):
		print_error("document_indexer.py not found")
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
	
	if debug:
		cmd.append("--debug")
		print_debug(f"Running indexer with command: {' '.join(cmd)}")
	
	# Run the indexer
	print_system(f"Indexing project '{project}'...")
	try:
		result = subprocess.run(cmd, check=True)
		print_system(f"Indexing complete for project '{project}'")
		return True
	except subprocess.CalledProcessError as e:
		print_error(f"Error indexing project: {e}")
		return False




def search_documents(query: str, documents: List[Document], project: str, 
		   document_dir: str, embedding_config: Optional[EmbeddingConfig] = None,
		   top_k: int = TOP_K_DOCUMENTS, debug: bool = False,
		   provider_cache: Optional[EmbeddingProviderCache] = None,
		   rag_mode: str = "chunk") -> List[Document]:
	"""
	Search for documents relevant to the query, handling different RAG modes.
	
	Args:
		query: The user's query
		documents: All available documents
		project: Current project name
		document_dir: Base document directory
		embedding_config: Optional embedding configuration
		top_k: Number of results to return
		debug: Enable debug output
		provider_cache: Cache for embedding providers
		rag_mode: RAG mode ("chunk", "file", or "none")
		
	Returns:
		List of relevant documents based on the RAG mode
	"""
	if not documents:
		print_system("No documents in index")
		return []
	
	if debug:
		print_debug(f"Searching for: '{query}'")
		print_debug(f"Using top_k value: {top_k}")
		print_debug(f"RAG mode: {rag_mode}")
		print_debug(f"Current project: {project}")
	
	# Try to import tqdm for progress bar
	try:
		from tqdm import tqdm
		has_tqdm = True
	except ImportError:
		has_tqdm = False
		if not debug:
			print_system("For progress bars, install tqdm: pip install tqdm")
	
	# Handle master project - include documents from all projects
	if project == MASTER_PROJECT:
		if debug:
			print_debug("Searching across all projects (master project)")
		# Documents already contain all projects when using master
	else:
		if debug:
			print_debug(f"Searching only in project: {project}")
			
			# Count how many documents we have from this project vs. total
			project_docs = [doc for doc in documents if doc.metadata.get('project', MASTER_PROJECT) == project]
			if debug:
				print_debug(f"Found {len(project_docs)} documents in project {project} out of {len(documents)} total documents")
			
			# If we're not in the master project, we might want to filter to only this project's documents
			# This depends on your application design - if you want to search only within the current project
			# Uncomment the next line:
			# documents = project_docs
	
	# Group documents by embedding model/type
	document_groups = {}
	for doc in documents:
		embedding_key = (
			doc.metadata.get('embedding_type', 'sentence_transformers'),
			doc.metadata.get('embedding_model', 'all-MiniLM-L6-v2')
		)
		if embedding_key not in document_groups:
			document_groups[embedding_key] = []
		document_groups[embedding_key].append(doc)
	
	# Use the provider cache if provided, or create a temporary one
	temp_cache = False
	if provider_cache is None:
		provider_cache = EmbeddingProviderCache(debug=debug)
		temp_cache = True
	
	# Get embedding provider using the cache
	embedding_provider = provider_cache.get_provider(
		project=project,
		document_dir=document_dir,
		config=embedding_config
	)
	
	# Get base embedding type and model
	base_embedding_type = embedding_provider.config.embedding_type
	base_embedding_model = embedding_provider.config.model_name
	base_key = (base_embedding_type, base_embedding_model)
	
	all_results = []
	
	try:
		# First, try to search documents with matching embedding type/model
		if base_key in document_groups:
			start_time = time.time()
			
			# Create query embedding
			print_system("Creating query embedding...")
			query_embedding = embedding_provider.create_embedding(query)
			search_time = time.time() - start_time
			
			if debug:
				print_debug(f"Created query embedding in {search_time:.2f} seconds")
				print_debug(f"Searching {len(document_groups[base_key])} documents with matching embedding model")
			else:
				print_system(f"Created embedding in {search_time:.2f}s. Calculating document similarity...")
			
			# Calculate similarities for the base model group with progress bar
			base_docs = document_groups[base_key]
			
			# Create iterator with progress bar if tqdm is available
			if has_tqdm and not debug and len(base_docs) > 10:
				base_docs_iter = tqdm(base_docs, desc="Searching documents", unit="doc")
			else:
				base_docs_iter = base_docs
			
			base_similarities = []
			for doc in base_docs_iter:
				if doc.embedding:
					# Calculate cosine similarity
					sim = cosine_similarity(
						[query_embedding], 
						[doc.embedding]
					)[0][0]
					base_similarities.append((doc, sim))
			
			all_results.extend(base_similarities)
			
			# If we don't have enough results and there are other embedding types
			if len(base_similarities) < top_k and len(document_groups) > 1:
				print_system(f"Searching documents with different embedding models...")
				
				# For each different embedding type, we need a new provider
				for key, docs in document_groups.items():
					if key == base_key:
						continue  # Skip the base key we already processed
					
					embedding_type, model_name = key
					if debug:
						print_debug(f"Searching {len(docs)} documents with embedding type: {embedding_type}, model: {model_name}")
					
					# Create a temporary config for this embedding type
					temp_config = EmbeddingConfig(embedding_type=embedding_type, model_name=model_name)
					
					# Get a provider for this config from the cache
					temp_provider = provider_cache.get_provider(
						project=project,
						document_dir=document_dir,
						config=temp_config
					)
					
					# Create query embedding with this provider
					start_time = time.time()
					temp_query_embedding = temp_provider.create_embedding(query)
					search_time = time.time() - start_time
					
					if debug:
						print_debug(f"Created query embedding with {model_name} in {search_time:.2f} seconds")
					
					# Calculate similarities with progress bar
					if has_tqdm and not debug and len(docs) > 10:
						docs_iter = tqdm(docs, desc=f"Searching {model_name}", unit="doc")
					else:
						docs_iter = docs
					
					other_similarities = []
					for doc in docs_iter:
						if doc.embedding:
							sim = cosine_similarity(
								[temp_query_embedding], 
								[doc.embedding]
							)[0][0]
							other_similarities.append((doc, sim))
					
					all_results.extend(other_similarities)
		
		# If temporary cache was created, clear it to free memory
		if temp_cache:
			provider_cache.clear_cache()
			
		# Sort all results by similarity score
		sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
		
		# Different handling based on RAG mode
		if rag_mode.lower() == "file":
			# For file mode: extract top_k distinct files, but return all chunks from those files
			distinct_files = set()
			file_scores = {}  # {file_path: best_score}
			
			# First pass: get best score for each file
			for doc, sim in sorted_results:
				file_path = doc.metadata.get('file_path', 'unknown')
				if file_path not in file_scores or sim > file_scores[file_path]:
					file_scores[file_path] = sim
			
			# Sort files by their best score
			top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
			
			if debug:
				print_debug(f"Top {len(top_files)} files by similarity score:")
				for file_path, score in top_files:
					print_debug(f"  {file_path}: {score:.4f}")
			
			# Set of top file paths
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
						results_from_top_files.append((doc, sim))
			
			# This will have all chunks from the top_k files
			top_distinct_results = results_from_top_files
			
		else:
			# Default "chunk" mode: just get top_k chunks regardless of file
			top_distinct_results = sorted_results[:top_k]
		
		# Just get the documents without the scores
		top_results = [doc for doc, sim in top_distinct_results]
		
		# In debug mode, print details about the relevant documents
		if debug:
			print_debug(f"Found {len(top_results)} relevant documents:")
			for i, (doc, sim) in enumerate(top_distinct_results):
				proj = doc.metadata.get('project', MASTER_PROJECT)
				file_path = doc.metadata.get('file_path', 'unknown')
				file_name = doc.metadata.get('file_name', 'unknown')
				chunk_index = doc.metadata.get('chunk_index', 0)
				total_chunks = doc.metadata.get('total_chunks', 0)
				emb_model = doc.metadata.get('embedding_model', 'unknown')
				
				print_debug(f"Document {i+1}:")
				print_debug(f"  Score: {sim:.4f}")
				print_debug(f"  Project: {proj}")
				print_debug(f"  File: {file_path}")
				print_debug(f"  Chunk: {chunk_index+1}/{total_chunks}")
				print_debug(f"  Embedding Model: {emb_model}")
				print_debug(f"  Content Preview: {doc.content[:100]}...")
				print()
		else:
			# For regular users, show a neat summary of discovered documents
			if top_results:
				# Count results by file to show a summary
				file_counts = {}
				for doc, sim in top_distinct_results:
					file_path = doc.metadata.get('file_path', 'unknown')
					file_name = os.path.basename(file_path)
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
					print_system(f"\nFound {len(file_counts)} relevant files:")
				else:
					print_system(f"\nFound {len(top_results)} relevant chunks from {len(file_counts)} files:")
				
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
					print_system(f"  {i+1}. {HIGHLIGHT_COLOR}{file_name}{RESET_COLOR}{SYSTEM_COLOR} - {score_percent}% match{project_info}{count_info}")
			else:
				print_system("No relevant documents found in the index.")
		
		return top_results
	
	except Exception as e:
		print_error(f"Error during search: {e}")
		if debug:
			print(traceback.format_exc())
		return []
	
	
	
	
	


# Function to handle retrieving full documents instead of chunks
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
				print_debug(f"File not found: {full_path}")
			return f"[File not found: {file_path}]"
			
		# Read the file content
		with open(full_path, 'r', encoding='utf-8') as f:
			content = f.read()
			
		if debug:
			print_debug(f"Loaded full document: {file_path} ({len(content)} characters)")
			
		return content
	except Exception as e:
		if debug:
			print_debug(f"Error loading document {file_path}: {e}")
		return f"[Error loading file: {file_path} - {str(e)}]"	
	


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
				print_debug("Using RAG mode 'none' - no document context provided to OpenAI")
				
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
				print_debug(f"Using RAG mode 'file' - retrieving full documents for {len(relevant_docs)} sources")
				
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
			print_debug(f"Sending prompt to OpenAI model: {model} using RAG mode: {rag_mode}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt, query, project, relevant_docs, prompts_dir, model)
			if log_path:
				print_debug(f"Saved prompt to {log_path}")
		
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
			print_debug(f"Received response from OpenAI ({model}) in {elapsed_time:.2f} seconds")
		
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






# Modify the ask_claude function to use the system prompt
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
		
		# Rest of the function remains the same...
		
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
				print_debug("Using RAG mode 'none' - no document context provided to Claude")
				
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
				print_debug(f"Using RAG mode 'file' - retrieving full documents for {len(relevant_docs)} sources")
				
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
			print_debug(f"Sending prompt to Claude model: {model} using RAG mode: {rag_mode}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt, query, project, relevant_docs, prompts_dir)
			if log_path:
				print_debug(f"Saved prompt to {log_path}")
		
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
			print_debug(f"Received response from Claude ({model}) in {elapsed_time:.2f} seconds")
		
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






# to use Simon Willison's LLM library
def ask_local_llm(query: str, relevant_docs: List[Document], project: str, local_model: str = "gpt4all", 
		debug: bool = False, prompts_dir: str = PROMPTS_DIR, rag_mode: str = "chunk",
		document_dir: str = DEFAULT_DOCUMENT_DIR,
		system_prompt: str = None) -> str:
	
	"""Process a user query using Simon Willison's LLM library.
	using the specified RAG mode.
	
	Args:
		query: The user's query
		relevant_docs: List of relevant documents
		project: Project name
		local_model: Name of the local model to use
		debug: Enable debug output
		prompts_dir: Directory to save prompts for debugging
		rag_mode: RAG mode to use ('chunk', 'file', or 'none')
		document_dir: Base document directory (needed for file mode)
		
	Returns:
		The LLM's response as a string
	"""
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
			print_debug(f"Using Simon Willison's llm with model: {local_model} and RAG mode: {rag_mode}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt_text, query, project, relevant_docs, prompts_dir)
			if log_path:
				print_debug(f"Saved prompt to {log_path}")
		
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		try:
			if debug:
				print_debug("Importing llm library...")
				
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
						print_debug(f"Found {len(model_names)} models via llm.get_models()")
			except Exception as e:
				if debug:
					print_debug(f"Error with llm.get_models(): {e}")
			
			# Method 2: Try using llm.list_models() if available
			if not model_names:
				try:
					if hasattr(llm, 'list_models'):
						model_list = llm.list_models()
						model_names = [str(m) for m in model_list]
						if debug:
							print_debug(f"Found {len(model_names)} models via llm.list_models()")
				except Exception as e:
					if debug:
						print_debug(f"Error with llm.list_models(): {e}")
			
			# Method 3: Check if llm-gpt4all plugin is installed and list its models
			if not model_names:
				try:
					# Try to import the specific gpt4all plugin
					import llm_gpt4all
					if hasattr(llm_gpt4all, 'get_models'):
						plugin_models = llm_gpt4all.get_models()
						model_names = [m.model_id for m in plugin_models]
						if debug:
							print_debug(f"Found {len(model_names)} models via llm_gpt4all plugin")
				except ImportError:
					if debug:
						print_debug("llm_gpt4all plugin not found")
			
			if debug:
				print_debug(f"Available models: {', '.join(model_names) if model_names else 'None found'}")
			
			# Use the specified model if available, otherwise use the first available model
			if model_names:
				if local_model in model_names:
					try:
						model = llm.get_model(local_model)
						if debug:
							print_debug(f"Using model: {local_model}")
					except Exception as model_err:
						if debug:
							print_debug(f"Error loading specified model: {model_err}")
						# Try using the first available model
						try:
							model = llm.get_model(model_names[0])
							if debug:
								print_debug(f"Using first available model: {model_names[0]}")
						except:
							raise ValueError(f"Could not load any models")
				else:
					try:
						# Use the first available model
						model = llm.get_model(model_names[0])
						if debug:
							print_debug(f"Model '{local_model}' not found, using '{model_names[0]}' instead")
					except Exception as e:
						if debug:
							print_debug(f"Error loading first model: {e}")
						raise ValueError(f"Model '{local_model}' not found and couldn't load alternatives")
				
				# Generate response
				try:
					if debug:
						print_debug(f"Generating response with {model}")
					
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
						print_debug(f"Generated response in {elapsed_time:.2f} seconds")
						print_debug(prompt_text)

					
					# Cancel the alarm
					signal.alarm(0)
					
					return result
				except Exception as e:
					if debug:
						print_debug(f"Error generating response: {e}")
					raise
			else:
				if debug:
					print_debug("No models available through llm library")
				raise ValueError("No local models available. Please install one using 'llm install'")
				
		except (ImportError, ModuleNotFoundError) as e:
			if debug:
				print_debug(f"llm library not found: {e}")
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


def ask_local_hf(query: str, relevant_docs: List[Document], project: str, local_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
		debug: bool = False, prompts_dir: str = PROMPTS_DIR, rag_mode: str = "chunk",
		document_dir: str = DEFAULT_DOCUMENT_DIR,
		system_prompt: str = None) -> str:
	"""Process a user query using Hugging Face transformers.
	Args:
		query: The user's query
		relevant_docs: List of relevant documents
		project: Project name
		local_model: Name of the Hugging Face model to use
		debug: Enable debug output
		prompts_dir: Directory to save prompts for debugging
		
	Returns:
		The LLM's response as a string
	"""
	try:
		
		# Prepare prompt with system message
		if system_message:
			prompt_text = f"SYSTEM: {system_message}\n\n"
		else:
			prompt_text = "You are a helpful assistant that provides accurate information based on the provided documents."


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
			print_debug(f"Using Hugging Face transformers with model: {local_model}")
			
			# Save the prompt to a JSON file
			log_path = save_prompt_to_json(prompt_text, query, project, relevant_docs, prompts_dir)
			if log_path:
				print_debug(f"Saved prompt to {log_path}")
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		try:
			if debug:
				print_debug("Importing transformers...")
			
			from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
			import torch
			
			if debug:
				print_debug("Successfully imported transformers!")
				print_debug(f"Torch version: {torch.__version__}")
				print_debug(f"CUDA available: {torch.cuda.is_available()}")
				print_debug(f"MPS available: {torch.backends.mps.is_available()}")
			
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
					print_debug(f"Mapped {local_model} to Hugging Face model: {hf_model_name}")
			else:
				hf_model_name = local_model
			
			if debug:
				print_debug(f"Loading model: {hf_model_name}")
				
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
				print_debug(f"Using device: {device}")
				
			# Initialize model with appropriate settings for the available hardware
			model = AutoModelForCausalLM.from_pretrained(
				hf_model_name,
				device_map="auto",
				torch_dtype=dtype,
				low_cpu_mem_usage=True
			)
			
			if debug:
				print_debug(f"Model loaded successfully")
				
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
				print_debug(f"Generating response...")
				
			outputs = generator(prompt_text, max_new_tokens=512, return_full_text=False)
			result = outputs[0]['generated_text']
			
			elapsed_time = time.time() - start_time
			if debug:
				print_debug(f"Generated response with {hf_model_name} in {elapsed_time:.2f} seconds")
			
			# Cancel the alarm
			signal.alarm(0)
			
			return result
			
		except (ImportError, ModuleNotFoundError) as e:
			if debug:
				print_debug(f"Transformers library not found: {e}")
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

	
def is_command(text: str) -> bool:
	"""Check if the input is a command rather than a question."""
	command_prefixes = [
		"help", "project ", "projects", "config", 
		"index", "index clear",
		"history", "history clear", "history save", 
		"rag mode ", "rag count ",
		"system prompt", "system prompt show", "system prompt clear",  # Add system prompt commands
		"defaults save", "defaults read",
		"exit", "quit", "llm ", "models"
	]
	return any(text.lower() == cmd or text.lower().startswith(cmd) for cmd in command_prefixes)
		





	
	
def interactive_mode(documents: List[Document], api_key: str, project: str, 
		document_dir: str, index_dir: str, 
		embedding_config: Optional[EmbeddingConfig] = None,
		debug: bool = False, prompts_dir: str = PROMPTS_DIR,
		llm_type: str = LLM_CLAUDE, local_model: str = DEFAULT_LOCAL_MODEL,
		hf_model: str = DEFAULT_HF_MODEL, claude_model: str = DEFAULT_CLAUDE_MODEL,
		history_dir: str = "history",
		rag_count: Optional[int] = None,
		system_prompt: Optional[str] = None) -> None:
	"""Run the application in interactive mode."""
	print_system(f"RAG Query Application - Interactive Mode (Project: {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR})")
	print_system("Type 'help' to see available commands")
	
	# Load complete project configuration
	project_config = load_project_config_file(project, document_dir)
	
	# Initialize variables
	current_project = project
	current_documents = documents
	current_embedding_config = embedding_config or get_project_embedding_config(project, document_dir, debug)
	
	# Get RAG settings from project config
	rag_settings = project_config.get("rag", {})
	current_llm_type = rag_settings.get("llm_type", llm_type)
	current_local_model = rag_settings.get("llm_model", local_model) if current_llm_type == LLM_LOCAL else local_model
	current_hf_model = rag_settings.get("llm_model", hf_model) if current_llm_type == LLM_HF else hf_model
	current_claude_model = rag_settings.get("llm_model", claude_model) if current_llm_type == LLM_CLAUDE else claude_model
	current_openai_model = rag_settings.get("llm_model", DEFAULT_OPENAI_MODEL) if current_llm_type == LLM_OPENAI else DEFAULT_OPENAI_MODEL
	current_rag_mode = rag_settings.get("rag_mode", "chunk")
	
	# Define get_current_model_name function BEFORE using it
	def get_current_model_name():
		if current_llm_type == LLM_LOCAL:
			return current_local_model
		elif current_llm_type == LLM_HF:
			return current_hf_model
		elif current_llm_type == LLM_CLAUDE:
			return current_claude_model
		elif current_llm_type == LLM_OPENAI:
			return current_openai_model
		else:
			return "unknown"	

	
	# Initialize system prompt from:
	# 1. Command line argument
	# 2. Project config
	# 3. System settings for the current model

	current_system_prompt = None

	if system_prompt:
		current_system_prompt = system_prompt
	elif "system_prompt" in rag_settings:
		current_system_prompt = rag_settings.get("system_prompt")
	else:
		# Try to get from model-specific system prompts
		current_model = get_current_model_name()
		system_settings = project_config.get("system", {})
		if current_model in system_settings:
			current_system_prompt = system_settings[current_model].get("system_prompt")
	
	if debug and current_system_prompt:
		print_debug(f"Using system prompt: \"{current_system_prompt}\"")
	
	if rag_count is not None:
		current_rag_count = rag_count
	else:
		current_rag_count = rag_settings.get("rag_count", TOP_K_DOCUMENTS)
		
			
			
	# Add function to save system prompt to project config
	def save_system_prompt(prompt: str, save_to_defaults: bool = False):
		"""Save system prompt to the project configuration."""
		project_config = load_project_config_file(current_project, document_dir)
		
		# Ensure rag section exists
		if "rag" not in project_config:
			project_config["rag"] = {}
		
		# Save to rag settings
		project_config["rag"]["system_prompt"] = prompt
		
		# If requested, also save to defaults
		if save_to_defaults:
			if "defaults" not in project_config:
				project_config["defaults"] = {}
			project_config["defaults"]["system_prompt"] = prompt
			
			# Save to system settings for the current model
			current_model = get_current_model_name()
			if "system" not in project_config:
				project_config["system"] = {}
			
			if current_model not in project_config["system"]:
				project_config["system"][current_model] = {}
			
			project_config["system"][current_model]["system_prompt"] = prompt
		
		# Save the updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Saved system prompt to project config")
			if save_to_defaults:
				print_debug(f"Also saved to defaults and system settings for model: {current_model}")
	
	
	# Modify save_current_settings_as_defaults function to include system prompt
	def save_current_settings_as_defaults():
		"""
		Save the current RAG settings as defaults in the project configuration.
		"""
		
		if debug:
			print_debug("save_current_settings_as_defaults")
		
		# Get the current model name based on LLM type
		current_model = get_current_model_name()
		
		# Get current project config
		project_config = load_project_config_file(current_project, document_dir)
		
		# Ensure we have a defaults section
		if "defaults" not in project_config:
			project_config["defaults"] = {}
		
		# Copy current RAG settings to defaults
		project_config["defaults"] = {
			"llm_type": current_llm_type,
			"llm_model": current_model,
			"rag_mode": current_rag_mode,
			"rag_count": current_rag_count
		}
		
		# Add system prompt if available
		if current_system_prompt:
			project_config["defaults"]["system_prompt"] = current_system_prompt
			
			# Also save to system section for the current model
			if "system" not in project_config:
				project_config["system"] = {}
				
			if current_model not in project_config["system"]:
				project_config["system"][current_model] = {}
				
			project_config["system"][current_model]["system_prompt"] = current_system_prompt
		
		# Save updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Saved current settings as defaults to {config_path}")
			print_debug(f"  llm_type: {current_llm_type}")
			print_debug(f"  llm_model: {current_model}")
			print_debug(f"  rag_mode: {current_rag_mode}")
			print_debug(f"  rag_count: {current_rag_count}")
			if current_system_prompt:
				print_debug(f"  system_prompt: \"{current_system_prompt}\"")
		
	# Modify load_defaults_to_current_settings to include system prompt
	def load_defaults_to_current_settings():
		"""
		Load default settings from project configuration and apply them to current RAG settings.
		
		Returns:
			Dictionary containing the loaded default settings or empty dict if none found
		"""
		# Get current project config
		project_config = load_project_config_file(current_project, document_dir)
		
		# Check if we have defaults
		if "defaults" not in project_config or not project_config["defaults"]:
			if debug:
				print_debug("No default settings found in project configuration")
			return {}
		
		# Get defaults
		defaults = project_config["defaults"]
		
		# Copy defaults to current RAG settings
		project_config["rag"] = dict(defaults)
		
		# Save updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Loaded default settings to current RAG settings:")
			print_debug(f"  llm_type: {defaults.get('llm_type')}")
			print_debug(f"  llm_model: {defaults.get('llm_model')}")
			print_debug(f"  rag_mode: {defaults.get('rag_mode')}")
			print_debug(f"  rag_count: {defaults.get('rag_count')}")
			if "system_prompt" in defaults:
				print_debug(f"  system_prompt: \"{defaults.get('system_prompt')}\"")
		
		return defaults
	
			
	# Function to save current RAG settings to project config
	def save_current_rag_settings():
		# Get current project config
		project_config = load_project_config_file(current_project, document_dir)
		
		# Get current model based on LLM type
		model_to_save = get_current_model_name()
		
		# Update RAG settings
		project_config["rag"] = {
			"llm_type": current_llm_type,
			"llm_model": model_to_save,
			"rag_mode": current_rag_mode,
			"rag_count": current_rag_count
		}
		
		# Add system prompt if available
		if current_system_prompt:
			project_config["rag"]["system_prompt"] = current_system_prompt
		
		# Save updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Saved updated RAG settings to {config_path}")
			print_debug(f"  llm_type: {current_llm_type}")
			print_debug(f"  llm_model: {model_to_save}")
			print_debug(f"  rag_mode: {current_rag_mode}")
			print_debug(f"  rag_count: {current_rag_count}")
			if current_system_prompt:
				print_debug(f"  system_prompt: \"{current_system_prompt}\"")
	
	# Initialize history
	history = CommandHistory(history_dir=history_dir)
	
	# Initialize embedding provider cache
	provider_cache = EmbeddingProviderCache(debug=debug)
	
	# Print the initial help info
	print_help_info(current_project, current_llm_type, get_current_model_name(), current_rag_mode, current_rag_count)
	
	while True:
		try:
			# Print the prompt with the current project and LLM highlighted
			current_model = get_current_model_name()
			prompt = f"\n{QUERY_COLOR}Enter your question [{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{QUERY_COLOR}] [{HIGHLIGHT_COLOR}{current_llm_type}:{current_model}{RESET_COLOR}{QUERY_COLOR}]: {RESET_COLOR}"
			
			# Use print and input separately to ensure proper scrolling behavior
			print(prompt, end='', flush=True)
			query = input().strip()
			
			# The rest of the function continues below...
			
			if not query:
				continue
			
			# Add to history - determine if it's a command or query
			is_query = not is_command(query)
			history.add(query, is_query=is_query)
			
			if query.lower() in ['exit', 'quit']:
				print_system("Exiting...")
				break
				
			elif query.lower() == 'help':
				# Display help information
				print_help_info(current_project, current_llm_type, get_current_model_name(), 
							  current_rag_mode, current_rag_count)
				continue
			
			# Handle special commands
			elif query.lower() == 'history':
				# Show command history
				entries = history.get_entries()
				if not entries:
					print_system("History is empty")
				else:
					print_system("\nCommand History:")
					for i, (cmd, is_query) in enumerate(entries[-20:], 1):  # Show last 20 entries
						cmd_type = "Query" if is_query else "Command"
						color = QUERY_COLOR if is_query else SYSTEM_COLOR
						print(f"{color}{i}. [{cmd_type}] {cmd}{RESET_COLOR}")
				continue
				
			elif query.lower() == 'history clear':
				# Clear command history
				confirm = input(f"{SYSTEM_COLOR}Are you sure you want to clear the command history? (y/n): {RESET_COLOR}").strip().lower()
				if confirm == 'y':
					history.clear()
					print_system("Command history cleared")
				continue
				
			elif query.lower() == 'history save':
				# Save command history to file
				filepath = history.save()
				if filepath:
					print_system(f"History saved to: {filepath}")
				continue
				
			# Add to the interactive_mode function where it handles commands
			elif query.lower() == 'defaults save':
				# Save current settings as defaults
				save_current_settings_as_defaults()
				
				current_model = get_current_model_name()
				print_system(f"Saved current settings as defaults:")
				print_system(f"  LLM Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				print_system(f"  LLM Model: {HIGHLIGHT_COLOR}{current_model}{RESET_COLOR}")
				print_system(f"  RAG Mode: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
				print_system(f"  Document Count: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
				if current_system_prompt:
					print_system(f"  System Prompt: {HIGHLIGHT_COLOR}\"{current_system_prompt}\"{RESET_COLOR}")
				continue
			
			elif query.lower() == 'defaults read':
				# Load defaults to current settings
				defaults = load_defaults_to_current_settings()
				
				if defaults:
					# Update current variables with defaults
					current_llm_type = defaults.get('llm_type', current_llm_type)
					
					# Update the appropriate model variable based on LLM type
					model_name = defaults.get('llm_model', '')
					if current_llm_type == LLM_LOCAL:
						current_local_model = model_name or current_local_model
					elif current_llm_type == LLM_HF:
						current_hf_model = model_name or current_hf_model
					elif current_llm_type == LLM_CLAUDE:
						current_claude_model = model_name or current_claude_model
					elif current_llm_type == LLM_OPENAI:
						current_openai_model = model_name or current_openai_model
					
					current_rag_mode = defaults.get('rag_mode', current_rag_mode)
					current_rag_count = defaults.get('rag_count', current_rag_count)
					
					# First check if there's a system prompt in defaults
					if "system_prompt" in defaults:
						current_system_prompt = defaults.get("system_prompt")
					else:
						# If not, check if there's a system prompt for the current model
						current_model = get_current_model_name()
						system_settings = project_config.get("system", {})
						if current_model in system_settings:
							model_settings = system_settings[current_model]
							if "system_prompt" in model_settings:
								current_system_prompt = model_settings["system_prompt"]
					
					print_system(f"Loaded default settings:")
					print_system(f"  LLM Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
					print_system(f"  LLM Model: {HIGHLIGHT_COLOR}{get_current_model_name()}{RESET_COLOR}")
					print_system(f"  RAG Mode: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
					print_system(f"  Document Count: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
					if current_system_prompt:
						print_system(f"  System Prompt: {HIGHLIGHT_COLOR}\"{current_system_prompt}\"{RESET_COLOR}")
				else:
					print_system("No default settings found in project configuration")
				continue			
			
			
			
			# Handle other special commands
			elif query.lower() == 'projects':
				projects = discover_projects(index_dir)
				print_system("\nAvailable Projects:")
				for p in projects:
					marker = "*" if p == current_project else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{p}{RESET_COLOR}")
				continue
			
			elif query.lower() == 'config':
				# Enhanced config command to show all settings
				print_system("\nCurrent Project Configuration:")
				print_system(f"Project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
				
				# Show embedding config
				print_system("\nIndexing Configuration:")
				print_system(f"  Embedding Type: {current_embedding_config.embedding_type}")
				print_system(f"  Embedding Model: {current_embedding_config.model_name}")
				print_system(f"  Embedding Dimensions: {current_embedding_config.dimensions}")
				
				# Show RAG config
				print_system("\nRAG Configuration:")
				print_system(f"  LLM Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				print_system(f"  LLM Model: {HIGHLIGHT_COLOR}{get_current_model_name()}{RESET_COLOR}")
				print_system(f"  RAG Mode: {current_rag_mode}")
				print_system(f"  Document Count: {current_rag_count}")
				
				# Show config file path
				config_path = get_project_config_path(current_project, document_dir)
				if os.path.exists(config_path):
					print_system(f"\nConfig File: {config_path}")
				else:
					legacy_path = get_project_config_path(current_project, document_dir, use_legacy=True)
					if os.path.exists(legacy_path):
						print_system(f"\nLegacy Config File: {legacy_path}")
					else:
						print_system("\nConfig File: Not found (using defaults)")
				continue
			
			# Set RAG count command
			elif query.lower().startswith('rag count '):
				try:
					count = int(query[10:].strip())
					if count < 1:
						print_error("RAG count must be at least 1")
					else:
						current_rag_count = count
						print_system(f"Set RAG document count to: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
						save_current_rag_settings()
				except ValueError:
					print_error("Invalid RAG count. Please specify a number.")
				continue
			
			# Set RAG mode command
			elif query.lower().startswith('rag mode '):
				mode = query[9:].strip().lower()
				if mode in ["chunk", "file", "none"]:
					current_rag_mode = mode
					print_system(f"Set RAG mode to: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
					save_current_rag_settings()
				else:
					print_error(f"Invalid RAG mode: {mode}")
					print_system("Valid modes are: chunk, file, none")
				continue
			
			# Clear the index
			elif query.lower() == 'index clear':
				# Ask for confirmation
				confirm = input(f"{SYSTEM_COLOR}Are you sure you want to clear the index for project '{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{SYSTEM_COLOR}'? This cannot be undone. (y/n): {RESET_COLOR}").strip().lower()
				
				if confirm == 'y':
					print_system(f"\nClearing index for project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
					success = clear_index(current_project, index_dir, debug)
					
					if success:
						# Reset the current documents to an empty list
						current_documents = []
						print_system(f"Project '{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{SYSTEM_COLOR}' index cleared successfully")
						print_system(f"The index is now empty")
					
				continue
	
			elif query.lower() == 'index':
				print_system(f"\nIndexing project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
				success = index_project(
					current_project, 
					document_dir, 
					index_dir, 
					debug=debug,
					auto_adjust_chunks=True,
					chars_per_dimension=DEFAULT_CHARS_PER_DIMENSION
				)
				
				if success:
					# Reload the project index
					index_path, backup_dir = get_index_path(index_dir, current_project)
					current_documents = load_index(index_path, backup_dir, debug)
					
					# Reload the embedding config
					current_embedding_config = get_project_embedding_config(current_project, document_dir, debug)
					
					# Since the config might have changed, clear the cache for this project
					if debug:
						print_debug(f"Clearing and reloading embedding provider for project: {current_project}")
					provider_cache.clear_cache()
					provider_cache.get_provider(current_project, document_dir, current_embedding_config)
					
					print_system(f"Project '{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{SYSTEM_COLOR}' re-indexed successfully")
					print_system(f"Loaded {len(current_documents)} documents")
				continue
				
			
			
			# Handle system prompt commands
			elif query.lower() == 'system prompt show':
				if current_system_prompt:
					print_system(f"Current system prompt:")
					print_system(f"{HIGHLIGHT_COLOR}\"{current_system_prompt}\"{RESET_COLOR}")
				else:
					print_system("No system prompt is currently set")
				continue
				
			elif query.lower() == 'system prompt clear':
				current_system_prompt = None
				save_system_prompt(None)
				print_system("System prompt cleared")
				save_current_rag_settings()
				continue
				
			elif query.lower().startswith('system prompt "') or query.lower().startswith('system prompt \''):
				# Extract the prompt text between quotes
				match = re.match(r'system prompt ["\'](.*)["\']$', query)
				if match:
					new_prompt = match.group(1)
					current_system_prompt = new_prompt
					save_system_prompt(new_prompt)
					print_system(f"System prompt set to:")
					print_system(f"{HIGHLIGHT_COLOR}\"{new_prompt}\"{RESET_COLOR}")
					save_current_rag_settings()
				else:
					print_error("Invalid system prompt format. Use: system prompt \"your prompt here\"")
				continue

			
				
			# Add to the models command in interactive_mode
			elif query.lower() == 'models':
				# List both llm and Hugging Face models
				print_system("\nCurrent LLM Settings:")
				print_system(f"  Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				
				if current_llm_type == LLM_LOCAL:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
				elif current_llm_type == LLM_HF:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
				elif current_llm_type == LLM_CLAUDE:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_claude_model}{RESET_COLOR}")
				elif current_llm_type == LLM_OPENAI:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_openai_model}{RESET_COLOR}")
				
				# Add OpenAI model listing
				print_system("\nAvailable OpenAI models (--llm openai):")
				openai_models = [
					OPENAI_O3_MINI,
					OPENAI_GPT3_TURBO,
					OPENAI_GPT4_TURBO,
					OPENAI_GPT4,
					OPENAI_GPT4O,
					OPENAI_GPT4O_MINI
				]
				for model in openai_models:
					marker = "*" if model == current_openai_model and current_llm_type == LLM_OPENAI else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{model}{RESET_COLOR}")
				
				# Add Claude model listing
				print_system("\nAvailable Claude models (--llm claude):")
				claude_models = [
					CLAUDE_HAIKU,
					CLAUDE_SONNET,
					CLAUDE_OPUS,
					CLAUDE_HAIKU_LEGACY,
					CLAUDE_SONNET_LEGACY
				]
				for model in claude_models:
					marker = "*" if model == current_claude_model and current_llm_type == LLM_CLAUDE else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{model}{RESET_COLOR}")

				
				# Try Simon Willison's llm library first
				llm_found = False
				try:
					import llm
					llm_found = True
					
					# Try different methods to get models
					try:
						models = llm.get_models()
						model_names = [m.model_id for m in models] if hasattr(models[0], 'model_id') else [str(m) for m in models]
					except:
						try:
							model_names = llm.list_models()
						except:
							# As a last resort, call the CLI command
							import subprocess
							result = subprocess.run(['llm', 'models'], capture_output=True, text=True)
							output = result.stdout
							model_names = []
							for line in output.split('\n'):
								if line.strip():
									parts = line.strip().split()
									if parts:
										model_names.append(parts[0])                    
					if model_names:
						print_system("\nAvailable Models from llm library (--llm local):")
						for name in model_names:
							marker = "*" if name == current_local_model and current_llm_type == LLM_LOCAL else " "
							print_system(f"{marker} {HIGHLIGHT_COLOR}{name}{RESET_COLOR}")
					else:
						print_system("\nNo models found through llm library.")
						print_system("You may need to install models with 'llm install <model>'")
				except ImportError:
					print_system("\nSimon Willison's llm library is not installed.")
					print_system("You can install it with: pip install llm")
				
				# Show some common Hugging Face models
				print_system("\nCommon Hugging Face models (--llm hf):")
				hf_models = [
					"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
					"nomic-ai/gpt4all-j",
					"EleutherAI/gpt-neo-1.3B",
					"mistralai/Mistral-7B-v0.1"
				]
				
				# Add shorthand names
				print_system("\nShorthand names for Hugging Face models:")
				shorthand_models = {
					"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
					"gpt4all": "nomic-ai/gpt4all-j",
					"gpt-neo": "EleutherAI/gpt-neo-1.3B",
					"mistral": "mistralai/Mistral-7B-v0.1"
				}
				
				for name, full_name in shorthand_models.items():
					marker = "*" if (name == current_hf_model or full_name == current_hf_model) and current_llm_type == LLM_HF else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{name}{RESET_COLOR} → {full_name}")
				
				print_system("\nUsage examples:")
				print_system("  llm claude")
				print_system("  llm local")
				print_system("  llm local orca-2-7b")
				print_system("  llm hf")
				print_system("  llm hf tinyllama")
				print_system("  llm hf TinyLlama/TinyLlama-1.1B-Chat-v1.0")
				
				continue
			
			
			

			
			elif query.lower().startswith('llm '):
				# Parse the llm command with more intuitive syntax
				parts = query[4:].strip().split(maxsplit=1)
				llm_choice = parts[0].lower() if parts else ""
				model_arg = parts[1] if len(parts) > 1 else None
				
				if llm_choice == LLM_CLAUDE:
					current_llm_type = LLM_CLAUDE
					if model_arg:
						current_claude_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_claude_model}{RESET_COLOR}")
					else:
						current_claude_model = DEFAULT_CLAUDE_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_claude_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				elif llm_choice == LLM_OPENAI:
					current_llm_type = LLM_OPENAI
					if model_arg:
						current_openai_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_openai_model}{RESET_COLOR}")
					else:
						current_openai_model = DEFAULT_OPENAI_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_openai_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				elif llm_choice == LLM_LOCAL:
					current_llm_type = LLM_LOCAL
					if model_arg:
						current_local_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
					else:
						current_local_model = DEFAULT_LOCAL_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				elif llm_choice == LLM_HF:
					current_llm_type = LLM_HF
					if model_arg:
						current_hf_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
					else:
						current_hf_model = DEFAULT_HF_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				else:
					print_error(f"Unknown LLM type: {llm_choice}")
					print_system("Valid options are:")
					print_system("  llm claude [model_name]")
					print_system("  llm openai [model_name]")
					print_system("  llm local [model_name]")
					print_system("  llm hf [model_name]")
				
				continue

			
			elif query.lower().startswith('project '):
				new_project = query[8:].strip()
				index_path, backup_dir = get_index_path(index_dir, new_project)
				
				if not os.path.exists(index_path):
					print_system(f"Project '{HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}{SYSTEM_COLOR}' not found or not indexed")
					# Ask if user wants to create it
					create = input(f"{SYSTEM_COLOR}Would you like to create and index project '{HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}{SYSTEM_COLOR}'? (y/n): {RESET_COLOR}").strip().lower()
					if create == 'y':
						# Create the project directory if needed
						if new_project != MASTER_PROJECT:
							project_dir = os.path.join(document_dir, new_project)
							os.makedirs(project_dir, exist_ok=True)
							print_system(f"Created project directory: {project_dir}")
						
						# Index the new project
						success = index_project(new_project, document_dir, index_dir, debug)
						if success:
							current_project = new_project
							current_documents = load_index(index_path, backup_dir, debug)
							
							# Load the project configuration
							project_config = load_project_config_file(new_project, document_dir)
							
							# Update embedding config
							current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
							
							# Update RAG settings from project config
							rag_settings = project_config.get("rag", {})
							current_llm_type = rag_settings.get("llm_type", current_llm_type)
							current_local_model = rag_settings.get("llm_model", current_local_model)
							current_hf_model = rag_settings.get("llm_model", current_hf_model) if current_llm_type == LLM_HF else current_hf_model
							current_rag_mode = rag_settings.get("rag_mode", current_rag_mode)
							current_rag_count = rag_settings.get("rag_count", current_rag_count)
							
							# Preload the embedding provider for the new project
							provider_cache.get_provider(current_project, document_dir, current_embedding_config)
							
							print_system(f"Switched to project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
					continue
				
				# Load the new project
				new_documents = load_index(index_path, backup_dir, debug)
				if new_documents:
					current_project = new_project
					current_documents = new_documents
					
					# Load the project configuration
					project_config = load_project_config_file(new_project, document_dir)
					
					# Update embedding config
					current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
					
					# Update RAG settings from project config
					rag_settings = project_config.get("rag", {})
					current_llm_type = rag_settings.get("llm_type", current_llm_type)
					current_local_model = rag_settings.get("llm_model", current_local_model)
					current_hf_model = rag_settings.get("llm_model", current_hf_model) if current_llm_type == LLM_HF else current_hf_model
					current_rag_mode = rag_settings.get("rag_mode", current_rag_mode)
					current_rag_count = rag_settings.get("rag_count", current_rag_count)
					
					# Preload the embedding provider for the new project
					provider_cache.get_provider(current_project, document_dir, current_embedding_config)
					
					print_system(f"Switched to project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
					print_system(f"Embedding Type: {current_embedding_config.embedding_type}")
					print_system(f"Embedding Model: {current_embedding_config.model_name}")
					print_system(f"Embedding Dimensions: {current_embedding_config.dimensions}")
					print_system(f"LLM Type: {current_llm_type}")
					print_system(f"RAG Mode: {current_rag_mode}")
					print_system(f"Document Count: {current_rag_count}")

				else:
					print_system(f"No documents found in project: {HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}")
					print_system(f"Add .txt or .md documents to the project folder: {HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}")

				continue
			
			# Regular query - search for relevant documents
			# Echo the query if it's not a command

			# In the interactive_mode function, modify the query handling section:
			# Regular query - search for relevant documents
			if not is_command(query):
				# Add a blank line after the query for better readability
				print()
				
				# In 'none' RAG mode, we skip the document search
				if current_rag_mode.lower() == "none":
					relevant_docs = []
					if debug:
						print_debug(f"Using RAG mode 'none' - skipping document search")
				else:
					print_system("Searching for relevant documents...")
					
					# Pass the provider cache to the search function
					relevant_docs = search_documents(
						query, current_documents, current_project, 
						document_dir, current_embedding_config, 
						top_k=current_rag_count,
						debug=debug, provider_cache=provider_cache
					)
				
				# If we found relevant documents and not in debug mode, confirm before querying
				if relevant_docs and debug:
					proceed = input(f"{SYSTEM_COLOR}Proceed with query using these sources? (Y/n): {RESET_COLOR}").strip().lower()
					if proceed == 'n':
						print_system("Query canceled")
						continue
				
				# Get the model name based on current LLM type
				model_name = None
				if current_llm_type == LLM_LOCAL:
					model_name = current_local_model
				elif current_llm_type == LLM_HF:
					model_name = current_hf_model
				
				# Ask the selected LLM
				print_system(f"Generating answer with {current_llm_type} {model_name} (RAG mode: {current_rag_mode})...")
				
				# Pass the system prompt to get_response
				answer = get_response(
					query, relevant_docs, api_key, current_project,
					current_llm_type, model_name, debug, prompts_dir,
					current_rag_mode, document_dir, current_system_prompt  # Pass system prompt
				)
				
				# Print the answer with proper colors
				print(f"\n{ANSWER_COLOR}Answer:{RESET_COLOR}")
				print(f"{ANSWER_COLOR}{answer}{RESET_COLOR}")
				
				
	
		except KeyboardInterrupt:
			print_system("\nInterrupted by user. Exiting...")
			break
		except Exception as e:
			print_error(f"{e}")
			if debug:
				print(traceback.format_exc())
				
				
				
				
				
def print_help_info(current_project: str, current_llm_type: str, current_model: str, 
	current_rag_mode: str, current_rag_count: int,
	current_system_prompt: str = None) -> None:		
	"""
	Print help information about available commands.
	"""
	
	print_system(f"Current Project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
	print_system(f"Current LLM: {HIGHLIGHT_COLOR}{current_llm_type}:{current_model}{RESET_COLOR}")
	print_system(f"Current RAG Mode: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
	print_system(f"Document Count: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
	
	# Add system prompt info if available
	if current_system_prompt:
		# If the prompt is long, truncate it for display
		display_prompt = current_system_prompt
		if len(display_prompt) > 50:
			display_prompt = display_prompt[:47] + "..."
		print_system(f"System Prompt: {HIGHLIGHT_COLOR}\"{display_prompt}\"{RESET_COLOR}")

	
	print_system("\nAvailable Commands:")
	print_system("  help                     Show this help information")
	print_system("  exit, quit               End the session")
	
	# Project commands
	print_system("\nProject Commands:")
	print_system("  projects                 List all available projects")
	print_system("  project <name>           Switch to a different project")
	print_system("  config                   Show current project configuration")
	
	# Index commands
	print_system("\nIndex Commands:")
	print_system("  index                    Re-index the current project")
	print_system("  index clear              Clear the current project's index")
	
	# RAG commands
	print_system("\nRAG Commands:")
	print_system("  rag mode <mode>          Set RAG mode (chunk, file, none)")
	print_system("  rag count <number>       Set number of documents to retrieve")
	print_system("  defaults save            Save current settings as defaults")
	print_system("  defaults read            Load default settings to current configuration")

	
	# LLM commands
	print_system("\nLLM Commands:")
	print_system("  models                   List available LLM models")
	print_system("  llm claude [model_name]  Use Claude API (with optional model)")
	print_system("  llm openai [model_name]  Use OpenAI API (with optional model)")
	print_system("  llm local [model_name]   Use a local model via llm library")
	print_system("  llm hf [model_name]      Use a Hugging Face model")
	
	# Add system prompt command to the help info
	print_system("\nSystem Prompt Commands:")
	print_system("  system prompt \"<prompt>\"  Set the system prompt for the current LLM")
	print_system("  system prompt show        Show the current system prompt")
	print_system("  system prompt clear       Clear the current system prompt")

		
	# History commands
	print_system("\nHistory Commands:")
	print_system("  history                  Show command history")
	print_system("  history clear            Clear command history")
	print_system("  history save             Save history to a file")
	
	print_system("\nFor any other input, the application will treat it as a query")
	print_system("and search for relevant documents to help answer it.")	
				
				








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
			print_debug(f"Using Claude API model: {claude_model} for response with RAG mode: {rag_mode}")
			if system_prompt:
				print_debug(f"Using system prompt: \"{system_prompt}\"")
		return ask_claude(query, relevant_docs, api_key, project, debug, prompts_dir, rag_mode, document_dir, claude_model, system_prompt)
	
	elif llm_type.lower() == LLM_OPENAI:
		openai_model = model_name or DEFAULT_OPENAI_MODEL
		if debug:
			print_debug(f"Using OpenAI API model: {openai_model} for response with RAG mode: {rag_mode}")
			if system_prompt:
				print_debug(f"Using system prompt: \"{system_prompt}\"")
		# For OpenAI, use the OPENAI_API_KEY environment variable if not provided
		openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
		if not openai_api_key:
			return "OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or provide it via --api-key."
		return ask_openai(query, relevant_docs, openai_api_key, project, debug, prompts_dir, rag_mode, document_dir, openai_model, system_prompt)
	
	elif llm_type.lower() == LLM_LOCAL:
		local_model = model_name or DEFAULT_LOCAL_MODEL
		if debug:
			print_debug(f"Using Simon Willison's llm with model: {local_model} and RAG mode: {rag_mode}")
			if system_prompt:
				print_debug(f"Using system prompt: \"{system_prompt}\"")
		try:
			return ask_local_llm(query, relevant_docs, project, local_model, debug, prompts_dir, rag_mode, document_dir, system_prompt)
		except Exception as e:
			if debug:
				print_debug(f"Error with local llm: {e}, falling back to Hugging Face")
			# If Simon's llm fails, fall back to Hugging Face
			return ask_local_hf(query, relevant_docs, project, DEFAULT_HF_MODEL, debug, prompts_dir, rag_mode, document_dir, system_prompt)
	
	elif llm_type.lower() == LLM_HF:
		hf_model = model_name or DEFAULT_HF_MODEL
		if debug:
			print_debug(f"Using Hugging Face transformers with model: {hf_model} and RAG mode: {rag_mode}")
			if system_prompt:
				print_debug(f"Using system prompt: \"{system_prompt}\"")
		return ask_local_hf(query, relevant_docs, project, hf_model, debug, prompts_dir, rag_mode, document_dir, system_prompt)
	
	else:
		print_error(f"Unknown LLM type: {llm_type}. Using Claude as fallback.")
		return ask_claude(query, relevant_docs, api_key, project, debug, prompts_dir, rag_mode, document_dir, DEFAULT_CLAUDE_MODEL, system_prompt)



				
def main():
	"""Main entry point for the query application."""
	parser = argparse.ArgumentParser(description="RAG Query Application with Project Support")
	
	parser.add_argument("--api-key", type=str, help="API key for selected LLM (Claude or OpenAI)")
	parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR, 
						help="Directory containing the document index")
	parser.add_argument("--document-dir", type=str, default=DEFAULT_DOCUMENT_DIR,
						help="Directory containing documents")
	parser.add_argument("--query", type=str, 
						help="Single query mode: ask a question and exit")
	parser.add_argument("--embedding-model", type=str,
						help="Embedding model to use")
	parser.add_argument("--embedding-type", type=str,
						help="Type of embedding to use (sentence_transformers, openai)")
	parser.add_argument("--debug", action="store_true",
						help="Enable debug logging")
	parser.add_argument("--project", type=str, default=MASTER_PROJECT,
						help="Project to query (default: master)")
	parser.add_argument("--list-projects", action="store_true",
						help="List all available projects")
	parser.add_argument("--index-clear", action="store_true",
						help="Clear (erase) the index for the specified project")
	parser.add_argument("--index", action="store_true",
						help="Index the specified project before querying")
	parser.add_argument("--no-color", action="store_true",
						help="Disable colored output")
	parser.add_argument("--prompts-dir", type=str, default=PROMPTS_DIR,
						help="Directory to save prompt logs (only in debug mode)")
	
	# for selecting the LLM
	parser.add_argument("--llm", type=str, default=DEFAULT_LLM_TYPE,
						help="LLM to use: 'claude', 'openai', 'local' (Simon Willison's llm), or 'hf' (Hugging Face)")
	parser.add_argument("--local-model", type=str, default=DEFAULT_LOCAL_MODEL,
						help="Local model to use when --llm=local (default: gpt4all)")
	parser.add_argument("--hf-model", type=str, default=DEFAULT_HF_MODEL,
						help="Hugging Face model to use when --llm=hf")
	parser.add_argument("--history-dir", type=str, default="history",
						help="Directory to store command history")
	
	# Add the new command line argument for rag-count
	parser.add_argument("--rag-count", type=int,
						help=f"Number of documents to retrieve (default: {TOP_K_DOCUMENTS})")
						
	parser.add_argument("--model", type=str, default=None,
					   help="Model to use for the selected LLM")
					   
	# add argument for system prompt
	parser.add_argument("--system-prompt", type=str, 
	   					help="System prompt to use for the LLM")

	
	args = parser.parse_args()
	
	# Rest of the main function...
	
	# Update API key handling
	api_key = args.api_key
	if args.llm.lower() == LLM_CLAUDE:
		api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
		if not api_key:
			print_error("Anthropic API key is required to use Claude.")
			print_error("Please provide it via --api-key or set the ANTHROPIC_API_KEY environment variable.")
	elif args.llm.lower() == LLM_OPENAI:
		api_key = api_key or os.environ.get("OPENAI_API_KEY")
		if not api_key:
			print_error("OpenAI API key is required to use OpenAI models.")
			print_error("Please provide it via --api-key or set the OPENAI_API_KEY environment variable.")
	

	
	# Set prompts directory from args (no need for global declaration)
	if args.prompts_dir != PROMPTS_DIR:
		# Only use a different directory if it was explicitly specified
		prompts_dir = args.prompts_dir
		# Create the directory if it doesn't exist
		os.makedirs(prompts_dir, exist_ok=True)
		# Update the save_prompt_to_json function to use this directory
		if args.debug:
			print_debug(f"Prompt logs will be saved to: {os.path.abspath(prompts_dir)}")
	else:
		prompts_dir = PROMPTS_DIR
		# Create the default directory if in debug mode
		if args.debug:
			os.makedirs(prompts_dir, exist_ok=True)
			print_debug(f"Prompt logs will be saved to: {os.path.abspath(prompts_dir)}")
	
	# Disable colors if requested
	global QUERY_COLOR, ANSWER_COLOR, DEBUG_COLOR, ERROR_COLOR, SYSTEM_COLOR, HIGHLIGHT_COLOR, RESET_COLOR
	if args.no_color or not COLORS_AVAILABLE:
		QUERY_COLOR = ""
		ANSWER_COLOR = ""
		DEBUG_COLOR = ""
		ERROR_COLOR = ""
		SYSTEM_COLOR = ""
		HIGHLIGHT_COLOR = ""
		RESET_COLOR = ""
	
	# Get API key from args or environment
	api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
	
	if not api_key:
		print_error("Anthropic API key is required to use Claude.")
		print_error("(optional) Please provide it via --api-key or set the ANTHROPIC_API_KEY environment variable.")
		# sys.exit(1)
	
	# Check if document directory exists
	if not os.path.exists(args.document_dir):
		print_system(f"Document directory not found: {args.document_dir}")
		print_system("Document directory created.")
		os.makedirs(args.document_dir, exist_ok=True)
		print_system("Please add your projects and files in the 'documents' directory.")
		# sys.exit(1)
	
	# Create the index directory if it doesn't exist
	os.makedirs(args.index_dir, exist_ok=True)
	
	# Create prompts directory if in debug mode
	if args.debug:
		os.makedirs(PROMPTS_DIR, exist_ok=True)
		print_debug(f"Prompt logs will be saved to: {os.path.abspath(PROMPTS_DIR)}")
	
	# Just list projects if requested
	if args.list_projects:
		projects = discover_projects(args.index_dir)
		if not projects:
			print_system("No indexed projects found.")
			return
			
		print_system("\nAvailable Projects:")
		for project in projects:
			# Get the document count for this project
			index_path, backup_dir = get_index_path(args.index_dir, project)
			try:
				with open(index_path, 'rb') as f:
					documents = pickle.load(f)
				
				# Show embedding information
				embedding_types = {}
				for doc in documents:
					emb_type = doc.metadata.get('embedding_type', 'unknown')
					emb_model = doc.metadata.get('embedding_model', 'unknown')
					key = f"{emb_type}/{emb_model}"
					embedding_types[key] = embedding_types.get(key, 0) + 1
				
				emb_info = ", ".join([f"{key}: {count}" for key, count in embedding_types.items()])
				print_system(f"  {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR} ({len(documents)} documents, {emb_info})")
				
				# Show config file path and info
				config_path = get_project_config_path(project, args.document_dir)
				if os.path.exists(config_path):
					try:
						config = EmbeddingConfig.from_json_file(config_path)
						print_system(f"    Config: {config.embedding_type}/{config.model_name}")
					except:
						print_system(f"    Config: Error loading {config_path}")
				else:
					print_system(f"    Config: Not found (using defaults)")
					
			except:
				print_system(f"  {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR} (error loading index)")
		return
	
	# Create embedding configuration from command line args
	embedding_config = None
	if args.embedding_model or args.embedding_type:
		embedding_config = EmbeddingConfig(
			embedding_type=args.embedding_type or DEFAULT_EMBEDDING_TYPE,
			model_name=args.embedding_model or DEFAULT_EMBEDDING_MODEL
		)
		
	# To clear the index
	# Add this code after the argument parsing but before loading the index:
	if args.index_clear:
		print_system(f"Clearing index for project: {HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}")
		success = clear_index(args.project, args.index_dir, args.debug)
		if not success:
			print_error("Failed to clear index.")
			sys.exit(1)
		if args.query is None:  # If not in query mode, exit after clearing
			print_system("Index cleared successfully. Exiting.")
			sys.exit(0)
		else:
			# If we're going to query, we need an empty documents list
			documents = []
	
	# Handle indexing if requested
	if args.index:
		print_system(f"Indexing project: {HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}")
		success = index_project(
			args.project, 
			args.document_dir, 
			args.index_dir, 
			debug=args.debug,
			auto_adjust_chunks=True,
			chars_per_dimension=DEFAULT_CHARS_PER_DIMENSION
		)
		if not success:
			print_error("Indexing failed. Exiting.")
			sys.exit(1)
					
						
	# Get index path for the specified project
	index_path, backup_dir = get_index_path(args.index_dir, args.project)
	
	# Check if the index exists
	if not os.path.exists(index_path):
		print_system(f"Index for project '{HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}{SYSTEM_COLOR}' not found: {index_path}")
		# Ask if user wants to create it
		create = input(f"{SYSTEM_COLOR}Would you like to create and index project '{HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}{SYSTEM_COLOR}'? (y/n): {RESET_COLOR}").strip().lower()
		if create == 'y':
			# Create the project directory if needed
			if args.project != MASTER_PROJECT:
				project_dir = os.path.join(args.document_dir, args.project)
				os.makedirs(project_dir, exist_ok=True)
				print_system(f"Created project directory: {project_dir}")
			
			# Index the project
			success = index_project(args.project, args.document_dir, args.index_dir, args.debug)
			if not success:
				print_error("Indexing failed. Exiting.")
				sys.exit(1)
		else:
			# List available projects
			projects = discover_projects(args.index_dir)
			if projects:
				print_system("\nAvailable Projects:")
				for project in projects:
					print_system(f"  {HIGHLIGHT_COLOR}{project}{RESET_COLOR}")
			sys.exit(1)
	
	# Print application info
	print_system(f"RAG Query Application with Project Support")
	print_system(f"Python version: {sys.version}")
	print_system(f"Project: {HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}")
	print_system(f"Index location: {index_path}")
	

	# Create embedding provider cache
	provider_cache = EmbeddingProviderCache(debug=args.debug)
	
	# Ensure the model is loaded at startup
	print_system("Loading embedding model...")
	start_time = time.time()
	provider = provider_cache.get_provider(args.project, args.document_dir, embedding_config)
	if args.debug:
		print_debug(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
	else:
		print_system(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")


	# If no custom embedding config was provided, use the project's config
	if embedding_config is None:
		embedding_config = get_project_embedding_config(args.project, args.document_dir, args.debug)
		
	# Preload the embedding provider
	if args.debug:
		print_debug(f"Preloading embedding provider for project: {args.project}")

	provider_cache.get_provider(args.project, args.document_dir, embedding_config)

	# Print more application info
	print_system(f"Embedding Type: {embedding_config.embedding_type}")
	print_system(f"Embedding Model: {embedding_config.model_name}")

	
	try:
		print_system(f"Anthropic SDK version: {anthropic.__version__}")
	except AttributeError:
		print_system("Anthropic SDK version: unknown")
	
	# Load document index for the project
	documents = load_index(index_path, backup_dir, args.debug)
	
	if not documents:
		print_error(f"No documents found in the project index. Please add documents and run `index`.")
		# sys.exit(1)
	
	# Display information about the embeddings in the index
	if args.debug:
		embedding_types = {}
		for doc in documents:
			emb_type = doc.metadata.get('embedding_type', 'unknown')
			emb_model = doc.metadata.get('embedding_model', 'unknown')
			key = f"{emb_type}/{emb_model}"
			embedding_types[key] = embedding_types.get(key, 0) + 1
		
		print_debug("\nEmbedding types in index:")
		for key, count in embedding_types.items():
			print_debug(f"  {key}: {count} documents")
	
	# Load project configuration to get rag_count
	project_config = load_project_config_file(args.project, args.document_dir)
	rag_settings = project_config.get("rag", {})
	
	# Determine the rag_count to use, in order of precedence:
	# 1. Command line argument (--rag-count)
	# 2. Project configuration
	# 3. Default value (TOP_K_DOCUMENTS)
	rag_count = args.rag_count
	if rag_count is None:
		rag_count = rag_settings.get("rag_count", TOP_K_DOCUMENTS)
	
	if args.debug:
		print_debug(f"Using rag_count: {rag_count}")
		if args.rag_count is not None:
			print_debug("Source: Command line argument")
		elif "rag_count" in rag_settings:
			print_debug("Source: Project configuration")
		else:
			print_debug("Source: Default value")
	
	
	# In the main function, for single query mode:
	
	if args.query:
		# Single query mode
		# Echo the query
		print(f"{QUERY_COLOR}{args.query}{RESET_COLOR}\n")
		
		# Get RAG mode from project config
		rag_mode = rag_settings.get("rag_mode", "chunk")
		
		# Get system prompt from args or project config
		system_prompt = args.system_prompt
		if not system_prompt:
			system_prompt = rag_settings.get("system_prompt")
			
			# If no system prompt in rag settings, check system section
			if not system_prompt:
				model_name = args.model or get_model_name_for_llm_type(args.llm)
				system_settings = project_config.get("system", {})
				if model_name in system_settings:
					system_prompt = system_settings[model_name].get("system_prompt")
		
		# In 'none' RAG mode, we skip the document search
		if rag_mode.lower() == "none":
			relevant_docs = []
			if args.debug:
				print_debug(f"Using RAG mode 'none' - skipping document search")
		else:
			print_system("Searching for relevant documents...")
			relevant_docs = search_documents(
				args.query, documents, args.project, 
				args.document_dir, embedding_config, 
				top_k=rag_count,
				debug=args.debug, provider_cache=provider_cache
			)
	
	
		
		# If we found relevant documents, confirm before querying
		if relevant_docs and not args.debug:
			proceed = input(f"{SYSTEM_COLOR}Proceed with query using these sources? (Y/n): {RESET_COLOR}").strip().lower()
			if proceed == 'n':
				print_system("Query canceled")
				sys.exit(0)
		
		# Determine the model name based on LLM type
		model_name = args.model  # Use model from command line if provided
		if not model_name:
			if args.llm == LLM_LOCAL:
				model_name = args.local_model
			elif args.llm == LLM_HF:
				model_name = args.hf_model
			elif args.llm == LLM_CLAUDE:
				model_name = DEFAULT_CLAUDE_MODEL
			elif args.llm == LLM_OPENAI:
				model_name = DEFAULT_OPENAI_MODEL
		
		# Use the selected LLM
		print_system(f"Generating answer with {args.llm} model: {model_name} (RAG mode: {rag_mode})...")
		
		# Generate the response
		answer = get_response(
			args.query, relevant_docs, api_key, args.project,
			args.llm, model_name, args.debug, prompts_dir,
			rag_mode, args.document_dir, system_prompt  # Pass system prompt here
		)
		
		# Print the answer with proper colors
		print(f"\n{ANSWER_COLOR}Answer:{RESET_COLOR}")
		print(f"{ANSWER_COLOR}{answer}{RESET_COLOR}")
	
	else:
		# Interactive mode
		# Determine the model based on the selected LLM
		model_to_use = args.model
		if not model_to_use:
			if args.llm == LLM_CLAUDE:
				model_to_use = DEFAULT_CLAUDE_MODEL
			elif args.llm == LLM_OPENAI:
				model_to_use = DEFAULT_OPENAI_MODEL
			
		# Pass system_prompt to interactive_mode
		interactive_mode(
			documents, api_key, args.project, 
			args.document_dir, args.index_dir, 
			embedding_config, args.debug, prompts_dir,
			args.llm, args.local_model, args.hf_model, model_to_use,
			args.history_dir, args.rag_count, args.system_prompt
		)

	


if __name__ == "__main__":
	main()