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
# MODEL = "claude-3-opus-20240229"
MODEL = "mistral-7b-instruct-v0"
MAX_TOKENS = 8096
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_TYPE = "sentence_transformers"
TOP_K_DOCUMENTS = 5
API_TIMEOUT = 60  # Timeout for API calls in seconds
MASTER_PROJECT = "master"  # Name for the master index
PROMPTS_DIR = "prompts"  # Directory to save prompt logs

# For the different models
# LLM types
LLM_CLAUDE = "claude"
LLM_LOCAL = "local"
LLM_HF = "hf"

# Default models
DEFAULT_LLM_TYPE = LLM_LOCAL
DEFAULT_LOCAL_MODEL = "mistral-7b-openorca"
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


def save_prompt_to_json(prompt: str, query: str, project: str, relevant_docs: List[Document], prompts_dir: str = PROMPTS_DIR) -> str:
	"""
	Save the prompt and related information to a JSON file.
	Returns the path to the saved file.
	"""
	# Create the prompts directory if it doesn't exist
	os.makedirs(prompts_dir, exist_ok=True)
	
	# Generate timestamp for the filename
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	file_path = os.path.join(prompts_dir, f"{timestamp}.json")
	
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
		"model": MODEL,
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


def get_project_config_path(project: str, document_dir: str) -> str:
	"""Get the path to the project's embedding config file."""
	if project == MASTER_PROJECT:
		config_path = os.path.join(document_dir, "embedding_config.json")
	else:
		config_path = os.path.join(document_dir, project, "embedding_config.json")
	return config_path


def save_embedding_config(project: str, document_dir: str, config: EmbeddingConfig) -> None:
	"""Save embedding configuration to the project directory."""
	config_path = get_project_config_path(project, document_dir)
	
	# Create directory if needed
	os.makedirs(os.path.dirname(config_path), exist_ok=True)
	
	# Save the config (without API key)
	try:
		config.save_to_file(config_path)
		print_system(f"Saved embedding configuration to {config_path}")
	except Exception as e:
		print_error(f"Error saving embedding configuration: {e}")


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
				 debug: bool = False) -> bool:
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
	
	if project != MASTER_PROJECT:
		cmd.extend(["--project", project])
	
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



# def get_multiline_input(prompt: str) -> str:
# 	"""
# 	Get user input with proper handling of long lines in the terminal.
# 	This ensures text scrolls up properly instead of overwriting the same line.
# 	
# 	Args:
# 		prompt: The prompt to display before input
# 		
# 	Returns:
# 		The user's input as a string
# 	"""
# 	# Print the prompt first
# 	print(prompt, end="", flush=True)
# 	
# 	# Use a list to collect input lines
# 	lines = []
# 	line = ""
# 	
# 	while True:
# 		char = sys.stdin.read(1)
# 		
# 		# Handle Enter key (line break)
# 		if char == '\n':
# 			return line
# 		# Handle backspace
# 		elif char == '\b' or ord(char) == 127:  # Different systems use different backspace codes
# 			if line:
# 				# Remove the last character
# 				line = line[:-1]
# 				# Clear the current line and reprint
# 				print("\r" + " " * (len(prompt) + len(line) + 1) + "\r" + prompt + line, end="", flush=True)
# 		# Normal character input
# 		else:
# 			line += char
# 			print(char, end="", flush=True)




def search_documents(query: str, documents: List[Document], project: str, 
			   document_dir: str, embedding_config: Optional[EmbeddingConfig] = None,
			   top_k: int = TOP_K_DOCUMENTS, debug: bool = False,
			   provider_cache: Optional[EmbeddingProviderCache] = None) -> List[Document]:
	"""
	Search for top-k distinct documents relevant to the query.
	Documents are ranked by semantic similarity, but only one chunk per distinct document is returned.
	
	Uses a provider cache to avoid reloading models when possible.
	"""
	if not documents:
		print_system("No documents in index")
		return []
	
	if debug:
		print_debug(f"Searching for: '{query}'")
	
	# Try to import tqdm for progress bar
	try:
		from tqdm import tqdm
		has_tqdm = True
	except ImportError:
		has_tqdm = False
		if not debug:
			print_system("For progress bars, install tqdm: pip install tqdm")
	
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
		
		# Extract only the top-k DISTINCT documents by file path
		top_distinct_results = []
		distinct_files = set()
		
		for doc, sim in sorted_results:
			# Get a unique identifier for this document (file path is a good choice)
			file_path = doc.metadata.get('file_path', 'unknown')
			
			# Only add if we haven't seen this file yet
			if file_path not in distinct_files:
				top_distinct_results.append((doc, sim))
				distinct_files.add(file_path)
				
				# Break once we have top_k distinct documents
				if len(top_distinct_results) >= top_k:
					break
		
		# Just get the documents without the scores
		top_results = [doc for doc, sim in top_distinct_results]
		
		# In debug mode, print details about the relevant documents
		if debug:
			print_debug(f"Found {len(top_results)} distinct relevant documents:")
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
				print_system(f"\nFound {len(top_results)} relevant sources:")
				for i, (doc, sim) in enumerate(top_distinct_results):
					file_path = doc.metadata.get('file_path', 'unknown')
					
					# Get just the filename from the path
					file_name = os.path.basename(file_path)
					
					# Get project if different from current
					doc_project = doc.metadata.get('project', MASTER_PROJECT)
					project_info = f" (project: {doc_project})" if doc_project != project and doc_project != MASTER_PROJECT else ""
					
					# Format the similarity score as percentage
					score_percent = int(sim * 100)
					
					# Print a clean summary line
					print_system(f"  {i+1}. {HIGHLIGHT_COLOR}{file_name}{RESET_COLOR}{SYSTEM_COLOR} - {score_percent}% match{project_info}")
			else:
				print_system("No relevant documents found in the index.")
		
		return top_results
		
	except Exception as e:
		print_error(f"Error during search: {e}")
		if debug:
			print(traceback.format_exc())
		return []


	
	
def ask_claude(query: str, relevant_docs: List[Document], api_key: str, project: str, debug: bool = False, prompts_dir: str = PROMPTS_DIR) -> str:
	"""Process a user query and return Claude's response."""
	try:
		client = anthropic.Anthropic(api_key=api_key)
		
		if not relevant_docs:
			# If no relevant documents found, just ask Claude directly
			prompt = f"""
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
			prompt = f"""
			User has asked: {query}
			
			I'm searching in the project: {project}
			
			I've retrieved the following documents that might help answer this question:
			
			{context}
			
			Please answer the user's question based on the information in these documents.
			If the documents don't contain the necessary information, please say so and answer based on your general knowledge.
			In your answer, cite which documents you used.
			"""
		
		if debug:
			print_debug("Sending prompt to Claude")
			
			# Save the prompt to a JSON file - FIXED PARAMETER ORDER
			log_path = save_prompt_to_json(prompt, query, project, relevant_docs, prompts_dir)
			if log_path:
				print_debug(f"Saved prompt to {log_path}")
		
		# Set up timeout
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(API_TIMEOUT)
		
		# Get response from Claude
		start_time = time.time()
		response = client.messages.create(
			model=MODEL,
			max_tokens=MAX_TOKENS,
			messages=[
				{"role": "user", "content": prompt}
			]
		)
		
		# Cancel the alarm
		signal.alarm(0)
		
		elapsed_time = time.time() - start_time
		if debug:
			print_debug(f"Received response from Claude in {elapsed_time:.2f} seconds")
		
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
				 debug: bool = False, prompts_dir: str = PROMPTS_DIR) -> str:
	"""
	Process a user query using Simon Willison's LLM library.
	
	Args:
		query: The user's query
		relevant_docs: List of relevant documents
		project: Project name
		local_model: Name of the local model to use
		debug: Enable debug output
		prompts_dir: Directory to save prompts for debugging
		
	Returns:
		The LLM's response as a string
	"""
	try:
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
			print_debug(f"Using Simon Willison's llm with model: {local_model}")
			
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
						response = model.prompt(prompt_text)
						result = str(response)
					except AttributeError:
						# Try alternate API style
						response = model.complete(prompt_text)
						result = response.text()
					
					elapsed_time = time.time() - start_time
					if debug:
						print_debug(f"Generated response in {elapsed_time:.2f} seconds")
					
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
				debug: bool = False, prompts_dir: str = PROMPTS_DIR) -> str:
	"""
	Process a user query using Hugging Face transformers.
	
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
		"exit", "quit", "llm ", "models"
	]
	return any(text.lower() == cmd or text.lower().startswith(cmd) for cmd in command_prefixes)		

		


def interactive_mode(documents: List[Document], api_key: str, project: str, 
				document_dir: str, index_dir: str, 
				embedding_config: Optional[EmbeddingConfig] = None,
				debug: bool = False, prompts_dir: str = PROMPTS_DIR,
				llm_type: str = LLM_CLAUDE, local_model: str = DEFAULT_LOCAL_MODEL,
				hf_model: str = DEFAULT_HF_MODEL, 
				history_dir: str = "history") -> None:
	"""Run the application in interactive mode."""
	print_system(f"RAG Query Application - Interactive Mode (Project: {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR})")
	print_system("Type 'help' to see available commands")
	
	# Initialize variables
	current_project = project
	current_documents = documents
	current_embedding_config = embedding_config or get_project_embedding_config(project, document_dir, debug)
	current_llm_type = llm_type
	current_local_model = local_model
	current_hf_model = hf_model
	
	# Initialize history
	history = CommandHistory(history_dir=history_dir)
	
	# Initialize embedding provider cache
	provider_cache = EmbeddingProviderCache(debug=debug)
	
	# Function to get the current model name based on LLM type
	def get_current_model_name():
		if current_llm_type == LLM_LOCAL:
			return current_local_model
		elif current_llm_type == LLM_HF:
			return current_hf_model
		else:
			return "API"
	
	# Print the initial help info
	print_help_info(current_project, current_llm_type, get_current_model_name())
	
	while True:
		try:
			# Print the prompt with the current project and LLM highlighted
			current_model = get_current_model_name()
			prompt = f"\n{QUERY_COLOR}Enter your question [{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{QUERY_COLOR}] [{HIGHLIGHT_COLOR}{current_llm_type}:{current_model}{RESET_COLOR}{QUERY_COLOR}]: {RESET_COLOR}"
			
			# Use print and input separately to ensure proper scrolling behavior
			print(prompt, end='', flush=True)
			query = input().strip()
			
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
				print_help_info(current_project, current_llm_type, get_current_model_name())
				continue
			
			# Handle special commands
			if query.lower() == 'history':
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
			
			# Handle other special commands
			if query.lower() == 'projects':
				projects = discover_projects(index_dir)
				print_system("\nAvailable Projects:")
				for p in projects:
					marker = "*" if p == current_project else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{p}{RESET_COLOR}")
				continue
			
			elif query.lower() == 'config':
				print_system("\nCurrent Embedding Configuration:")
				print_system(f"Project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
				print_system(f"Embedding Type: {current_embedding_config.embedding_type}")
				print_system(f"Embedding Model: {current_embedding_config.model_name}")
				print_system(f"Embedding Dimensions: {current_embedding_config.dimensions}")
				
				# Show config file path
				config_path = get_project_config_path(current_project, document_dir)
				if os.path.exists(config_path):
					print_system(f"Config File: {config_path}")
				else:
					print_system("Config File: Not found (using defaults)")
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
				success = index_project(current_project, document_dir, index_dir, debug)
				
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
				
			elif query.lower() == 'models':
				# List both llm and Hugging Face models
				print_system("\nCurrent LLM Settings:")
				print_system(f"  Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				
				if current_llm_type == LLM_LOCAL:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
				elif current_llm_type == LLM_HF:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
				
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
					print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				
				elif llm_choice == LLM_LOCAL:
					current_llm_type = LLM_LOCAL
					if model_arg:
						current_local_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
					else:
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
				
				elif llm_choice == LLM_HF:
					current_llm_type = LLM_HF
					if model_arg:
						current_hf_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
					else:
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
				
				else:
					print_error(f"Unknown LLM type: {llm_choice}")
					print_system("Valid options are:")
					print_system("  llm claude")
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
							current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
							
							# Preload the embedding provider for the new project
							provider_cache.get_provider(current_project, document_dir, current_embedding_config)
							
							print_system(f"Switched to project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
					continue
				
				# Load the new project
				new_documents = load_index(index_path, backup_dir, debug)
				if new_documents:
					current_project = new_project
					current_documents = new_documents
					# Load the project's embedding configuration
					current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
					
					# Preload the embedding provider for the new project
					provider_cache.get_provider(current_project, document_dir, current_embedding_config)
					
					print_system(f"Switched to project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
					print_system(f"Embedding Type: {current_embedding_config.embedding_type}")
					print_system(f"Embedding Model: {current_embedding_config.model_name}")
					print_system(f"Embedding Dimensions: {current_embedding_config.dimensions}")

				else:
					print_system(f"No documents found in project: {HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}")
				continue
			
			# Regular query - search for relevant documents
			# Echo the query if it's not a command
			if not is_command(query):
				# Add a blank line after the query for better readability
				print()
				print_system("Searching for relevant documents...")
				
				# Pass the provider cache to the search function
				relevant_docs = search_documents(
					query, current_documents, current_project, 
					document_dir, current_embedding_config, 
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
				print_system(f"Generating answer with {current_llm_type} {model_name} ...")
				answer = get_response(
					query, relevant_docs, api_key, current_project,
					current_llm_type, model_name, debug, prompts_dir
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


				
def print_help_info(current_project: str, current_llm_type: str, current_model: str) -> None:
				"""
				Print help information about available commands.
				
				Args:
					current_project: The current project name
					current_llm_type: The current LLM type
					current_model: The current model name
				"""
				print_system(f"\nRAG Query Application - Help")
				print_system(f"Current Project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
				print_system(f"Current LLM: {HIGHLIGHT_COLOR}{current_llm_type}:{current_model}{RESET_COLOR}")
				
				print_system("\nAvailable Commands:")
				print_system("  help                     Show this help information")
				print_system("  exit, quit               End the session")
				
				# Project commands
				print_system("\nProject Commands:")
				print_system("  projects                 List all available projects")
				print_system("  project <name>           Switch to a different project")
				print_system("  config                   Show current embedding configuration")
				
				# Index commands
				print_system("\nIndex Commands:")
				print_system("  index                    Re-index the current project")
				print_system("  index clear              Clear the current project's index")
				
				# LLM commands
				print_system("\nLLM Commands:")
				print_system("  models                   List available LLM models")
				print_system("  llm claude               Use Claude API")
				print_system("  llm local [model_name]   Use a local model via llm library")
				print_system("  llm hf [model_name]      Use a Hugging Face model")
				
				# History commands
				print_system("\nHistory Commands:")
				print_system("  history                  Show command history")
				print_system("  history clear            Clear command history")
				print_system("  history save             Save history to a file")
				
				print_system("\nFor any other input, the application will treat it as a query")
				print_system("and search for relevant documents to help answer it.")
				
								
				
def get_response(query: str, relevant_docs: List[Document], api_key: str, project: str, 
							llm_type: str = LLM_CLAUDE, model_name: str = None,
							debug: bool = False, prompts_dir: str = PROMPTS_DIR) -> str:
	"""
	Get a response using the selected LLM.
	
	Args:
		query: User's query
		relevant_docs: Relevant documents
		api_key: API key (for Claude)
		project: Project name
		llm_type: Type of LLM to use ('claude', 'local', or 'hf')
		model_name: Which model to use (depends on llm_type)
		debug: Enable debug mode
		prompts_dir: Directory to save prompts
		
	Returns:
		Response from the selected LLM
	"""
	if llm_type.lower() == LLM_CLAUDE:
		if debug:
			print_debug("Using Claude API for response")
		return ask_claude(query, relevant_docs, api_key, project, debug, prompts_dir)
	
	elif llm_type.lower() == LLM_LOCAL:
		local_model = model_name or DEFAULT_LOCAL_MODEL
		if debug:
			print_debug(f"Using Simon Willison's llm with model: {local_model}")
		try:
			return ask_local_llm(query, relevant_docs, project, local_model, debug, prompts_dir)
		except Exception as e:
			if debug:
				print_debug(f"Error with local llm: {e}, falling back to Hugging Face")
			# If Simon's llm fails, fall back to Hugging Face
			return ask_local_hf(query, relevant_docs, project, DEFAULT_HF_MODEL, debug, prompts_dir)
	
	elif llm_type.lower() == LLM_HF:
		hf_model = model_name or DEFAULT_HF_MODEL
		if debug:
			print_debug(f"Using Hugging Face transformers with model: {hf_model}")
		return ask_local_hf(query, relevant_docs, project, hf_model, debug, prompts_dir)
	
	else:
		print_error(f"Unknown LLM type: {llm_type}. Using Claude as fallback.")
		return ask_claude(query, relevant_docs, api_key, project, debug, prompts_dir)					


				
def main():
	"""Main entry point for the query application."""
	parser = argparse.ArgumentParser(description="RAG Query Application with Project Support")
	
	parser.add_argument("--api-key", type=str, help="Anthropic API key")
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
						help="LLM to use: 'claude' (default, uses API), 'local' (Simon Willison's llm), or 'hf' (Hugging Face)")
	parser.add_argument("--local-model", type=str, default=DEFAULT_LOCAL_MODEL,
						help="Local model to use when --llm=local (default: gpt4all)")
	parser.add_argument("--hf-model", type=str, default=DEFAULT_HF_MODEL,
						help="Hugging Face model to use when --llm=hf")
	parser.add_argument("--history-dir", type=str, default="history",
						help="Directory to store command history")


	
	args = parser.parse_args()
	
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
		success = index_project(args.project, args.document_dir, args.index_dir, args.debug)
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


	# THIS MIGHT NOT NOW BE NEEDED
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
	
	# In the main function:
	# In the main function:
	if args.query:
		# Single query mode
		# Echo the query
		print(f"{QUERY_COLOR}{args.query}{RESET_COLOR}\n")
		
		print_system("Searching for relevant documents...")
		relevant_docs = search_documents(
			args.query, documents, args.project, 
			args.document_dir, embedding_config, 
			debug=args.debug, provider_cache=provider_cache
		)
		
		# If we found relevant documents, confirm before querying
		if relevant_docs and not args.debug:
			proceed = input(f"{SYSTEM_COLOR}Proceed with query using these sources? (Y/n): {RESET_COLOR}").strip().lower()
			if proceed == 'n':
				print_system("Query canceled")
				sys.exit(0)
		
		# Determine the model name based on LLM type
		model_name = None
		if args.llm == LLM_LOCAL:
			model_name = args.local_model
		elif args.llm == LLM_HF:
			model_name = args.hf_model
		
		# Use the selected LLM
		print_system(f"Generating answer with {args.llm} {model_name} ...")
		answer = get_response(
			args.query, relevant_docs, api_key, args.project,
			args.llm, model_name, args.debug, prompts_dir
		)
		
		# Print the answer with proper colors
		print(f"\n{ANSWER_COLOR}Answer:{RESET_COLOR}")
		print(f"{ANSWER_COLOR}{answer}{RESET_COLOR}")	
	
	else:
		# Interactive mode
		interactive_mode(
			documents, api_key, args.project, 
			args.document_dir, args.index_dir, 
			embedding_config, args.debug, prompts_dir,
			args.llm, args.local_model, args.hf_model,
			args.history_dir
		)

if __name__ == "__main__":
	main()