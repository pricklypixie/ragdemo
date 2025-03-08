#!/usr/bin/env python3
"""
RAG Query Application with Project Support

This application:
1. Loads document indexes created by the document_indexer
2. Supports querying specific projects or the master index
3. Retrieves relevant documents based on the query
4. Sends the query and context to Claude for answering
5. Supports on-demand indexing of projects
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

# Force CPU usage instead of Metal on MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_FALLBACK_POLICY"] = "0" 

import anthropic
from sklearn.metrics.pairwise import cosine_similarity

# Import our embedding library
from embeddings import EmbeddingConfig, get_embedding_provider, load_project_config

# Constants
MODEL = "claude-3-opus-20240229"
MAX_TOKENS = 4096
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_TYPE = "sentence_transformers"
TOP_K_DOCUMENTS = 3
API_TIMEOUT = 60  # Timeout for API calls in seconds
MASTER_PROJECT = "master"  # Name for the master index


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
		print(f"Saved embedding configuration to {config_path}")
	except Exception as e:
		print(f"Error saving embedding configuration: {e}")


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
		print(f"Error discovering projects: {e}")
	
	return projects


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


def get_project_embedding_config(project: str, document_dir: str, debug: bool = False) -> EmbeddingConfig:
	"""Get the embedding configuration for a project, loading from project config if available."""
	config_path = get_project_config_path(project, document_dir)
	
	if os.path.exists(config_path):
		try:
			config = EmbeddingConfig.from_json_file(config_path)
			if debug:
				print(f"[DEBUG] Loaded project embedding config from: {config_path}")
				print(f"[DEBUG] Embedding type: {config.embedding_type}")
				print(f"[DEBUG] Embedding model: {config.model_name}")
			return config
		except Exception as e:
			print(f"Error loading project config, using defaults: {e}")
	else:
		if debug:
			print(f"[DEBUG] No project config found at {config_path}, using defaults")
	
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
		print(f"Created default embedding configuration at {config_path}")
	
	# Get the embedding config for the project
	config = get_project_embedding_config(project, document_dir, debug)
	
	# Check for document_indexer.py
	if not os.path.exists("document_indexer.py"):
		print("Error: document_indexer.py not found")
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


def search_documents(query: str, documents: List[Document], project: str, 
					 document_dir: str, embedding_config: Optional[EmbeddingConfig] = None,
					 top_k: int = TOP_K_DOCUMENTS, debug: bool = False) -> List[Document]:
	"""Search for documents relevant to the query."""
	if not documents:
		print("No documents in index")
		return []
	
	if debug:
		print(f"[DEBUG] Searching for: '{query}'")
	
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
	
	# If no embedding config provided, use the project's config
	if embedding_config is None:
		embedding_config = get_project_embedding_config(project, document_dir, debug)
	
	# Create query embedding with the project's embedding provider
	embedding_provider = get_embedding_provider(
		project_dir=project, 
		document_dir=document_dir, 
		config=embedding_config,
		debug=debug
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
			query_embedding = embedding_provider.create_embedding(query)
			search_time = time.time() - start_time
			
			if debug:
				print(f"[DEBUG] Created query embedding in {search_time:.2f} seconds")
				print(f"[DEBUG] Searching {len(document_groups[base_key])} documents with matching embedding model")
			
			# Calculate similarities for the base model group
			base_similarities = []
			for doc in document_groups[base_key]:
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
				print(f"Searching documents with different embedding models...")
				
				# For each different embedding type, we need a new provider
				for key, docs in document_groups.items():
					if key == base_key:
						continue  # Skip the base key we already processed
					
					embedding_type, model_name = key
					if debug:
						print(f"[DEBUG] Searching {len(docs)} documents with embedding type: {embedding_type}, model: {model_name}")
					
					# Create a temporary provider for this embedding type
					temp_config = EmbeddingConfig(embedding_type=embedding_type, model_name=model_name)
					temp_provider = get_embedding_provider(config=temp_config, debug=debug)
					
					# Create query embedding with this provider
					start_time = time.time()
					temp_query_embedding = temp_provider.create_embedding(query)
					search_time = time.time() - start_time
					
					if debug:
						print(f"[DEBUG] Created query embedding with {model_name} in {search_time:.2f} seconds")
					
					# Calculate similarities
					other_similarities = []
					for doc in docs:
						if doc.embedding:
							sim = cosine_similarity(
								[temp_query_embedding], 
								[doc.embedding]
							)[0][0]
							other_similarities.append((doc, sim))
					
					all_results.extend(other_similarities)
		
		# Sort all results by similarity and take top k
		sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
		top_results = [doc for doc, sim in sorted_results[:top_k]]
		
		# In debug mode, print details about the relevant documents
		if debug:
			print(f"[DEBUG] Found {len(top_results)} relevant documents:")
			for i, (doc, sim) in enumerate(sorted_results[:top_k]):
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
		
		return top_results
		
	except Exception as e:
		print(f"Error during search: {e}")
		if debug:
			print(traceback.format_exc())
		return []


def ask_claude(query: str, relevant_docs: List[Document], api_key: str, project: str, debug: bool = False) -> str:
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
			print("[DEBUG] Sending prompt to Claude")
		
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
			print(f"[DEBUG] Received response from Claude in {elapsed_time:.2f} seconds")
		
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


def interactive_mode(documents: List[Document], api_key: str, project: str, 
					document_dir: str, index_dir: str, 
					embedding_config: Optional[EmbeddingConfig] = None,
					debug: bool = False) -> None:
	"""Run the application in interactive mode."""
	print(f"RAG Query Application - Interactive Mode (Project: {project})")
	print("Enter 'exit' or 'quit' to end the session")
	print("Enter 'project <name>' to switch projects")
	print("Enter 'projects' to list available projects")
	print("Enter 'index' to re-index the current project")
	print("Enter 'config' to show current embedding configuration")
	
	current_project = project
	current_documents = documents
	current_embedding_config = embedding_config or get_project_embedding_config(project, document_dir, debug)
	
	while True:
		try:
			query = input("\nEnter your question: ").strip()
			
			if query.lower() in ['exit', 'quit']:
				print("Exiting...")
				break
			
			if not query:
				continue
			
			# Handle special commands
			if query.lower() == 'projects':
				projects = discover_projects(index_dir)
				print("\nAvailable Projects:")
				for p in projects:
					marker = "*" if p == current_project else " "
					print(f"{marker} {p}")
				continue
			
			elif query.lower() == 'config':
				print("\nCurrent Embedding Configuration:")
				print(f"Project: {current_project}")
				print(f"Embedding Type: {current_embedding_config.embedding_type}")
				print(f"Embedding Model: {current_embedding_config.model_name}")
				
				# Show config file path
				config_path = get_project_config_path(current_project, document_dir)
				if os.path.exists(config_path):
					print(f"Config File: {config_path}")
				else:
					print("Config File: Not found (using defaults)")
				continue
			
			elif query.lower() == 'index':
				print(f"\nIndexing project: {current_project}")
				success = index_project(current_project, document_dir, index_dir, debug)
				
				if success:
					# Reload the project index
					index_path, backup_dir = get_index_path(index_dir, current_project)
					current_documents = load_index(index_path, backup_dir, debug)
					
					# Reload the embedding config
					current_embedding_config = get_project_embedding_config(current_project, document_dir, debug)
					
					print(f"Project '{current_project}' re-indexed successfully")
					print(f"Loaded {len(current_documents)} documents")
				continue
			
			elif query.lower().startswith('project '):
				new_project = query[8:].strip()
				index_path, backup_dir = get_index_path(index_dir, new_project)
				
				if not os.path.exists(index_path):
					print(f"Project '{new_project}' not found or not indexed")
					# Ask if user wants to create it
					create = input(f"Would you like to create and index project '{new_project}'? (y/n): ").strip().lower()
					if create == 'y':
						# Create the project directory if needed
						if new_project != MASTER_PROJECT:
							project_dir = os.path.join(document_dir, new_project)
							os.makedirs(project_dir, exist_ok=True)
							print(f"Created project directory: {project_dir}")
						
						# Index the new project
						success = index_project(new_project, document_dir, index_dir, debug)
						if success:
							current_project = new_project
							current_documents = load_index(index_path, backup_dir, debug)
							current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
							print(f"Switched to project: {current_project}")
					continue
				
				# Load the new project
				new_documents = load_index(index_path, backup_dir, debug)
				if new_documents:
					current_project = new_project
					current_documents = new_documents
					# Load the project's embedding configuration
					current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
					print(f"Switched to project: {current_project}")
					print(f"Embedding Type: {current_embedding_config.embedding_type}")
					print(f"Embedding Model: {current_embedding_config.model_name}")
				else:
					print(f"No documents found in project: {new_project}")
				continue
			
			# Regular query - search for relevant documents
			relevant_docs = search_documents(
				query, current_documents, current_project, 
				document_dir, current_embedding_config, debug=debug
			)
			
			# Ask Claude
			answer = ask_claude(query, relevant_docs, api_key, current_project, debug)
			
			print("\nAnswer:")
			print(answer)
		except KeyboardInterrupt:
			print("\nInterrupted by user. Exiting...")
			break
		except Exception as e:
			print(f"Error: {e}")
			if debug:
				print(traceback.format_exc())


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
	parser.add_argument("--index", action="store_true",
						help="Index the specified project before querying")
	
	args = parser.parse_args()
	
	# Get API key from args or environment
	api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
	
	if not api_key:
		print("Error: Anthropic API key is required. Please provide it via --api-key or set the ANTHROPIC_API_KEY environment variable.")
		sys.exit(1)
	
	# Check if document directory exists
	if not os.path.exists(args.document_dir):
		print(f"Error: Document directory not found: {args.document_dir}")
		print("Please create the document directory and add your files.")
		sys.exit(1)
	
	# Create the index directory if it doesn't exist
	os.makedirs(args.index_dir, exist_ok=True)
	
	# Just list projects if requested
	if args.list_projects:
		projects = discover_projects(args.index_dir)
		if not projects:
			print("No indexed projects found.")
			return
			
		print("\nAvailable Projects:")
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
				print(f"  {project} ({len(documents)} documents, {emb_info})")
				
				# Show config file path and info
				config_path = get_project_config_path(project, args.document_dir)
				if os.path.exists(config_path):
					try:
						config = EmbeddingConfig.from_json_file(config_path)
						print(f"    Config: {config.embedding_type}/{config.model_name}")
					except:
						print(f"    Config: Error loading {config_path}")
				else:
					print(f"    Config: Not found (using defaults)")
					
			except:
				print(f"  {project} (error loading index)")
		return
	
	# Create embedding configuration from command line args
	embedding_config = None
	if args.embedding_model or args.embedding_type:
		embedding_config = EmbeddingConfig(
			embedding_type=args.embedding_type or DEFAULT_EMBEDDING_TYPE,
			model_name=args.embedding_model or DEFAULT_EMBEDDING_MODEL
		)
	
	# Handle indexing if requested
	if args.index:
		print(f"Indexing project: {args.project}")
		success = index_project(args.project, args.document_dir, args.index_dir, args.debug)
		if not success:
			print("Indexing failed. Exiting.")
			sys.exit(1)
	
	# Get index path for the specified project
	index_path, backup_dir = get_index_path(args.index_dir, args.project)
	
	# Check if the index exists
	if not os.path.exists(index_path):
		print(f"Index for project '{args.project}' not found: {index_path}")
		# Ask if user wants to create it
		create = input(f"Would you like to create and index project '{args.project}'? (y/n): ").strip().lower()
		if create == 'y':
			# Create the project directory if needed
			if args.project != MASTER_PROJECT:
				project_dir = os.path.join(args.document_dir, args.project)
				os.makedirs(project_dir, exist_ok=True)
				print(f"Created project directory: {project_dir}")
			
			# Index the project
			success = index_project(args.project, args.document_dir, args.index_dir, args.debug)
			if not success:
				print("Indexing failed. Exiting.")
				sys.exit(1)
		else:
			# List available projects
			projects = discover_projects(args.index_dir)
			if projects:
				print("\nAvailable Projects:")
				for project in projects:
					print(f"  {project}")
			sys.exit(1)
	
	# If no custom embedding config was provided, use the project's config
	if embedding_config is None:
		embedding_config = get_project_embedding_config(args.project, args.document_dir, args.debug)
	
	# Print application info
	print(f"RAG Query Application with Project Support")
	print(f"Python version: {sys.version}")
	print(f"Project: {args.project}")
	print(f"Embedding Type: {embedding_config.embedding_type}")
	print(f"Embedding Model: {embedding_config.model_name}")
	print(f"Index location: {index_path}")
	
	try:
		print(f"Anthropic SDK version: {anthropic.__version__}")
	except AttributeError:
		print("Anthropic SDK version: unknown")
	
	# Load document index for the project
	documents = load_index(index_path, backup_dir, args.debug)
	
	if not documents:
		print(f"No documents found in the project index. Please add documents and run indexing.")
		sys.exit(1)
	
	# Display information about the embeddings in the index
	if args.debug:
		embedding_types = {}
		for doc in documents:
			emb_type = doc.metadata.get('embedding_type', 'unknown')
			emb_model = doc.metadata.get('embedding_model', 'unknown')
			key = f"{emb_type}/{emb_model}"
			embedding_types[key] = embedding_types.get(key, 0) + 1
		
		print("\n[DEBUG] Embedding types in index:")
		for key, count in embedding_types.items():
			print(f"[DEBUG]   {key}: {count} documents")
	
	if args.query:
		# Single query mode
		relevant_docs = search_documents(args.query, documents, args.project, 
										args.document_dir, embedding_config, debug=args.debug)
		answer = ask_claude(args.query, relevant_docs, api_key, args.project, args.debug)
		print(answer)
	else:
		# Interactive mode
		interactive_mode(documents, api_key, args.project, 
						args.document_dir, args.index_dir, 
						embedding_config, args.debug)


if __name__ == "__main__":
	main()