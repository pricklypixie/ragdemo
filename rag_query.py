#!/usr/bin/env python3
"""
RAG Query Application with Project Support

This application:
1. Loads document indexes created by the document_indexer
2. Supports querying specific projects or the master index
3. Retrieves relevant documents based on the query
4. Sends the query and context to Claude for answering
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
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Force CPU usage instead of Metal on MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_FALLBACK_POLICY"] = "0" 

import anthropic
from sklearn.metrics.pairwise import cosine_similarity

# Constants
MODEL = "claude-3-opus-20240229"
MAX_TOKENS = 4096
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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


def get_embedding_model(model_name: str, debug: bool = False):
	"""Load and return the embedding model."""
	try:
		# Import here to ensure environment variables take effect
		import torch
		from sentence_transformers import SentenceTransformer
		
		if debug:
			print(f"[DEBUG] PyTorch version: {torch.__version__}")
			print(f"[DEBUG] Loading model: {model_name} on CPU")
		
		# Force CPU
		model = SentenceTransformer(model_name, device="cpu")
		
		if debug:
			print(f"[DEBUG] Model loaded successfully")
			
		return model
	except ImportError as e:
		print(f"Error: Required package not installed - {e}")
		print("Please install with: pip install sentence-transformers torch")
		sys.exit(1)
	except Exception as e:
		print(f"Error loading model: {e}")
		if debug:
			print(traceback.format_exc())
		sys.exit(1)


def create_embedding(model, text: str, debug: bool = False) -> List[float]:
	"""Generate an embedding for the given text."""
	if debug:
		print(f"[DEBUG] Generating embedding for text of length {len(text)}")
	
	try:
		import torch
		with torch.no_grad():
			# Process a single item, no batching
			embedding = model.encode(
				text,
				convert_to_numpy=True,
				show_progress_bar=False,
				batch_size=1
			).tolist()
		
		if debug:
			print(f"[DEBUG] Generated embedding with dimension {len(embedding)}")
			
		return embedding
	except Exception as e:
		print(f"Error generating embedding: {e}")
		if debug:
			print(traceback.format_exc())
		return []


def search_documents(query: str, documents: List[Document], model, 
					 top_k: int = TOP_K_DOCUMENTS, debug: bool = False) -> List[Document]:
	"""Search for documents relevant to the query."""
	if not documents:
		print("No documents in index")
		return []
	
	if debug:
		print(f"[DEBUG] Searching for: '{query}'")
	
	# Create query embedding
	try:
		start_time = time.time()
		query_embedding = create_embedding(model, query, debug)
		search_time = time.time() - start_time
		if debug:
			print(f"[DEBUG] Created query embedding in {search_time:.2f} seconds")
		
		# Calculate similarities
		similarities = []
		for doc in documents:
			if doc.embedding:
				# Calculate cosine similarity
				sim = cosine_similarity(
					[query_embedding], 
					[doc.embedding]
				)[0][0]
				similarities.append((doc, sim))
		
		# Sort by similarity (highest first) and take top k
		sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
		top_results = [doc for doc, sim in sorted_results[:top_k]]
		
		if debug:
			print(f"[DEBUG] Found {len(top_results)} relevant documents")
			for i, (doc, sim) in enumerate(sorted_results[:top_k]):
				project = doc.metadata.get('project', MASTER_PROJECT)
				print(f"[DEBUG]   Result {i+1}: {doc.metadata.get('file_path')} "
					  f"(project: {project}, score: {sim:.4f})")
		
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


def interactive_mode(documents: List[Document], model, api_key: str, project: str, debug: bool = False) -> None:
	"""Run the application in interactive mode."""
	print(f"RAG Query Application - Interactive Mode (Project: {project})")
	print("Enter 'exit' or 'quit' to end the session")
	print("Enter 'project <name>' to switch projects")
	print("Enter 'projects' to list available projects")
	
	current_project = project
	current_documents = documents
	
	while True:
		try:
			query = input("\nEnter your question: ").strip()
			
			if query.lower() in ['exit', 'quit']:
				print("Exiting...")
				break
			
			if not query:
				continue
			
			# Handle project switching
			if query.lower() == 'projects':
				projects = discover_projects(DEFAULT_INDEX_DIR)
				print("\nAvailable Projects:")
				for p in projects:
					marker = "*" if p == current_project else " "
					print(f"{marker} {p}")
				continue
			
			if query.lower().startswith('project '):
				new_project = query[8:].strip()
				index_path, backup_dir = get_index_path(DEFAULT_INDEX_DIR, new_project)
				
				if not os.path.exists(index_path):
					print(f"Project '{new_project}' not found or not indexed")
					continue
				
				# Load the new project
				new_documents = load_index(index_path, backup_dir, debug)
				if new_documents:
					current_project = new_project
					current_documents = new_documents
					print(f"Switched to project: {current_project}")
				else:
					print(f"No documents found in project: {new_project}")
				continue
			
			# Search for relevant documents
			relevant_docs = search_documents(query, current_documents, model, debug=debug)
			
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
	parser.add_argument("--query", type=str, 
						help="Single query mode: ask a question and exit")
	parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
						help="Sentence Transformer model to use for embeddings")
	parser.add_argument("--debug", action="store_true",
						help="Enable debug logging")
	parser.add_argument("--project", type=str, default=MASTER_PROJECT,
						help="Project to query (default: master)")
	parser.add_argument("--list-projects", action="store_true",
						help="List all available projects")
	
	args = parser.parse_args()
	
	# Get API key from args or environment
	api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
	
	if not api_key:
		print("Error: Anthropic API key is required. Please provide it via --api-key or set the ANTHROPIC_API_KEY environment variable.")
		sys.exit(1)
	
	# Check if index directory exists
	if not os.path.exists(args.index_dir):
		print(f"Error: Index directory not found: {args.index_dir}")
		print("Please run the indexer first to create an index.")
		sys.exit(1)
	
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
				print(f"  {project} ({len(documents)} documents)")
			except:
				print(f"  {project} (error loading index)")
		return
	
	# Get index path for the specified project
	index_path, backup_dir = get_index_path(args.index_dir, args.project)
	
	if not os.path.exists(index_path):
		print(f"Error: Index for project '{args.project}' not found: {index_path}")
		# List available projects
		projects = discover_projects(args.index_dir)
		if projects:
			print("\nAvailable Projects:")
			for project in projects:
				print(f"  {project}")
		sys.exit(1)
	
	# Print application info
	print(f"RAG Query Application with Project Support")
	print(f"Python version: {sys.version}")
	print(f"Using embedding model: {args.embedding_model}")
	print(f"Project: {args.project}")
	print(f"Index location: {index_path}")
	
	try:
		print(f"Anthropic SDK version: {anthropic.__version__}")
	except AttributeError:
		print("Anthropic SDK version: unknown")
	
	# Load embedding model
	print("Loading embedding model...")
	model = get_embedding_model(args.embedding_model, args.debug)
	
	# Load document index for the project
	documents = load_index(index_path, backup_dir, args.debug)
	
	if not documents:
		print(f"No documents found in the project index. Please run the indexer first.")
		sys.exit(1)
	
	if args.query:
		# Single query mode
		relevant_docs = search_documents(args.query, documents, model, debug=args.debug)
		answer = ask_claude(args.query, relevant_docs, api_key, args.project, args.debug)
		print(answer)
	else:
		# Interactive mode
		interactive_mode(documents, model, api_key, args.project, args.debug)


if __name__ == "__main__":
	main()