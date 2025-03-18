#!/usr/bin/env python3
"""
Document Indexer for RAG Applications with Project Support

This tool:
1. Indexes documents from a local directory
2. Creates embeddings using the configurable embedding library
3. Supports project-based indexing (subdirectories as separate projects)
4. Saves separate indexes for each project and a master index
"""

import os
import sys
import json
import argparse
import glob
import pickle
import numpy as np
import time
import traceback
import gc
import warnings
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path



# Import our custom embedding library
from embeddings import EmbeddingConfig, get_embedding_provider, load_project_config

# Filter resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker")

# Constants
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
MAX_CHUNK_SIZE = 1500  # Characters
MASTER_PROJECT = "master"  # Name for the master index


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

def split_into_paragraphs(text: str) -> List[str]:
	"""Split text into paragraphs based on double newlines."""
	# Handle different line ending styles
	text = text.replace('\r\n', '\n')
	text = text.replace('\n', '\n\n')
	text = text.replace('\n\n\n', '\n\n')
	
	# Split on paragraph breaks (double newlines)
	paragraphs = text.split('\n\n')
	
	# Filter out empty paragraphs and strip whitespace
	paragraphs = [p.strip() for p in paragraphs if p.strip()]
	
	return paragraphs


def create_paragraph_chunks(text: str, max_chunk_size: int, debug: bool = False) -> List[str]:
	"""
	Create chunks by grouping paragraphs with overlap between chunks.
	Each chunk (except the first) begins with the last paragraph of the previous chunk.
	"""
	if debug:
		print(f"[DEBUG] Creating paragraph chunks for text of length {len(text)}")
	
	# If text is smaller than max chunk size, return as is
	if len(text) <= max_chunk_size:
		if debug:
			print(f"[DEBUG] Text smaller than max chunk size, returning as single chunk")
		return [text]
	
	# Split into paragraphs
	paragraphs = split_into_paragraphs(text)
	if debug:
		print(f"[DEBUG] Split text into {len(paragraphs)} paragraphs")
	
	chunks = []
	current_chunk = []
	current_size = 0
	last_paragraph = None
	i = 0
	
	while i < len(paragraphs):
		paragraph = paragraphs[i]
		
		# If this is the first paragraph of a new chunk and we have a previous chunk's
		# last paragraph, add it first (for overlap)
		if not current_chunk and last_paragraph is not None:
			current_chunk.append(last_paragraph)
			current_size = len(last_paragraph)
			if debug:
				print(f"[DEBUG] Starting chunk with overlap paragraph: {len(last_paragraph)} chars")
		
		# Calculate size with this paragraph added
		new_size = current_size
		if current_chunk:  # Add separator size if there's already content
			new_size += 4  # For '\n\n'
		new_size += len(paragraph)
		
		# Check if adding this paragraph would exceed the max size and we already have content
		if new_size > max_chunk_size and current_chunk:
			# Remember the last paragraph for the next chunk
			last_paragraph = current_chunk[-1]
			
			# Save the current chunk
			chunks.append('\n\n'.join(current_chunk))
			if debug:
				print(f"[DEBUG] {current_chunk}")
				print(f"[DEBUG] Created chunk {len(chunks)} with {current_size} chars and {len(current_chunk)} paragraphs")
				
				# Bad fix for files that do not chunk properly
				# If more than 100 chunks, abort
				
				if len(chunks) > 100:
					break
			
			# Start a new chunk
			current_chunk = []
			current_size = 0
			
			# Don't increment i - we'll reconsider this paragraph for the next chunk
			continue
		
		# Special case: If a single paragraph is larger than max_chunk_size,
		# we'll have to include it as its own chunk
		if len(paragraph) > max_chunk_size and not current_chunk:
			if debug:
				print(f"[DEBUG] WARNING: Paragraph {i+1} is larger than max_chunk_size ({len(paragraph)} > {max_chunk_size})")
			chunks.append(paragraph)
			if debug:
				print(f"[DEBUG] Created chunk {len(chunks)} with {len(paragraph)} chars (single large paragraph)")
			last_paragraph = paragraph
			i += 1
			continue
		
		# Add the paragraph to the current chunk
		current_chunk.append(paragraph)
		current_size = new_size
		i += 1
	
	# Don't forget to save the last chunk if there's anything left
	if current_chunk:
		chunks.append('\n\n'.join(current_chunk))
		if debug:
			print(f"[DEBUG] Created final chunk {len(chunks)} with {current_size} chars and {len(current_chunk)} paragraphs")
	
	if debug:
		print(f"[DEBUG] Created a total of {len(chunks)} chunks")
	
	return chunks


def get_project_path(file_path: str, document_dir: str) -> str:
	"""
	Determine the project for a file based on its subdirectory.
	Returns MASTER_PROJECT if the file is in the root document directory.
	"""
	rel_path = os.path.relpath(file_path, document_dir)
	parts = Path(rel_path).parts
	
	if len(parts) <= 1:
		# File is in the root documents directory
		return MASTER_PROJECT
	else:
		# File is in a subdirectory - use the first subdirectory as project name
		return parts[0]


def get_index_path(index_dir: str, project: str) -> Tuple[str, str]:
	"""Get the index path and backup directory for a project."""
	if project == MASTER_PROJECT:
		index_path = os.path.join(index_dir, "document_index.pkl")
	else:
		# Create project subdirectory in the index directory
		project_dir = os.path.join(index_dir, project)
		os.makedirs(project_dir, exist_ok=True)
		index_path = os.path.join(project_dir, "document_index.pkl")
	
	# Create backup directory
	backup_dir = os.path.join(os.path.dirname(index_path), "backups")
	os.makedirs(backup_dir, exist_ok=True)
	
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


def save_index(documents: List[Document], index_path: str, backup_dir: str, debug: bool = False) -> None:
	"""Save the document index to disk."""
	# Create a backup of the current index
	if os.path.exists(index_path):
		backup_file = os.path.join(
			backup_dir, 
			f"document_index_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
		)
		try:
			with open(index_path, 'rb') as src, open(backup_file, 'wb') as dst:
				dst.write(src.read())
			if debug:
				print(f"[DEBUG] Created backup: {backup_file}")
		except Exception as e:
			print(f"Error creating backup: {e}")
	
	# Save the updated index
	try:
		with open(index_path, 'wb') as f:
			pickle.dump(documents, f)
		print(f"Saved {len(documents)} documents to index: {index_path}")
	except Exception as e:
		print(f"Error saving index: {e}")


def index_file(file_path: str, project_dir: str, document_dir: str, 
			   project_indexes: Dict[str, List[Document]], 
			   max_chunk_size: int, embedding_config: Optional[EmbeddingConfig] = None,
			   debug: bool = False) -> None:
	"""
	Index a single file by creating paragraph-based chunks with overlap and generating embeddings.
	Updates both the project-specific index and the master index.
	"""
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			content = f.read()
		
		rel_path = os.path.relpath(file_path, document_dir)
		file_name = os.path.basename(file_path)
		file_size = os.path.getsize(file_path)
		
		# Get the project for this file
		project = get_project_path(file_path, document_dir)
		
		if debug:
			print(f"[DEBUG] Processing file: {rel_path} (project: {project}, size: {file_size} bytes)")
		
		# Get file stats
		stats = os.stat(file_path)
		modified_time = datetime.fromtimestamp(stats.st_mtime)
		
		# Ensure both project and master indexes exist in our dictionary
		if project not in project_indexes:
			project_indexes[project] = []
		if MASTER_PROJECT not in project_indexes:
			project_indexes[MASTER_PROJECT] = []
		
		# Check if file is already indexed in the project index
		project_docs = project_indexes[project]
		existing_docs = [doc for doc in project_docs 
					   if doc.metadata.get('file_path') == rel_path and
						  doc.metadata.get('last_modified') == modified_time.isoformat()]
		
		if existing_docs:
			print(f"File already indexed in project '{project}': {rel_path}")
			return
		
		# Check if file is already indexed in master (if project is not master)
		if project != MASTER_PROJECT:
			master_docs = project_indexes[MASTER_PROJECT]
			existing_master_docs = [doc for doc in master_docs 
							   if doc.metadata.get('file_path') == rel_path and
								  doc.metadata.get('last_modified') == modified_time.isoformat()]
			
			if existing_master_docs:
				print(f"File already indexed in master: {rel_path}")
				# Only need to update the project index in this case
				project_docs.extend(existing_master_docs)
				return
		
		# Remove any old versions of this file from both indexes
		project_indexes[project] = [doc for doc in project_indexes[project] 
								  if doc.metadata.get('file_path') != rel_path]
		
		if project != MASTER_PROJECT:
			project_indexes[MASTER_PROJECT] = [doc for doc in project_indexes[MASTER_PROJECT] 
											if doc.metadata.get('file_path') != rel_path]
		
		print(f"Indexing: {rel_path} (project: {project})")
		
		# Get the embedding provider for this project
		embedding_provider = get_embedding_provider(
			project_dir=project, 
			document_dir=document_dir, 
			config=embedding_config,
			debug=debug
		)
		
		if debug:
			print(f"[DEBUG] Using embedding model: {embedding_provider.config.model_name} "
				  f"(type: {embedding_provider.config.embedding_type})")
		
		# Use paragraph-based chunking with overlap
		chunks = create_paragraph_chunks(content, max_chunk_size, debug)
		print(f"  Split into {len(chunks)} chunks")
		
		# Store new document chunks
		new_docs = []
		
		for i, chunk in enumerate(chunks):
			print(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
			
			# Get paragraph info for this chunk
			paragraphs = split_into_paragraphs(chunk)
			
			metadata = {
				'file_path': rel_path,
				'file_name': file_name,
				'project': project,
				'embedding_model': embedding_provider.config.model_name,
				'embedding_type': embedding_provider.config.embedding_type,
				'chunk_index': i,
				'total_chunks': len(chunks),
				'chunk_size': len(chunk),
				'paragraphs': len(paragraphs),
				'last_modified': modified_time.isoformat()
			}
			
			# Generate embedding for the chunk
			embedding = embedding_provider.create_embedding(chunk)
			
			if embedding:
				doc = Document(content=chunk, metadata=metadata, embedding=embedding)
				new_docs.append(doc)
				print(f"  Successfully indexed chunk {i+1}/{len(chunks)}")
			else:
				print(f"  Failed to generate embedding for chunk {i+1}")
		
		# Add the new documents to both the project index and master index (if different)
		project_indexes[project].extend(new_docs)
		if project != MASTER_PROJECT:
			project_indexes[MASTER_PROJECT].extend(new_docs)
			
	except Exception as e:
		print(f"Error indexing file {file_path}: {e}")
		if debug:
			print(traceback.format_exc())


def discover_projects(document_dir: str) -> List[str]:
	"""Discover all projects (subdirectories) in the document directory."""
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


def index_directory(document_dir: str, index_dir: str, max_chunk_size: int,
					embedding_config: Optional[EmbeddingConfig] = None,
					project: Optional[str] = None, debug: bool = False) -> None:
	"""
	Index all supported documents in the specified directory.
	Can be limited to a specific project (subdirectory).
	"""
	# Load all project indexes first
	projects = [project] if project else discover_projects(document_dir)
	
	if debug:
		print(f"[DEBUG] Projects to index: {projects}")
	
	# Dictionary to hold indexes for each project
	project_indexes = {}
	
	# Load existing indexes for each project
	for proj in projects:
		index_path, backup_dir = get_index_path(index_dir, proj)
		project_indexes[proj] = load_index(index_path, backup_dir, debug)
	
	# Find all files to process
	if project and project != MASTER_PROJECT:
		# Index only files in the specified project subdirectory
		project_dir = os.path.join(document_dir, project)
		files = glob.glob(os.path.join(project_dir, "**/*.txt"), recursive=True)
		files += glob.glob(os.path.join(project_dir, "**/*.md"), recursive=True)
	else:
		# Index all files in the document directory
		files = glob.glob(os.path.join(document_dir, "**/*.txt"), recursive=True)
		files += glob.glob(os.path.join(document_dir, "**/*.md"), recursive=True)
	
	if not files:
		print(f"No supported files found to index")
		return
	
	print(f"Found {len(files)} files to index")
	
	# Sort files by size (smallest first) to get some quick wins
	files = sorted(files, key=os.path.getsize)
	
	for i, file_path in enumerate(files, 1):
		print(f"\nProcessing file {i}/{len(files)}: {file_path}")
		try:
			# Process the file
			index_file(file_path, project or MASTER_PROJECT, document_dir, 
					   project_indexes, max_chunk_size, embedding_config, debug)
			
			# Save project indexes after each file
			for proj, docs in project_indexes.items():
				if docs:  # Only save if there are documents
					index_path, backup_dir = get_index_path(index_dir, proj)
					save_index(docs, index_path, backup_dir, debug)
			
			# Force garbage collection
			gc.collect()
			
		except Exception as e:
			print(f"Failed to process {file_path}: {e}")
			if debug:
				print(traceback.format_exc())
	
	# Print summary
	print("\nIndexing Summary:")
	for proj, docs in project_indexes.items():
		print(f"Project '{proj}': {len(docs)} document chunks")


def main():
	"""Main entry point for the indexer application."""
	parser = argparse.ArgumentParser(description="Document Indexer for RAG Applications with Project Support")
	
	parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR, 
						help="Directory to store the document index")
	parser.add_argument("--document-dir", type=str, default=DEFAULT_DOCUMENT_DIR, 
						help="Directory containing documents to index")
	parser.add_argument("--embedding-type", type=str, default="sentence_transformers",
						help="Type of embedding to use (sentence_transformers, openai)")
	parser.add_argument("--embedding-model", type=str, 
						help="Embedding model to use")
	parser.add_argument("--max-chunk-size", type=int, default=MAX_CHUNK_SIZE,
						help="Maximum size of document chunks in characters")
	parser.add_argument("--debug", action="store_true",
						help="Enable debug logging")
	parser.add_argument("--file", type=str,
						help="Index a single file instead of a directory")
	parser.add_argument("--project", type=str,
						help="Index a specific project (subdirectory) only")
	parser.add_argument("--list-projects", action="store_true", 
						help="List all available projects")
	
	args = parser.parse_args()
	
	# Create directories
	os.makedirs(args.index_dir, exist_ok=True)
	
	# Just list projects if requested
	if args.list_projects:
		if not os.path.exists(args.document_dir):
			print(f"Error: Document directory not found: {args.document_dir}")
			return
		
		projects = discover_projects(args.document_dir)
		print("\nAvailable Projects:")
		for project in projects:
			# Check if this project has an index
			index_path, _ = get_index_path(args.index_dir, project)
			has_index = os.path.exists(index_path)
			status = "indexed" if has_index else "not indexed"
			
			print(f"  {project} ({status})")
		return
	
	# Create embedding configuration from command line args
	embedding_config = None
	if args.embedding_model or args.embedding_type:
		embedding_config = EmbeddingConfig(
			embedding_type=args.embedding_type,
			model_name=args.embedding_model
		)
	
	print(f"Document Indexer with Project Support")
	if embedding_config:
		print(f"Embedding type: {embedding_config.embedding_type}")
		print(f"Embedding model: {embedding_config.model_name}")
	print(f"Max chunk size: {args.max_chunk_size} characters")
	print(f"Index directory: {args.index_dir}")
	
	if args.file:
		# Index a single file
		if not os.path.exists(args.file):
			print(f"Error: File not found: {args.file}")
			return
		
		print(f"Indexing single file: {args.file}")
		
		# Get the project for this file
		project = get_project_path(args.file, args.document_dir)
		
		# Load both project and master indexes
		project_indexes = {}
		for proj in [project, MASTER_PROJECT]:
			index_path, backup_dir = get_index_path(args.index_dir, proj)
			project_indexes[proj] = load_index(index_path, backup_dir, args.debug)
		
		# Index the file
		index_file(args.file, project, args.document_dir, 
				   project_indexes, args.max_chunk_size, embedding_config, args.debug)
		
		# Save indexes
		for proj, docs in project_indexes.items():
			if docs:  # Only save if there are documents
				index_path, backup_dir = get_index_path(args.index_dir, proj)
				save_index(docs, index_path, backup_dir, args.debug)
		
	else:
		# Index a directory
		if not os.path.exists(args.document_dir):
			print(f"Error: Document directory not found: {args.document_dir}")
			return
		
		if args.project:
			project_dir = os.path.join(args.document_dir, args.project)
			if not os.path.exists(project_dir) or not os.path.isdir(project_dir):
				print(f"Error: Project directory not found: {project_dir}")
				return
			print(f"Indexing project: {args.project}")
			index_directory(args.document_dir, args.index_dir, args.max_chunk_size, 
						   embedding_config, args.project, args.debug)
		else:
			print(f"Indexing all documents")
			index_directory(args.document_dir, args.index_dir, args.max_chunk_size,
						   embedding_config, None, args.debug)
	
	print("\nIndexing complete")


if __name__ == "__main__":
	main()