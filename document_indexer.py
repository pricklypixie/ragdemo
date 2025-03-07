#!/usr/bin/env python3
"""
Document Indexer for RAG Applications with Project Support

This tool:
1. Indexes documents from a local directory
2. Creates embeddings using sentence-transformers
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

# Filter resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker")

# Force CPU usage instead of Metal on MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_FALLBACK_POLICY"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA

# Set threading options
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
MAX_CHUNK_SIZE = 1500  # Characters
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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


def split_into_paragraphs(text: str) -> List[str]:
	"""Split text into paragraphs based on double newlines."""
	# Handle different line ending styles
	text = text.replace('\r\n', '\n')
	
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
				print(f"[DEBUG] Created chunk {len(chunks)} with {current_size} chars and {len(current_chunk)} paragraphs")
			
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


def index_file(file_path: str, model, document_dir: str, project_indexes: Dict[str, List[Document]], 
			   max_chunk_size: int, debug: bool = False) -> None:
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
				'chunk_index': i,
				'total_chunks': len(chunks),
				'chunk_size': len(chunk),
				'paragraphs': len(paragraphs),
				'last_modified': modified_time.isoformat()
			}
			
			# Generate embedding for the chunk
			embedding = create_embedding(model, chunk, debug)
			
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


def index_directory(document_dir: str, index_dir: str, model, max_chunk_size: int, 
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
	
	import torch
	torch.set_grad_enabled(False)  # Ensure no gradients are computed
	
	for i, file_path in enumerate(files, 1):
		print(f"\nProcessing file {i}/{len(files)}: {file_path}")
		try:
			# Process the file
			index_file(file_path, model, document_dir, project_indexes, max_chunk_size, debug)
			
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
	parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
						help="Sentence Transformer model to use for embeddings")
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
	
	print(f"Document Indexer with Project Support")
	print(f"Embedding model: {args.embedding_model}")
	print(f"Max chunk size: {args.max_chunk_size} characters")
	print(f"Index directory: {args.index_dir}")
	
	# Load embedding model
	print("Loading embedding model...")
	model = get_embedding_model(args.embedding_model, args.debug)
	
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
		index_file(args.file, model, args.document_dir, project_indexes, args.max_chunk_size, args.debug)
		
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
			index_directory(args.document_dir, args.index_dir, model, args.max_chunk_size, args.project, args.debug)
		else:
			print(f"Indexing all documents")
			index_directory(args.document_dir, args.index_dir, model, args.max_chunk_size, None, args.debug)
	
	print("\nIndexing complete")


if __name__ == "__main__":
	main()