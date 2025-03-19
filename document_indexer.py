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
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import re


# Import our custom embedding library
from embeddings import EmbeddingConfig, get_embedding_provider, load_project_config

# Filter resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker")

# Constants
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
MAX_CHUNK_SIZE = 3500  # Characters - this should be a default and should change depending on embedding model
MIN_CHUNK_SIZE = 50  # Minimum characters for a chunk to be indexed
MAX_CHUNKS = 100 # temporary fix for files that don't chunk properly
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


def split_into_paragraphs(text: str) -> List[str]:
	"""Split text into paragraphs based on double newlines."""
	# Normalize line endings
	text = text.replace('\r\n', '\n')
	
	# Ensure paragraphs are separated by exactly one blank line
	text = text.replace('\n\n\n', '\n\n')
	while '\n\n\n' in text:
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
	
	# Remove any inline R script
	pattern = r'\n\\\`\\\`\\\`\{r[^}]*\}[\s\S]*?\\\`\\\`\\\`\n'
	text = re.sub(pattern, '', text)

	
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
		
		# Check if we've reached the maximum number of chunks
		if len(chunks) >= MAX_CHUNKS:
			if debug:
				print(f"WARNING: Reached maximum number of chunks ({MAX_CHUNKS}). Stopping chunking process.")
			break

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
		
		if debug:
			print(f"Saved {len(documents)} documents to index: {index_path}")
	except Exception as e:
		print(f"Error saving index: {e}")


def get_accurate_chunk_count(files: List[str], max_chunk_size: int, debug: bool = False) -> int:
	"""
	Pre-process all files to get an accurate count of chunks that will be created.
	"""
	print("Calculating exact number of chunks (pre-processing documents)...")
	total_chunks = 0
	skipped_files = 0
	
	# Use a progress bar for the pre-processing phase
	with tqdm(total=len(files), desc="Pre-processing", unit="file") as pbar:
		for file_path in files:
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					content = f.read()
				
				# Use the actual chunking logic to get an accurate count
				chunks = create_paragraph_chunks(content, max_chunk_size, False)
				
				# Filter out chunks that are too small
				chunks = [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_SIZE]
				
				# Apply cap of MAX_CHUNKS per file
				actual_chunks = min(len(chunks), MAX_CHUNKS)
				total_chunks += actual_chunks
				
				file_name = os.path.basename(file_path)
				if debug and actual_chunks > 0:
					print(f"[DEBUG] {file_name}: {actual_chunks} chunks")
				
			except Exception as e:
				if debug:
					print(f"[DEBUG] Error pre-processing {file_path}: {e}")
				skipped_files += 1
			
			pbar.update(1)
	
	if skipped_files > 0:
		print(f"Warning: {skipped_files} files could not be pre-processed")
	
	return total_chunks


def index_file(file_path: str, project_dir: str, document_dir: str, 
				   project_indexes: Dict[str, List[Document]], 
				   max_chunk_size: int, embedding_config: Optional[EmbeddingConfig] = None,
				   debug: bool = False) -> None:
		"""
		Index a single file by creating paragraph-based chunks with overlap and generating embeddings.
		Updates both the project-specific index and the master index.
		"""
		# Get the project for this file
		project = get_project_path(file_path, document_dir)
		
		# Create the embedding provider
		embedding_provider = get_embedding_provider(
			project_dir=project, 
			document_dir=document_dir, 
			config=embedding_config,
			debug=debug
		)
		
		# Call the new function that takes a provider
		index_file_with_provider(
			file_path, 
			project, 
			document_dir, 
			project_indexes, 
			max_chunk_size, 
			embedding_provider,
			embedding_config,
			debug
		)


def estimate_total_chunks(files: List[str], max_chunk_size: int, debug: bool = False) -> int:
	"""
	Estimate the total number of chunks across all files to be indexed.
	This is used for progress tracking.
	"""
	total_chunks = 0
	for file_path in files:
		try:
			with open(file_path, 'r', encoding='utf-8') as f:
				content = f.read()
			
			# Quickly estimate number of chunks
			estimated_chunks = max(1, len(content) // (max_chunk_size // 2))
			# Apply cap of MAX_CHUNKS per file
			total_chunks += min(estimated_chunks, MAX_CHUNKS)
			
			if debug:
				file_name = os.path.basename(file_path)
				print(f"[DEBUG] Estimated {estimated_chunks} chunks for {file_name}")
				
		except Exception as e:
			if debug:
				print(f"[DEBUG] Error estimating chunks for {file_path}: {e}")
			# Default conservative estimate
			total_chunks += 1
			
	return total_chunks


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


def get_project_embedding_config(project: str, document_dir: str, debug: bool = False) -> Optional[EmbeddingConfig]:
	"""Get project-specific embedding configuration if available."""
	try:
		return load_project_config(project, document_dir)
	except Exception as e:
		if debug:
			print(f"[DEBUG] Error loading project config for {project}: {e}")
		return None


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
	
	# Dictionary to hold embedding providers for each project (to reuse them)
	embedding_providers = {}
	
	# Load existing indexes for each project
	for proj in projects:
		index_path, backup_dir = get_index_path(index_dir, proj)
		project_indexes[proj] = load_index(index_path, backup_dir, debug)
		
		# Initialize embedding provider for this project
		if proj not in embedding_providers:
			if debug:
				print(f"[DEBUG] Initializing embedding provider for project: {proj}")
			
			proj_embedding_config = embedding_config
			if proj_embedding_config is None:
				proj_embedding_config = get_project_embedding_config(proj, document_dir, debug)
			
			embedding_providers[proj] = get_embedding_provider(
				project_dir=proj, 
				document_dir=document_dir, 
				config=proj_embedding_config,
				debug=debug
			)
			
			if debug:
				print(f"[DEBUG] Embedding provider initialized for {proj}: "
					  f"{embedding_providers[proj].config.embedding_type}/{embedding_providers[proj].config.model_name}")
	
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
	
	# Get accurate chunk count instead of estimation
	total_chunks = get_accurate_chunk_count(files, max_chunk_size, debug)
	print(f"Total chunks to index: {total_chunks}")
	
	# Track files where indexing was aborted due to MAX_CHUNKS limit
	aborted_files = set()
	
	# Create progress bar for all chunks with improved settings
	chunk_pbar = tqdm(total=total_chunks, desc="Total Progress", unit="chunk", 
				 mininterval=0.1, maxinterval=1.0, position=1, leave=True)
	
	# Create a file info line above the progress bar
	file_info = tqdm(total=0, bar_format='{desc}', position=0, leave=True)
	
	# Process each file
	for i, file_path in enumerate(files, 1):
		rel_path = os.path.relpath(file_path, document_dir)
		# Update file info line (above the chunk progress bar)
		file_info.set_description_str(f"Processing: {rel_path} ({i}/{len(files)})")
		
		try:
			# Get file's project
			file_project = get_project_path(file_path, document_dir)
			
			# Process the file using the project's embedding provider
			chunks_reached_limit = index_file_with_provider(
				file_path, 
				file_project, 
				document_dir, 
				project_indexes, 
				max_chunk_size, 
				embedding_providers[file_project],  # Pass the provider directly
				embedding_config,
				debug,
				chunk_pbar  # Pass the progress bar
			)
			
			# Track files that hit the MAX_CHUNKS limit
			if chunks_reached_limit:
				aborted_files.add(rel_path)
			
			# Save project indexes after each file
			for proj, docs in project_indexes.items():
				if docs:  # Only save if there are documents
					index_path, backup_dir = get_index_path(index_dir, proj)
					save_index(docs, index_path, backup_dir, debug)
			
			# Force garbage collection
			gc.collect()
			
		except Exception as e:
			print(f"\nFailed to process {rel_path}: {e}")
			if debug:
				print(traceback.format_exc())
	
	# Close the progress bars
	file_info.close()
	chunk_pbar.close()
	
	# Report files where indexing was aborted
	if aborted_files:
		print("\nThe following files reached the maximum chunk limit and were partially indexed:")
		for file_path in sorted(aborted_files):
			print(f"  - {file_path}")
	
	# Print summary
	print("\nIndexing Summary:")
	for proj, docs in project_indexes.items():
		print(f"Project '{proj}': {len(docs)} document chunks")


def index_file_with_provider(file_path: str, project: str, document_dir: str, 
									project_indexes: Dict[str, List[Document]], 
									max_chunk_size: int, 
									embedding_provider, # Direct provider instance
									embedding_config: Optional[EmbeddingConfig] = None,
									debug: bool = False,
									progress_bar=None) -> bool:
	"""
	Index a single file using a pre-initialized embedding provider.
	Updates both the project-specific index and the master index.
	Returns True if the MAX_CHUNKS limit was reached during processing.
	"""
	chunks_reached_limit = False
	
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			content = f.read()
		
		rel_path = os.path.relpath(file_path, document_dir)
		file_name = os.path.basename(file_path)
		file_size = os.path.getsize(file_path)
		
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
			if debug:
				print(f"File already indexed in project '{project}': {rel_path}")
			# Update the progress bar for the chunks we're skipping
			if progress_bar:
				progress_bar.update(len(existing_docs))
			return False
		
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
				# Update the progress bar for the chunks we're skipping
				if progress_bar:
					progress_bar.update(len(existing_master_docs))
				return False
		
		# Remove any old versions of this file from both indexes
		project_indexes[project] = [doc for doc in project_indexes[project] 
								  if doc.metadata.get('file_path') != rel_path]
		
		if project != MASTER_PROJECT:
			project_indexes[MASTER_PROJECT] = [doc for doc in project_indexes[MASTER_PROJECT] 
											if doc.metadata.get('file_path') != rel_path]
		
		if debug:
			print(f"Indexing: {rel_path} (project: {project})")
		# else:
			print(f"[DEBUG] Using embedding model: {embedding_provider.config.model_name} "
				  f"(type: {embedding_provider.config.embedding_type})")
				  
		# Remove any inline R script
		pattern = r'\n\\\`\\\`\\\`\{r[^}]*\}[\s\S]*?\\\`\\\`\\\`\n'
		content = re.sub(pattern, '', content)
		
		# some debug code
		if debug:
			if rel_path == 'textbook/02-10-evolution-of-marketing/01-10-40-notes-marketing-management-process.md':
				print(content)
		
		# Use paragraph-based chunking with overlap
		chunks = create_paragraph_chunks(content, max_chunk_size, debug)
		
		# Check if we reached the MAX_CHUNKS limit
		if len(chunks) >= MAX_CHUNKS:
			chunks_reached_limit = True
		
		# Filter out chunks that are too small
		original_chunk_count = len(chunks)
		chunks = [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_SIZE]
		
		if debug:
			print(f"  Split into {len(chunks)} chunks (removed {original_chunk_count - len(chunks)} chunks below MIN_CHUNK_SIZE)")
		
		# Store new document chunks
		new_docs = []
		
		for i, chunk in enumerate(chunks):
			if debug:
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
			
			# Generate embedding for the chunk using the provided embedder
			embedding = embedding_provider.create_embedding(chunk)
			
			if embedding:
				doc = Document(content=chunk, metadata=metadata, embedding=embedding)
				new_docs.append(doc)
				if debug:
					print(f"  Successfully indexed chunk {i+1}/{len(chunks)}")
				# Update the progress bar
				if progress_bar:
					progress_bar.update(1)
			else:
				if debug or progress_bar is None:
					print(f"  Failed to generate embedding for chunk {i+1}")
				# Still update the progress bar even if embedding failed
				if progress_bar:
					progress_bar.update(1)
		
		# Add the new documents to both the project index and master index (if different)
		project_indexes[project].extend(new_docs)
		if project != MASTER_PROJECT:
			project_indexes[MASTER_PROJECT].extend(new_docs)
			
		return chunks_reached_limit
		
	except Exception as e:
		print(f"Error indexing file {file_path}: {e}")
		if debug:
			print(traceback.format_exc())
		return False

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
	parser.add_argument("--min-chunk-size", type=int, default=MIN_CHUNK_SIZE,
						help="Minimum size of document chunks in characters to be indexed")
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
	print(f"Min chunk size: {args.min_chunk_size} characters")
	print(f"Index directory: {args.index_dir}")
	
	# Track files where indexing was aborted due to MAX_CHUNKS limit
	aborted_files = set()
	
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
		
		# Get accurate chunk count for this file
		rel_path = os.path.relpath(args.file, args.document_dir)
		print(f"Preparing to index: {rel_path}")
		
		with open(args.file, 'r', encoding='utf-8') as f:
			content = f.read()
		chunks = create_paragraph_chunks(content, args.max_chunk_size, args.debug)
		chunks = [chunk for chunk in chunks if len(chunk) >= args.min_chunk_size]
		
		# Create file info line and progress bar on separate lines
		file_info = tqdm(total=0, bar_format='{desc}', position=0, leave=True)
		file_info.set_description_str(f"Processing: {rel_path}")
		
		# Create progress bar for this file with improved settings
		with tqdm(total=len(chunks), desc="Indexing", unit="chunk", 
			mininterval=0.1, maxinterval=1.0, position=1, leave=True) as pbar:
			# Index the file
			result = index_file_with_provider(
				args.file, 
				project, 
				args.document_dir, 
				project_indexes, 
				args.max_chunk_size, 
				get_embedding_provider(project, args.document_dir, embedding_config, args.debug),
				embedding_config, 
				args.debug,
				pbar
			)
			
			if result:
				aborted_files.add(rel_path)
	
		# Close the file info line
		file_info.close()
		
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
	
	# Report files where indexing was aborted
	if aborted_files:
		print("\nThe following files reached the maximum chunk limit and were partially indexed:")
		for file_path in sorted(aborted_files):
			print(f"  - {file_path}")
	
	print("\nIndexing complete")


if __name__ == "__main__":
	main()