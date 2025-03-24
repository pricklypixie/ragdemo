#!/usr/bin/env python3
"""
Document Indexer for RAG Applications with Project Support

This tool:
1. Indexes documents from a local directory
2. Creates embeddings using the configurable embedding library
3. Supports project-based indexing (subdirectories as separate projects)
4. Saves separate indexes for each project and a master index
5. Uses embedding-aware chunking for optimal results
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
DEFAULT_MAX_CHUNK_SIZE = 1500  # Characters - this is now a default value
DEFAULT_MIN_CHUNK_SIZE = 50    # Minimum characters for a chunk to be indexed
DEFAULT_CHARS_PER_DIMENSION = 4  # Default characters per embedding dimension for auto-sizing
MAX_CHUNKS = 100  # temporary fix for files that don't chunk properly
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
	
	# clean text before further processing
	text = clean_text(text, debug)
	
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




def calculate_optimal_chunk_size(embedding_provider, chars_per_dimension: int, 
							 default_size: int = DEFAULT_MAX_CHUNK_SIZE, 
							 debug: bool = False) -> int:
	"""
	Calculate an optimal chunk size based on the embedding dimension.
	Using chars_per_dimension as a multiplier for the embedding dimension.
	
	Args:
		embedding_provider: The embedding provider to get dimensions from
		chars_per_dimension: Characters per embedding dimension
		default_size: Default size to use if calculation fails
		debug: Whether to print debug information
		
	Returns:
		Optimal chunk size in characters
	"""
	try:
		dimension = embedding_provider.get_embedding_dimension()
		if dimension <= 0:
			if debug:
				print(f"[DEBUG] Invalid embedding dimension: {dimension}, using default size")
			return default_size
		
		# Calculate chunk size as a multiple of the embedding dimension
		chunk_size = dimension * chars_per_dimension
		
		# Ensure it's within reasonable bounds
		chunk_size = max(DEFAULT_MIN_CHUNK_SIZE * 2, min(chunk_size, 8000))
		
		if debug:
			print(f"[DEBUG] Calculated optimal chunk size: {chunk_size} chars "
				f"(dimension: {dimension} × {chars_per_dimension} chars/dim)")
		
		return chunk_size
	except Exception as e:
		if debug:
			print(f"[DEBUG] Error calculating optimal chunk size: {e}")
		return default_size


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
				chunks = [chunk for chunk in chunks if len(chunk) >= DEFAULT_MIN_CHUNK_SIZE]
				
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


# def discover_projects(document_dir: str) -> List[str]:
# 	"""Discover all projects (subdirectories) in the document directory."""
# 	projects = [MASTER_PROJECT]  # Master project is always included
# 	
# 	try:
# 		# Get all subdirectories in the document directory
# 		for item in os.listdir(document_dir):
# 			item_path = os.path.join(document_dir, item)
# 			if os.path.isdir(item_path):
# 				projects.append(item)
# 		
# 		return projects
# 	except Exception as e:
# 		print(f"Error discovering projects: {e}")
# 		return [MASTER_PROJECT]  # Return at least the master project


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


# Add these functions to document_indexer.py, replacing the existing function for project config

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

def load_project_config(project: str, document_dir: str) -> Optional[EmbeddingConfig]:
	"""
	Load project-specific embedding configuration.
	
	Args:
		project: Project name
		document_dir: Base documents directory
		
	Returns:
		EmbeddingConfig for the project or None if not found
	"""
	# Try new project_config.json first
	config_path = get_project_config_path(project, document_dir, use_legacy=False)
	if os.path.exists(config_path):
		try:
			with open(config_path, 'r') as f:
				project_config = json.load(f)
			
			# Get the indexing section
			indexing_config = project_config.get("indexing", {})
			return EmbeddingConfig.from_dict(indexing_config)
		except Exception as e:
			print(f"Error loading project config from {config_path}: {e}")
	
	# Fall back to legacy embedding_config.json
	legacy_path = get_project_config_path(project, document_dir, use_legacy=True)
	if os.path.exists(legacy_path):
		try:
			return EmbeddingConfig.from_json_file(legacy_path)
		except Exception as e:
			print(f"Error loading legacy config from {legacy_path}: {e}")
	
	return None

def save_project_config(project: str, document_dir: str, embedding_config: EmbeddingConfig = None) -> None:
	"""
	Save project configuration to file.
	
	Args:
		project: The project name
		document_dir: Base documents directory
		embedding_config: Optional embedding configuration to include
	"""
	config_path = get_project_config_path(project, document_dir, use_legacy=False)
	
	# Create default project config
	project_config = {
		"indexing": {
			"embedding_type": "sentence_transformers",
			"model_name": "all-mpnet-base-v2",
			"api_key": None,
			"additional_params": {}
		},
		"rag": {
			"llm_type": "local",
			"llm_model": "mistral-7b-instruct-v0",
			"rag_mode": "chunk",
			"rag_count": 3
		}
	}
	
	# Update with the provided embedding config if available
	if embedding_config:
		project_config["indexing"] = embedding_config.to_indexing_dict()
	
	# Create directory if needed
	os.makedirs(os.path.dirname(config_path), exist_ok=True)
	
	# Write the config file
	with open(config_path, 'w') as f:
		json.dump(project_config, f, indent=2)
	
	print(f"Saved project configuration to {config_path}")





def index_directory(document_dir: str, index_dir: str, max_chunk_size: int,
			   embedding_config: Optional[EmbeddingConfig] = None,
			   project: Optional[str] = None, debug: bool = False,
			   auto_adjust_chunks: bool = False,
			   chars_per_dimension: int = DEFAULT_CHARS_PER_DIMENSION) -> None:
	"""
	Index all supported documents in the specified directory.
	Can be limited to a specific project (subdirectory).
	
	Args:
		document_dir: Directory containing documents to index
		index_dir: Directory to store the index
		max_chunk_size: Maximum size of document chunks
		embedding_config: Embedding configuration
		project: Specific project to index (None for all)
		debug: Whether to enable debug output
		auto_adjust_chunks: Whether to adjust chunk size based on embedding dim
		chars_per_dimension: Characters per embedding dimension for auto-sizing
	"""
	# Determine which projects to index
	if project is None:
		# When no project is specified, index all projects
		projects = discover_projects(document_dir)
		if debug:
			print(f"[DEBUG] Discovered all projects to index: {projects}")
	else:
		# When a project is specified, only index that project
		projects = [project]
		if debug:
			print(f"[DEBUG] Indexing single project: {project}")
	
	if debug:
		print(f"[DEBUG] Projects to index: {projects}")
	
	# Dictionary to hold indexes for each project
	project_indexes = {}
	
	# Dictionary to hold embedding providers for each project (to reuse them)
	embedding_providers = {}
	
	# Dictionary to store optimal chunk sizes for each project
	project_chunk_sizes = {}
	
	# Load existing indexes for projects we'll update
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
			
			# Calculate optimal chunk size if auto-adjust is enabled
			if auto_adjust_chunks:
				project_chunk_sizes[proj] = calculate_optimal_chunk_size(
					embedding_providers[proj],
					chars_per_dimension,
					max_chunk_size,
					debug
				)
			else:
				project_chunk_sizes[proj] = max_chunk_size
	
	# Find all files to process based on the project(s)
	files = []
	if project == MASTER_PROJECT or project is None:
		# For master project or when indexing all projects, find all files
		files = glob.glob(os.path.join(document_dir, "**/*.txt"), recursive=True)
		files += glob.glob(os.path.join(document_dir, "**/*.md"), recursive=True)
	else:
		# For a specific project, only find files in that project's directory
		project_dir = os.path.join(document_dir, project)
		files = glob.glob(os.path.join(project_dir, "**/*.txt"), recursive=True)
		files += glob.glob(os.path.join(project_dir, "**/*.md"), recursive=True)
	
	if not files:
		print(f"No supported files found to index")
		return
	
	print(f"Found {len(files)} files to index")
	
	# Sort files by size (smallest first) to get some quick wins
	files = sorted(files, key=os.path.getsize)
	
	# Inform about chunk sizes if auto-adjusting
	if auto_adjust_chunks:
		print("\nUsing embedding-aware chunk sizes:")
		for proj, size in project_chunk_sizes.items():
			model_name = embedding_providers[proj].config.model_name
			print(f"  Project '{proj}' using model '{model_name}': {size} chars")
	
	# Get accurate chunk count using project-specific chunk sizes
	if auto_adjust_chunks:
		# We need to determine the chunk size for each file based on its project
		total_chunks = 0
		print("Calculating exact number of chunks (pre-processing documents)...")
		skipped_files = 0
		
		with tqdm(total=len(files), desc="Pre-processing", unit="file") as pbar:
			for file_path in files:
				try:
					# Get file's project
					file_project = get_project_path(file_path, document_dir)
					# Use the project's chunk size
					proj_chunk_size = project_chunk_sizes.get(file_project, max_chunk_size)
					
					with open(file_path, 'r', encoding='utf-8') as f:
						content = f.read()
					
					chunks = create_paragraph_chunks(content, proj_chunk_size, False)
					chunks = [chunk for chunk in chunks if len(chunk) >= DEFAULT_MIN_CHUNK_SIZE]
					actual_chunks = min(len(chunks), MAX_CHUNKS)
					total_chunks += actual_chunks
					
				except Exception as e:
					if debug:
						print(f"[DEBUG] Error pre-processing {file_path}: {e}")
					skipped_files += 1
				
				pbar.update(1)
		
		if skipped_files > 0:
			print(f"Warning: {skipped_files} files could not be pre-processed")
	else:
		# If not auto-adjusting, use the same chunk size for all files
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
			
			# Skip this file if its project is not in our list of projects to index
			if file_project not in projects and file_project != MASTER_PROJECT:
				if debug:
					print(f"[DEBUG] Skipping file from project '{file_project}' that is not being indexed")
				continue
				
			# IMPORTANT CHANGE: 
			# Determine which project indexes to update for this file
			projects_to_update = []
			
			if project == MASTER_PROJECT or project is None:
				# If we're indexing the master project or all projects, update both master and file's project
				if file_project == MASTER_PROJECT:
					# For files in the root, update only master
					projects_to_update = [MASTER_PROJECT]
				else:
					# For files in project subdirectories, update both the specific project and master
					projects_to_update = [file_project, MASTER_PROJECT]
			else:
				# If we're indexing a specific project, only update that project's index
				if file_project == project:
					projects_to_update = [project]
				else:
					# Skip files that don't belong to the current project
					if debug:
						print(f"[DEBUG] Skipping file '{rel_path}' as it belongs to project '{file_project}' not '{project}'")
					continue
			
			if debug:
				print(f"[DEBUG] Processing file: {rel_path} (project: {file_project})")
				print(f"[DEBUG] Will update indexes for: {projects_to_update}")
			
			# Get the project-specific chunk size
			proj_chunk_size = project_chunk_sizes.get(file_project, max_chunk_size)
			
			# Create a list to store the new document chunks
			new_docs = []
			
			# Process the file using the project's embedding provider and chunk size
			with open(file_path, 'r', encoding='utf-8') as f:
				content = f.read()
			
			# Get file stats
			stats = os.stat(file_path)
			modified_time = datetime.fromtimestamp(stats.st_mtime)
			
			# Clean text before further processing
			content = clean_text(content, debug)
			
			# Use paragraph-based chunking with overlap
			chunks = create_paragraph_chunks(content, proj_chunk_size, debug)
			
			# Check if we reached the MAX_CHUNKS limit
			if len(chunks) >= MAX_CHUNKS:
				aborted_files.add(rel_path)
			
			# Filter out chunks that are too small
			original_chunk_count = len(chunks)
			chunks = [chunk for chunk in chunks if len(chunk) >= DEFAULT_MIN_CHUNK_SIZE]
			
			if debug:
				print(f"  Split into {len(chunks)} chunks (removed {original_chunk_count - len(chunks)} chunks below MIN_CHUNK_SIZE)")
			
			# Get embedding provider for this file's project
			provider = embedding_providers[file_project]
			
			# Process each chunk
			for j, chunk in enumerate(chunks):
				if debug:
					print(f"  Processing chunk {j+1}/{len(chunks)} ({len(chunk)} chars)")
				
				# Get paragraph info for this chunk
				paragraphs = split_into_paragraphs(chunk)
				
				metadata = {
					'file_path': rel_path,
					'file_name': os.path.basename(file_path),
					'project': file_project,
					'embedding_model': provider.config.model_name,
					'embedding_type': provider.config.embedding_type,
					'chunk_index': j,
					'total_chunks': len(chunks),
					'chunk_size': len(chunk),
					'paragraphs': len(paragraphs),
					'last_modified': modified_time.isoformat()
				}
				
				# Generate embedding for the chunk
				embedding = provider.create_embedding(chunk)
				
				if embedding:
					doc = Document(content=chunk, metadata=metadata, embedding=embedding)
					new_docs.append(doc)
					if debug:
						print(f"  Successfully indexed chunk {j+1}/{len(chunks)}")
					# Update the progress bar
					chunk_pbar.update(1)
				else:
					if debug or chunk_pbar is None:
						print(f"  Failed to generate embedding for chunk {j+1}")
					# Still update the progress bar even if embedding failed
					chunk_pbar.update(1)
			
			# Update the project indexes
			for proj in projects_to_update:
				# Remove any old versions of this file from the index
				project_indexes[proj] = [doc for doc in project_indexes[proj] 
									if doc.metadata.get('file_path') != rel_path]
				
				# Add the new documents to the index
				project_indexes[proj].extend(new_docs)
				
				# Save the updated index after each file
				index_path, backup_dir = get_index_path(index_dir, proj)
				save_index(project_indexes[proj], index_path, backup_dir, debug)
			
			# Force garbage collection
			gc.collect()
			
		except Exception as e:
			print(f"\nFailed to process {rel_path}: {e}")
			if debug:
				print(traceback.format_exc())
			# Still update the progress bar for skipped files
			# Estimate number of chunks we would have had
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					content = f.read()
				estimated_chunks = max(1, min(len(content) // (max_chunk_size // 2), MAX_CHUNKS))
				chunk_pbar.update(estimated_chunks)
			except:
				# If we can't even read the file, just update by 1
				chunk_pbar.update(1)
	
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





def clean_text(text: str, debug: bool = False) -> str:
		"""
		Clean and normalize text before chunking and embedding.
		
		This function:
		1. Normalizes line endings
		2. Removes code blocks (R, Python, etc.)
		3. Removes HTML tags
		4. Standardizes whitespace
		5. Handles special characters and symbols
		
		Args:
			text: The input text to clean
			debug: Whether to print debug information
			
		Returns:
			Cleaned text ready for chunking
		"""
		if debug:
			original_length = len(text)
			print(f"[DEBUG] Cleaning text of length {original_length}")
		
		# Step 1: Normalize line endings
		text = text.replace('\r\n', '\n').replace('\r', '\n')
		

		
		# Step 2: Remove code blocks - handle various languages
		code_block_pattern = r'```[a-zA-Z0-9_\-+]*\s*[\s\S]*?```'
		text = re.sub(code_block_pattern, '\n', text, flags=re.DOTALL)
		
		# Additional pattern for R markdown code blocks
		r_block_pattern = r'\n\\\`\\\`\\\`\{r[^}]*\}[\s\S]*?\\\`\\\`\\\`\n'
		text = re.sub(r_block_pattern, '\n', text, flags=re.DOTALL)
		
		# Remove other markdown examples from specific projects
		
		include_pattern = r'\n{%.*?%}\n'
		text = re.sub(include_pattern, '\n', text, flags=re.DOTALL)
		
		include_pattern = r'\n{:.*?}\n'
		text = re.sub(include_pattern, '\n', text, flags=re.DOTALL)


		
		# Step 3: Remove HTML tags
		html_pattern = r'<[^>]+>'
		text = re.sub(html_pattern, '', text)
		
		# Step 4: Standardize whitespace
		# Replace multiple spaces with a single space
		text = re.sub(r' +', ' ', text)
		# Replace multiple newlines with at most two (to preserve paragraph breaks)
		text = re.sub(r'\n{3,}', '\n\n', text)
		
		# Step 5: Special handling for common markdown elements
		# Remove horizontal rules
		text = re.sub(r'---+', '', text)
		# Simplify bullet points
		text = re.sub(r'^\s*[*\-+]\s+', '• ', text, flags=re.MULTILINE)
		
		# Remove URLs (optional - could replace with [URL] if preferred)
		url_pattern = r'https?://\S+'
		text = re.sub(url_pattern, '[URL]', text)
		
		text = text.replace('\n', '\n\n')		
		while '\n\n\n' in text:
			text = text.replace('\n\n\n', '\n\n')

		
		# Strip extra whitespace at beginning and end
		text = text.strip()
		
		if debug:
			cleaned_length = len(text)
			reduction = original_length - cleaned_length
			percent = (reduction / original_length) * 100 if original_length > 0 else 0
			print(f"[DEBUG] Removed {reduction} characters ({percent:.1f}%) during cleaning")
		
		return text



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
				  
		# Clean text before further processing
				
		content = clean_text(content, debug)
				
		# Use paragraph-based chunking with overlap
		chunks = create_paragraph_chunks(content, max_chunk_size, debug)
		
		# Check if we reached the MAX_CHUNKS limit
		if len(chunks) >= MAX_CHUNKS:
			chunks_reached_limit = True
		
		# Filter out chunks that are too small
		original_chunk_count = len(chunks)
		chunks = [chunk for chunk in chunks if len(chunk) >= DEFAULT_MIN_CHUNK_SIZE]
		
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
	parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_MAX_CHUNK_SIZE,
					help="Maximum size of document chunks in characters")
	parser.add_argument("--min-chunk-size", type=int, default=DEFAULT_MIN_CHUNK_SIZE,
					help="Minimum size of document chunks in characters to be indexed")
	parser.add_argument("--debug", action="store_true",
					help="Enable debug logging")
	parser.add_argument("--file", type=str,
					help="Index a single file instead of a directory")
	parser.add_argument("--project", type=str,
					help="Index a specific project (subdirectory) only")
	parser.add_argument("--list-projects", action="store_true", 
					help="List all available projects")
	# Add new arguments for embedding-aware chunking
	parser.add_argument("--auto-adjust-chunks", action="store_true",
					help="Automatically adjust chunk size based on embedding model dimension")
	parser.add_argument("--chars-per-dimension", type=int, default=DEFAULT_CHARS_PER_DIMENSION,
					help="Characters per embedding dimension when auto-adjusting chunks")
	
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
	if args.auto_adjust_chunks:
		print(f"Auto-adjusting chunk sizes based on embedding dimensions")
		print(f"Characters per dimension: {args.chars_per_dimension}")
	
	# Track files where indexing was aborted due to MAX_CHUNKS limit
	aborted_files = set()
	
	if args.file:
		# Index a single file
		if not os.path.exists(args.file):
			print(f"Error: File not found: {args.file}")
			return
		
		print(f"Indexing single file: {args.file}")
		
		# Get the project for this file
		file_project = get_project_path(args.file, args.document_dir)
		
		# If a project was specified and the file doesn't belong to that project, skip it
		if args.project and args.project != MASTER_PROJECT and file_project != args.project:
			print(f"Error: File {args.file} belongs to project '{file_project}', not '{args.project}'")
			return
		
		# Determine which projects to update for this file
		projects_to_update = []
		if args.project:
			# If a specific project was specified, only update that project
			projects_to_update = [args.project]
		else:
			# If no project was specified, update the file's project and master
			if file_project == MASTER_PROJECT:
				projects_to_update = [MASTER_PROJECT]
			else:
				projects_to_update = [file_project, MASTER_PROJECT]
		
		# Load indexes for all projects we need to update
		project_indexes = {}
		for proj in projects_to_update:
			index_path, backup_dir = get_index_path(args.index_dir, proj)
			project_indexes[proj] = load_index(index_path, backup_dir, args.debug)
		
		# Create the embedding provider for the file's project
		embedding_provider = get_embedding_provider(
			project_dir=file_project, 
			document_dir=args.document_dir, 
			config=embedding_config,
			debug=args.debug
		)
		
		# Calculate optimal chunk size if auto-adjust is enabled
		if args.auto_adjust_chunks:
			max_chunk_size = calculate_optimal_chunk_size(
				embedding_provider,
				args.chars_per_dimension,
				args.max_chunk_size,
				args.debug
			)
			print(f"Using embedding-aware chunk size: {max_chunk_size} characters")
		else:
			max_chunk_size = args.max_chunk_size
		
		# Get accurate chunk count for this file
		rel_path = os.path.relpath(args.file, args.document_dir)
		print(f"Preparing to index: {rel_path}")
		
		with open(args.file, 'r', encoding='utf-8') as f:
			content = f.read()
		chunks = create_paragraph_chunks(content, max_chunk_size, args.debug)
		chunks = [chunk for chunk in chunks if len(chunk) >= args.min_chunk_size]
		
		# Create file info line and progress bar on separate lines
		file_info = tqdm(total=0, bar_format='{desc}', position=0, leave=True)
		file_info.set_description_str(f"Processing: {rel_path}")
		
		# Create progress bar for this file with improved settings
		with tqdm(total=len(chunks), desc="Indexing", unit="chunk", 
			mininterval=0.1, maxinterval=1.0, position=1, leave=True) as pbar:
			
			# Process the file
			new_docs = []
			stats = os.stat(args.file)
			modified_time = datetime.fromtimestamp(stats.st_mtime)
			
			# Use paragraph-based chunking with overlap
			content = clean_text(content, args.debug)
			chunks = create_paragraph_chunks(content, max_chunk_size, args.debug)
			
			# Check if we reached the MAX_CHUNKS limit
			if len(chunks) >= MAX_CHUNKS:
				aborted_files.add(rel_path)
			
			# Filter out chunks that are too small
			chunks = [chunk for chunk in chunks if len(chunk) >= args.min_chunk_size]
			
			# Process each chunk
			for i, chunk in enumerate(chunks):
				# Get paragraph info for this chunk
				paragraphs = split_into_paragraphs(chunk)
				
				metadata = {
					'file_path': rel_path,
					'file_name': os.path.basename(args.file),
					'project': file_project,
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
					pbar.update(1)
				else:
					pbar.update(1)
			
			# Update all required project indexes
			for proj in projects_to_update:
				# Remove any old versions of this file
				project_indexes[proj] = [doc for doc in project_indexes[proj] 
									if doc.metadata.get('file_path') != rel_path]
				# Add the new documents
				project_indexes[proj].extend(new_docs)
	
		# Close the file info line
		file_info.close()
		
		# Save indexes for all projects
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
			if args.project != MASTER_PROJECT:
				project_dir = os.path.join(args.document_dir, args.project)
				if not os.path.exists(project_dir) or not os.path.isdir(project_dir):
					print(f"Error: Project directory not found: {project_dir}")
					return
			print(f"Indexing project: {args.project}")
			# Only index the specified project
			index_directory(
				args.document_dir, args.index_dir, args.max_chunk_size, 
				embedding_config, args.project, args.debug,
				args.auto_adjust_chunks, args.chars_per_dimension
			)
		else:
			print(f"Indexing all documents")
			# Index all projects
			index_directory(
				args.document_dir, args.index_dir, args.max_chunk_size,
				embedding_config, None, args.debug,
				args.auto_adjust_chunks, args.chars_per_dimension
			)
	
	# Report files where indexing was aborted
	if aborted_files:
		print("\nThe following files reached the maximum chunk limit and were partially indexed:")
		for file_path in sorted(aborted_files):
			print(f"  - {file_path}")
	
	print("\nIndexing complete")




if __name__ == "__main__":
	main()