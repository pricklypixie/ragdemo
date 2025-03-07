#!/usr/bin/env python3
"""
RAG CLI Application using Claude and the Anthropic API.

This application:
1. Indexes documents in a local directory using CPU-based embeddings
2. Accepts CLI queries
3. Retrieves relevant documents based on semantic similarity
4. Sends the query with context to Claude to generate an answer
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
import atexit
import warnings
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

import torch

# Filter resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker")

# Force CPU usage instead of Metal on MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_FALLBACK_POLICY"] = "0"

# Set number of threads for CPU operations
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads

# Disable parallelism in joblib (used by sentence-transformers)
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # Disable multiprocessing in joblib

import anthropic
from sklearn.metrics.pairwise import cosine_similarity

# Constants
MODEL = "claude-3-opus-20240229"
MAX_TOKENS = 4096
DEFAULT_INDEX_DIR = "document_index"
DEFAULT_DOCUMENT_DIR = "documents"
CHUNK_SIZE = 1000
OVERLAP = 200
TOP_K_DOCUMENTS = 3
API_TIMEOUT = 60  # Timeout for API calls in seconds
API_RETRY_ATTEMPTS = 3  # Number of retry attempts for API calls
API_RETRY_DELAY = 2  # Delay between retry attempts in seconds


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
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert the document to a dictionary for serialization."""
		return {
			"content": self.content,
			"metadata": self.metadata,
			"embedding": self.embedding if self.embedding is not None else []
		}
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'Document':
		"""Create a Document instance from a dictionary."""
		return cls(
			content=data["content"],
			metadata=data["metadata"],
			embedding=data.get("embedding", [])
		)


def timeout_handler(signum, frame):
	"""Signal handler for timeouts."""
	raise APITimeoutError("API call timed out")


def cleanup_resources():
	"""Clean up resources at exit."""
	# This helps ensure joblib resources are released
	import gc
	gc.collect()
	
	# Try to clean up multiprocessing resources
	try:
		from joblib.externals.loky import get_reusable_executor
		get_reusable_executor().shutdown(wait=True)
	except:
		pass


# Register the cleanup function to run at exit
atexit.register(cleanup_resources)


class SingleThreadSentenceTransformerEmbeddings:
	"""Service to handle generating embeddings using sentence-transformers in single-thread mode."""
	
	def __init__(self, model_name: str = "all-MiniLM-L6-v2", debug: bool = False):
		self.debug = debug
		self.model_name = model_name
		self.model = None  # Lazy-load the model
	
	def _load_model(self):
		"""Lazy-load the sentence transformer model, ensuring CPU usage and no parallelism."""
		try:
			# Force CPU
			import torch
			if self.debug:
				print(f"[DEBUG] PyTorch version: {torch.__version__}")
				print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
				print(f"[DEBUG] MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
				print(f"[DEBUG] Forcing CPU usage for embeddings")
			
			# Import here to ensure environment variables take effect
			from sentence_transformers import SentenceTransformer
			
			if self.debug:
				print(f"[DEBUG] Loading Sentence Transformer model: {self.model_name}")
			
			# Explicitly set device to CPU
			self.model = SentenceTransformer(self.model_name, device="cpu")
			
			# Disable batching to avoid multiprocessing
			if hasattr(self.model, 'max_seq_length'):
				if self.debug:
					print(f"[DEBUG] Setting batch size to 1 to avoid multiprocessing")
				self.model.max_seq_length = 512
			
			if self.debug:
				print(f"[DEBUG] Successfully loaded model on CPU")
		except ImportError:
			print("Error: sentence-transformers package is not installed.")
			print("Please install it with: pip install sentence-transformers")
			raise
	
	def create_embedding(self, text: str) -> List[float]:
		"""Generate an embedding for the given text using sentence-transformers."""
		if self.model is None:
			self._load_model()
			
		if self.debug:
			print(f"[DEBUG] Generating embedding for text of length {len(text)}")
			
		# Generate embedding - use single item not batch to avoid parallel processing
		with torch.no_grad():
			# Disable batching by passing a single string, not a list
			embedding = self.model.encode(text, 
										 convert_to_numpy=True,
										 batch_size=1,  # Force single batch
										 show_progress_bar=False).tolist()
		
		if self.debug:
			print(f"[DEBUG] Generated embedding with dimension {len(embedding)}")
			
		return embedding


class DocumentStore:
	"""Manages document storage, indexing and retrieval."""
	
	def __init__(self, index_dir: str, embedding_model: Optional[str] = None, debug: bool = False):
		self.index_dir = index_dir
		self.embedding_service = SingleThreadSentenceTransformerEmbeddings(
			model_name=embedding_model or "all-MiniLM-L6-v2", 
			debug=debug
		)
		self.documents: List[Document] = []
		self.index_path = os.path.join(index_dir, "document_index.pkl")
		self.debug = debug
		
		# Create index directory if it doesn't exist
		os.makedirs(index_dir, exist_ok=True)
		
		# Create a backup directory for the index
		self.backup_dir = os.path.join(index_dir, "backups")
		os.makedirs(self.backup_dir, exist_ok=True)
		
		# Load existing index if it exists
		if os.path.exists(self.index_path):
			self.load_index()
	
	def debug_log(self, message: str) -> None:
		"""Print debug messages if debug mode is enabled."""
		if self.debug:
			print(f"[DEBUG] {message}")
	
	def load_index(self) -> None:
		"""Load the document index from disk."""
		try:
			with open(self.index_path, 'rb') as f:
				self.documents = pickle.load(f)
			print(f"Loaded {len(self.documents)} documents from index")
		except Exception as e:
			print(f"Error loading index: {e}")
			# Try to load from backup if main index fails
			backup_files = sorted(glob.glob(os.path.join(self.backup_dir, "*.pkl")), reverse=True)
			if backup_files:
				print(f"Attempting to load from latest backup: {backup_files[0]}")
				try:
					with open(backup_files[0], 'rb') as f:
						self.documents = pickle.load(f)
					print(f"Loaded {len(self.documents)} documents from backup")
					return
				except Exception as backup_error:
					print(f"Error loading backup: {backup_error}")
			
			self.documents = []
	
	def save_index(self) -> None:
		"""Save the document index to disk."""
		# Create a backup of the current index
		if os.path.exists(self.index_path):
			backup_file = os.path.join(
				self.backup_dir, 
				f"document_index_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
			)
			try:
				with open(self.index_path, 'rb') as src, open(backup_file, 'wb') as dst:
					dst.write(src.read())
				self.debug_log(f"Created backup: {backup_file}")
			except Exception as e:
				print(f"Error creating backup: {e}")
		
		# Save the updated index
		try:
			with open(self.index_path, 'wb') as f:
				pickle.dump(self.documents, f)
			print(f"Saved {len(self.documents)} documents to index")
		except Exception as e:
			print(f"Error saving index: {e}")
	
	def chunk_text(self, text: str) -> List[str]:
		"""Split text into overlapping chunks of specified size."""
		if len(text) <= CHUNK_SIZE:
			return [text]
		
		chunks = []
		start = 0
		while start < len(text):
			end = min(start + CHUNK_SIZE, len(text))
			# If we're not at the end, try to break at a paragraph or sentence boundary
			if end < len(text):
				# Try to find paragraph break
				paragraph_break = text.rfind('\n\n', start, end)
				if paragraph_break != -1 and paragraph_break > start + CHUNK_SIZE // 2:
					end = paragraph_break
				else:
					# Try to find sentence break
					sentence_breaks = [text.rfind('. ', start, end), 
									  text.rfind('! ', start, end),
									  text.rfind('? ', start, end)]
					best_break = max(sentence_breaks)
					if best_break != -1 and best_break > start + CHUNK_SIZE // 2:
						end = best_break + 1  # Include the period
			
			chunks.append(text[start:end])
			start = end - OVERLAP
		
		return chunks
	
	def create_embedding(self, text: str) -> List[float]:
		"""Generate an embedding for the given text."""
		self.debug_log(f"Generating embedding for text of length {len(text)}")
		
		try:
			start_time = time.time()
			embedding = self.embedding_service.create_embedding(text)
			elapsed_time = time.time() - start_time
			self.debug_log(f"Embedding generated successfully in {elapsed_time:.2f} seconds")
			
			return embedding
				
		except Exception as e:
			print(f"Error generating embedding: {e}")
			self.debug_log(f"Exception details: {traceback.format_exc()}")
			raise
	
	def index_directory(self, directory: str) -> None:
		"""Index all supported documents in the specified directory."""
		files = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
		files += glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
		
		if not files:
			print(f"No supported files found in {directory}")
			return
		
		print(f"Found {len(files)} files to index")
		
		# Sort files by size (smallest first) to get some quick wins
		files = sorted(files, key=os.path.getsize)
		
		total_files = len(files)
		for i, file_path in enumerate(files, 1):
			print(f"Processing file {i}/{total_files}: {file_path}")
			try:
				self.index_file(file_path)
				# Save index after each file to avoid losing work
				self.save_index()
				
				# Force garbage collection after each file to release resources
				import gc
				gc.collect()
				
			except Exception as e:
				print(f"Failed to index {file_path}: {e}")
				self.debug_log(f"Exception details: {traceback.format_exc()}")
	
	def index_file(self, file_path: str) -> None:
		"""Index a single file by creating chunks and embeddings."""
		try:
			with open(file_path, 'r', encoding='utf-8') as f:
				content = f.read()
			
			rel_path = os.path.relpath(file_path)
			file_name = os.path.basename(file_path)
			file_size = os.path.getsize(file_path)
			
			self.debug_log(f"Processing file: {rel_path} ({file_size} bytes)")
			
			# Get file stats
			stats = os.stat(file_path)
			modified_time = datetime.fromtimestamp(stats.st_mtime)
			
			# Check if file is already indexed and up to date
			existing_docs = [doc for doc in self.documents 
						   if doc.metadata.get('file_path') == rel_path and
							  doc.metadata.get('last_modified') == modified_time.isoformat()]
			
			if existing_docs:
				print(f"File already indexed: {rel_path}")
				return
			
			# Remove any old versions of this file from the index
			self.documents = [doc for doc in self.documents 
						   if doc.metadata.get('file_path') != rel_path]
			
			print(f"Indexing: {rel_path}")
			chunks = self.chunk_text(content)
			print(f"  Split into {len(chunks)} chunks")
			
			for i, chunk in enumerate(chunks):
				print(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
				metadata = {
					'file_path': rel_path,
					'file_name': file_name,
					'chunk_index': i,
					'total_chunks': len(chunks),
					'chunk_size': len(chunk),
					'last_modified': modified_time.isoformat()
				}
				
				# Generate embedding for the chunk
				try:
					embedding = self.create_embedding(chunk)
					doc = Document(content=chunk, metadata=metadata, embedding=embedding)
					self.documents.append(doc)
					print(f"  Successfully indexed chunk {i+1}/{len(chunks)}")
					
					# Periodically save for very large files
					if len(chunks) > 10 and (i+1) % 5 == 0:
						self.debug_log("Saving intermediate progress...")
						self.save_index()
						
				except Exception as e:
					print(f"  Error creating embedding for chunk {i+1}: {e}")
					self.debug_log(f"Exception details: {traceback.format_exc()}")
					
		except Exception as e:
			print(f"Error indexing file {file_path}: {e}")
			self.debug_log(f"Exception details: {traceback.format_exc()}")
	
	def search(self, query: str, top_k: int = TOP_K_DOCUMENTS) -> List[Document]:
		"""Search for documents relevant to the query."""
		if not self.documents:
			print("No documents in index")
			return []
		
		self.debug_log(f"Searching for: '{query}'")
		
		# Create query embedding
		try:
			start_time = time.time()
			query_embedding = self.create_embedding(query)
			search_time = time.time() - start_time
			self.debug_log(f"Created query embedding in {search_time:.2f} seconds")
			
			# Calculate similarities
			similarities = []
			for doc in self.documents:
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
			
			self.debug_log(f"Found {len(top_results)} relevant documents")
			for i, (doc, sim) in enumerate(sorted_results[:top_k]):
				self.debug_log(f"  Result {i+1}: {doc.metadata.get('file_path')} (score: {sim:.4f})")
			
			return top_results
			
		except Exception as e:
			print(f"Error during search: {e}")
			self.debug_log(f"Exception details: {traceback.format_exc()}")
			return []


class RAGApplication:
	"""Main RAG application class that handles user interaction."""
	
	def __init__(self, 
				 api_key: str,
				 index_dir: str = DEFAULT_INDEX_DIR,
				 document_dir: str = DEFAULT_DOCUMENT_DIR,
				 embedding_model: Optional[str] = None,
				 debug: bool = False):
		self.api_key = api_key
		self.client = anthropic.Anthropic(api_key=api_key)
		self.document_store = DocumentStore(index_dir, embedding_model, debug)
		self.document_dir = document_dir
		self.debug = debug
	
	def debug_log(self, message: str) -> None:
		"""Print debug messages if debug mode is enabled."""
		if self.debug:
			print(f"[DEBUG] {message}")
	
	def index_documents(self) -> None:
		"""Index all documents in the document directory."""
		if not os.path.exists(self.document_dir):
			print(f"Document directory {self.document_dir} does not exist.")
			return
		
		self.document_store.index_directory(self.document_dir)
	
	def ask_question(self, query: str) -> str:
		"""Process a user query and return Claude's response."""
		try:
			# Search for relevant documents
			relevant_docs = self.document_store.search(query)
			
			if not relevant_docs:
				# If no relevant documents found, just ask Claude directly
				prompt = f"""
				User has asked: {query}
				
				Please note that I couldn't find any relevant documents in my knowledge base to help answer this question.
				Please answer based on your general knowledge, and mention that no specific documents were found.
				"""
			else:
				# Build context from relevant documents
				context_pieces = []
				for i, doc in enumerate(relevant_docs):
					source = f"{doc.metadata.get('file_path', 'Unknown document')}"
					context_pieces.append(f"Document {i+1} (Source: {source}):\n{doc.content}")
				
				context = "\n\n".join(context_pieces)
				
				# Prepare prompt with context
				prompt = f"""
				User has asked: {query}
				
				I've retrieved the following documents that might help answer this question:
				
				{context}
				
				Please answer the user's question based on the information in these documents.
				If the documents don't contain the necessary information, please say so and answer based on your general knowledge.
				In your answer, cite which documents you used.
				"""
			
			self.debug_log("Sending prompt to Claude")
			
			# Set up timeout
			signal.signal(signal.SIGALRM, timeout_handler)
			signal.alarm(API_TIMEOUT)
			
			# Get response from Claude
			start_time = time.time()
			response = self.client.messages.create(
				model=MODEL,
				max_tokens=MAX_TOKENS,
				messages=[
					{"role": "user", "content": prompt}
				]
			)
			
			# Cancel the alarm
			signal.alarm(0)
			
			elapsed_time = time.time() - start_time
			self.debug_log(f"Received response from Claude in {elapsed_time:.2f} seconds")
			
			return response.content[0].text
			
		except APITimeoutError:
			return "I'm sorry, but the request to Claude timed out. Please try again with a simpler question or check your internet connection."
		except Exception as e:
			self.debug_log(f"Exception details: {traceback.format_exc()}")
			return f"I'm sorry, but an error occurred while processing your request: {str(e)}"
		finally:
			# Make sure to cancel the alarm
			signal.alarm(0)
	
	def interactive_mode(self) -> None:
		"""Run the application in interactive mode."""
		print("RAG CLI Application - Interactive Mode")
		print("Enter 'exit' or 'quit' to end the session")
		
		while True:
			try:
				query = input("\nEnter your question: ").strip()
				
				if query.lower() in ['exit', 'quit']:
					print("Exiting...")
					break
				
				if not query:
					continue
				
				answer = self.ask_question(query)
				print("\nAnswer:")
				print(answer)
			except KeyboardInterrupt:
				print("\nInterrupted by user. Exiting...")
				break
			except Exception as e:
				print(f"Error: {e}")
				if self.debug:
					print(traceback.format_exc())

# Setup proper signal handlers
def setup_signal_handlers():
	"""Set up signal handlers for graceful termination."""
	def signal_handler(sig, frame):
		print("\nReceived termination signal. Cleaning up...")
		cleanup_resources()
		sys.exit(0)
	
	# Register signal handlers
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)


def main():
	"""Main entry point for the application."""
	# Set up signal handlers
	setup_signal_handlers()
	
	parser = argparse.ArgumentParser(description="RAG CLI Application with Claude")
	
	parser.add_argument("--api-key", type=str, help="Anthropic API key")
	parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR, 
						help="Directory to store the document index")
	parser.add_argument("--document-dir", type=str, default=DEFAULT_DOCUMENT_DIR, 
						help="Directory containing documents to index")
	parser.add_argument("--index", action="store_true", 
						help="Index documents in the document directory")
	parser.add_argument("--query", type=str, 
						help="Single query mode: ask a question and exit")
	parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
						help="Sentence Transformer model to use for embeddings")
	parser.add_argument("--debug", action="store_true",
						help="Enable debug logging")
	
	args = parser.parse_args()
	
	# Get API key from args or environment
	api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
	
	if not api_key:
		print("Error: Anthropic API key is required. Please provide it via --api-key or set the ANTHROPIC_API_KEY environment variable.")
		sys.exit(1)
	
	# Print application info
	print(f"RAG CLI Application with Claude")
	print(f"Python version: {sys.version}")
	print(f"Using embedding model: {args.embedding_model}")
	print(f"Running on: {sys.platform}")
	
	try:
		print(f"Anthropic SDK version: {anthropic.__version__}")
	except AttributeError:
		print("Anthropic SDK version: unknown")
	
	# Check if sentence-transformers is installed
	try:
		import sentence_transformers
		print(f"Sentence Transformers version: {sentence_transformers.__version__}")
	except ImportError:
		print("Error: The sentence-transformers package is required for embeddings.")
		print("Please install it with: pip install sentence-transformers")
		sys.exit(1)
	
	app = RAGApplication(
		api_key=api_key,
		index_dir=args.index_dir,
		document_dir=args.document_dir,
		embedding_model=args.embedding_model,
		debug=args.debug
	)
	
	if args.index:
		app.index_documents()
	
	if args.query:
		# Single query mode
		answer = app.ask_question(args.query)
		print(answer)
	elif not args.index:
		# If no specific command is given, enter interactive mode
		app.interactive_mode()


if __name__ == "__main__":
	main()