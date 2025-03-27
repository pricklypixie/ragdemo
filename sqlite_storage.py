#!/usr/bin/env python3
"""
SQLite Vector Storage Module for RAG Applications

This module provides vector database storage for document embeddings using SQLite with
the sqlite-vec extension for efficient vector similarity search.
"""

import os
import sys
import json
import sqlite3
import numpy as np
import time
import traceback
import struct
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

# Try to import the Python package for sqlite-vec
try:
	import sqlite_vec
except ImportError:
	# We'll handle this during database initialization
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

def serialize_f32(vector: List[float]) -> bytes:
	"""Serializes a list of floats into a compact "raw bytes" format"""
	return struct.pack("%sf" % len(vector), *vector)

def deserialize_f32(data: bytes, dimension: int) -> List[float]:
	"""Deserializes binary data back into a list of floats"""
	return list(struct.unpack("%sf" % dimension, data))

class SQLiteVectorStore:
	"""Vector store implementation using SQLite with the sqlite-vec extension."""
	
	def __init__(self, db_path: str, dimension: int, debug: bool = False):
		"""
		Initialize SQLite vector database.
		
		Args:
			db_path: Path to the SQLite database file
			dimension: Dimension of embeddings to be stored
			debug: Whether to print debug information
		"""
		self.db_path = db_path
		self.dimension = dimension
		self.debug = debug
		self.conn = None
		
		if debug:
			print(f"[DEBUG] Initializing SQLiteVectorStore with db_path: {db_path}, dimension: {dimension}")

		# Create database directory if it doesn't exist
		os.makedirs(os.path.dirname(db_path), exist_ok=True)
		
		# Initialize database
		try:
			self._init_db()
			# Ensure search function exists
			self._ensure_search_function()

		except Exception as e:
			if debug:
				print(f"[DEBUG] Error during initialization: {e}")
				import traceback
				print(traceback.format_exc())
	
	def _debug_log(self, message: str) -> None:
		"""Print debug message if debug mode is enabled."""
		if self.debug:
			print(f"[DEBUG] SQLiteVectorStore: {message}")
	

	
	
	
	def _init_db(self) -> None:
		"""Initialize the SQLite database with required tables and extensions."""
		try:
			# Create database directory if it doesn't exist
			os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
			
			# Connect to the database
			self.conn = sqlite3.connect(self.db_path)
			
			# Enable foreign keys
			self.conn.execute("PRAGMA foreign_keys = ON")
			
			# Load the SQLite vector extension
			self.conn.enable_load_extension(True)
		
			# Try different ways to load the extension
			try:
				import sqlite_vec
				sqlite_vec.load(self.conn)
				# Check if loaded successfully
				vec_version = self.conn.execute("SELECT vec_version()").fetchone()[0]
				self._debug_log(f"Loaded sqlite-vec extension version {vec_version}")
			except (ImportError, AttributeError, sqlite3.OperationalError) as e:
				self._debug_log(f"Could not load via Python API: {e}")
				
				# Try loading directly using various paths
				extension_paths = [
					"sqlite_vec",  # Default path
					"/opt/homebrew/lib/sqlite-vec",  # Mac Homebrew path
					"/usr/local/lib/sqlite-vec",  # Common Unix path
					"/usr/lib/sqlite3/sqlite-vec",  # Linux path
					"sqlite_vec.so"  # Direct .so file
				]
				
				loaded = False
				for path in extension_paths:
					try:
						self._debug_log(f"Trying to load extension from: {path}")
						self.conn.load_extension(path)
						loaded = True
						self._debug_log(f"Successfully loaded from {path}")
						break
					except sqlite3.OperationalError as e:
						self._debug_log(f"Failed to load from {path}: {e}")
				
				if not loaded:
					self._debug_log("Could not load sqlite-vec extension. Vector search will not be available.")
			
			# Create metadata table
			self.conn.execute("""
			CREATE TABLE IF NOT EXISTS metadata (
				id INTEGER PRIMARY KEY,
				file_path TEXT NOT NULL,
				file_name TEXT NOT NULL,
				project TEXT NOT NULL,
				embedding_model TEXT NOT NULL,
				embedding_type TEXT NOT NULL,
				chunk_index INTEGER NOT NULL,
				total_chunks INTEGER NOT NULL,
				chunk_size INTEGER NOT NULL,
				paragraphs INTEGER,
				last_modified TEXT,
				json_metadata TEXT,
				UNIQUE(file_path, chunk_index)
			)
			""")
			
			# Create content table
			self.conn.execute("""
			CREATE TABLE IF NOT EXISTS content (
				id INTEGER PRIMARY KEY,
				metadata_id INTEGER NOT NULL,
				content TEXT NOT NULL,
				FOREIGN KEY(metadata_id) REFERENCES metadata(id) ON DELETE CASCADE
			)
			""")
			
			# Create embeddings table
			self.conn.execute(f"""
			CREATE TABLE IF NOT EXISTS embeddings (
				id INTEGER PRIMARY KEY,
				metadata_id INTEGER NOT NULL,
				embedding BLOB NOT NULL,
				FOREIGN KEY(metadata_id) REFERENCES metadata(id) ON DELETE CASCADE
			)
			""")
			
			# Create vector index for the embeddings using vec0
			try:
				# First check if the extension is really loaded
				vec_version = self.conn.execute("SELECT vec_version()").fetchone()[0]
				self._debug_log(f"Extension confirmed loaded: {vec_version}")
				
				# Use vec0 with the proper float[dimension] syntax
				table_exists = self.conn.execute(
					"SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_index'"
				).fetchone()
				
				if not table_exists:
					self._debug_log(f"Creating vector table with dimension {self.dimension}")
					# Create the vector table with the correct column type
					self.conn.execute(f"""
					CREATE VIRTUAL TABLE embeddings_index USING vec0(
						embedding float[{self.dimension}],
						metadata_id INTEGER
					)
					""")
					self._debug_log(f"Successfully created vector index with dimension {self.dimension}")
				
			except sqlite3.OperationalError as e:
				self._debug_log(f"Error creating vector index: {e}")
				if self.debug:
					import traceback
					self._debug_log(traceback.format_exc())
			
			# Create indexes for faster retrieval
			self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_project ON metadata(project)")
			self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_file_path ON metadata(file_path)")
			self.conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_id ON embeddings(metadata_id)")
			
			self.conn.commit()
			self._debug_log(f"Initialized database at {self.db_path} with dimension {self.dimension}")
				
		except Exception as e:
			self._debug_log(f"Error initializing database: {e}")
			if self.debug:
				import traceback
				traceback.print_exc()
			if self.conn:
				self.conn.close()
				self.conn = None
			raise
	
	
	
	
	
	def _ensure_connection(self) -> sqlite3.Connection:
		"""Ensure we have an active database connection."""
		if self.conn is None:
			self.conn = sqlite3.connect(self.db_path)
			self.conn.execute("PRAGMA foreign_keys = ON")
			
			# Load the SQLite vector extension
			self.conn.enable_load_extension(True)
			try:
				# Try Python API first
				import sqlite_vec
				sqlite_vec.load(self.conn)
			except (ImportError, AttributeError):
				# Fall back to manual loading
				try:
					self.conn.load_extension("sqlite_vec")
				except sqlite3.OperationalError:
					try:
						# Try MacOS Homebrew path
						self.conn.load_extension("/opt/homebrew/lib/sqlite-vec")
					except sqlite3.OperationalError:
						try:
							# Try Linux path
							self.conn.load_extension("/usr/lib/sqlite3/sqlite-vec")
						except sqlite3.OperationalError as e:
							self._debug_log(f"Failed to load sqlite-vec extension: {e}")
		
		return self.conn
		
		
	def _ensure_search_function(self) -> None:
		"""Ensure the vector search function is available."""
		conn = self._ensure_connection()
		
		try:
			# First check if the embeddings_index table exists
			table_exists = conn.execute(
				"SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_index'"
			).fetchone()
			
			if not table_exists:
				self._debug_log("Cannot create search function because embeddings_index table doesn't exist yet")
				return  # Skip creating search function if table doesn't exist
			
			# Try to use the search function to see if it exists
			try:
				# Test if the function already exists
				conn.execute("SELECT * FROM embeddings_index_search LIMIT 0")
				self._debug_log("Vector search function already exists")
				return  # Function exists
			except sqlite3.OperationalError:
				# Need to create the function
				try:
					# Use the correct syntax for creating a search table
					conn.execute("""
					CREATE VIRTUAL TABLE embeddings_index_search 
					USING vec0(embeddings_index, embedding, metadata_id);
					""")
					self._debug_log("Created vector search function")
				except sqlite3.OperationalError as e:
					self._debug_log(f"Error creating search function: {e}")
					# Don't raise here, we'll use fallback method later
		except Exception as e:
			self._debug_log(f"Error ensuring search function: {e}")
			# Don't raise here, we'll use fallback method later
			
			
			
			
			
			
				
	def close(self) -> None:
		"""Close the database connection."""
		if self.conn:
			self.conn.close()
			self.conn = None
	
	def store_document(self, document: Document) -> bool:
		"""
		Store a document in the database.
		
		Args:
			document: Document to store
			
		Returns:
			True if successful, False otherwise
		"""
		if not document.embedding:
			self._debug_log("Cannot store document without embedding")
			return False
		
		conn = self._ensure_connection()
		
		try:
			# Begin transaction
			conn.execute("BEGIN")
			
			# Convert metadata to JSON for flexible storage of additional fields
			json_metadata = json.dumps({
				k: v for k, v in document.metadata.items() 
				if k not in ['file_path', 'file_name', 'project', 'embedding_model', 
							'embedding_type', 'chunk_index', 'total_chunks', 
							'chunk_size', 'paragraphs', 'last_modified']
			})
			
			# Insert metadata
			cursor = conn.execute("""
			INSERT OR REPLACE INTO metadata 
			(file_path, file_name, project, embedding_model, embedding_type, 
			 chunk_index, total_chunks, chunk_size, paragraphs, last_modified, json_metadata)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			""", (
				document.metadata.get('file_path', 'unknown'),
				document.metadata.get('file_name', 'unknown'),
				document.metadata.get('project', 'master'),
				document.metadata.get('embedding_model', 'unknown'),
				document.metadata.get('embedding_type', 'unknown'),
				document.metadata.get('chunk_index', 0),
				document.metadata.get('total_chunks', 1),
				document.metadata.get('chunk_size', len(document.content)),
				document.metadata.get('paragraphs', 1),
				document.metadata.get('last_modified', None),
				json_metadata
			))
			
			metadata_id = cursor.lastrowid
			
			# Insert content
			conn.execute("""
			INSERT OR REPLACE INTO content (metadata_id, content)
			VALUES (?, ?)
			""", (metadata_id, document.content))
			
			# Insert embedding - store as binary using struct.pack
			embedding_blob = serialize_f32(document.embedding)
			conn.execute("""
			INSERT OR REPLACE INTO embeddings (metadata_id, embedding)
			VALUES (?, ?)
			""", (metadata_id, embedding_blob))
			
			# Delete existing vector entries for this metadata_id
			try:
				conn.execute("DELETE FROM embeddings_index WHERE metadata_id = ?", (metadata_id,))
			except sqlite3.OperationalError:
				pass  # Table might not exist yet
			
			# Insert into vector index
			try:
				conn.execute("""
				INSERT INTO embeddings_index (rowid, embedding, metadata_id)
				VALUES (?, ?, ?)
				""", (metadata_id, embedding_blob, metadata_id))
				
				self._debug_log(f"Stored document in vector index with metadata_id {metadata_id}")
			except sqlite3.OperationalError as e:
				self._debug_log(f"Error inserting into vector index: {e}")
				if "no such table" in str(e):
					# Create the table if it doesn't exist
					try:
						conn.execute(f"""
						CREATE VIRTUAL TABLE embeddings_index USING vec0(
							embedding float[{self.dimension}],
							metadata_id INTEGER
						)
						""")
						
						conn.execute("""
						INSERT INTO embeddings_index (rowid, embedding, metadata_id)
						VALUES (?, ?, ?)
						""", (metadata_id, embedding_blob, metadata_id))
						
						self._debug_log(f"Created vector index and inserted document")
					except sqlite3.OperationalError as e2:
						self._debug_log(f"Failed to create vector index: {e2}")
			
			# Commit transaction
			conn.commit()
			self._debug_log(f"Stored document {document.metadata.get('file_path')} chunk {document.metadata.get('chunk_index')}")
			return True
			
		except Exception as e:
			conn.rollback()
			self._debug_log(f"Error storing document: {e}")
			if self.debug:
				traceback.print_exc()
			return False
	









		
		
	def batch_store_documents(self, documents: List[Document]) -> int:
		"""Store multiple documents in the database using efficient batch operations."""
		if not documents:
			return 0
		
		conn = self._ensure_connection()
		successful = 0
		
		try:
			# Begin transaction
			conn.execute("BEGIN")
			
			# First collect all file paths we'll be updating
			file_paths = set()
			for doc in documents:
				file_path = doc.metadata.get('file_path', 'unknown')
				if file_path:
					file_paths.add(file_path)
			
			# Delete all existing entries for these file paths in one operation
			if file_paths:
				placeholders = ','.join(['?'] * len(file_paths))
				
				# First delete from vector index if it exists
				try:
					# Get a list of metadata_ids to delete from embeddings_index
					cursor = conn.execute(f"""
					SELECT id FROM metadata WHERE file_path IN ({placeholders})
					""", list(file_paths))
					
					metadata_ids = [row[0] for row in cursor.fetchall()]
					
					# If we found any, delete them from embeddings_index
					if metadata_ids:
						metadata_placeholders = ','.join(['?'] * len(metadata_ids))
						conn.execute(f"""
						DELETE FROM embeddings_index WHERE metadata_id IN ({metadata_placeholders})
						""", metadata_ids)
				except sqlite3.OperationalError as e:
					if self.debug:
						self._debug_log(f"No need to delete from embeddings_index: {e}")
				
				# Delete from content and embeddings via metadata join
				conn.execute(f"""
				DELETE FROM content WHERE metadata_id IN 
				(SELECT id FROM metadata WHERE file_path IN ({placeholders}))
				""", list(file_paths))
				
				conn.execute(f"""
				DELETE FROM embeddings WHERE metadata_id IN 
				(SELECT id FROM metadata WHERE file_path IN ({placeholders}))
				""", list(file_paths))
				
				# Finally delete from metadata
				conn.execute(f"""
				DELETE FROM metadata WHERE file_path IN ({placeholders})
				""", list(file_paths))
			
			# Now batch insert all documents at once
			metadata_values = []
			content_values = []
			embedding_values = []
			vector_values = []
			
			# Get the next available ID for metadata
			cursor = conn.execute("SELECT MAX(id) FROM metadata")
			result = cursor.fetchone()
			next_id = (result[0] or 0) + 1
			
			for doc in documents:
				if not doc.embedding:
					continue
					
				file_path = doc.metadata.get('file_path', 'unknown')
				json_metadata = json.dumps({
					k: v for k, v in doc.metadata.items() 
					if k not in ['file_path', 'file_name', 'project', 'embedding_model', 
							'embedding_type', 'chunk_index', 'total_chunks', 
							'chunk_size', 'paragraphs', 'last_modified']
				})
				
				metadata_id = next_id
				next_id += 1
				
				# Add to metadata batch
				metadata_values.append((
					metadata_id,
					doc.metadata.get('file_path', 'unknown'),
					doc.metadata.get('file_name', 'unknown'),
					doc.metadata.get('project', 'master'),
					doc.metadata.get('embedding_model', 'unknown'),
					doc.metadata.get('embedding_type', 'unknown'),
					doc.metadata.get('chunk_index', 0),
					doc.metadata.get('total_chunks', 1),
					doc.metadata.get('chunk_size', len(doc.content)),
					doc.metadata.get('paragraphs', 1),
					doc.metadata.get('last_modified', None),
					json_metadata
				))
				
				# Add to content batch
				content_values.append((
					metadata_id,
					doc.content
				))
				
				# Add to embeddings batch
				embedding_blob = serialize_f32(doc.embedding)
				embedding_values.append((
					metadata_id,
					embedding_blob
				))
				
				# Add to vector index batch
				vector_values.append((
					metadata_id,  # Use as rowid
					embedding_blob,
					metadata_id   # metadata_id
				))
				
				successful += 1
				
			# Execute all batch inserts
			
			# Metadata batch insert
			conn.executemany("""
			INSERT INTO metadata 
			(id, file_path, file_name, project, embedding_model, embedding_type, 
			chunk_index, total_chunks, chunk_size, paragraphs, last_modified, json_metadata)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			""", metadata_values)
			
			# Content batch insert
			conn.executemany("""
			INSERT INTO content (metadata_id, content)
			VALUES (?, ?)
			""", content_values)
			
			# Embeddings batch insert
			conn.executemany("""
			INSERT INTO embeddings (metadata_id, embedding)
			VALUES (?, ?)
			""", embedding_values)
			
			# Vector index batch insert (might fail if table doesn't exist)
			try:
				# First check if table exists
				table_exists = conn.execute(
					"SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_index'"
				).fetchone()
				
				if table_exists:
					# Delete any existing entries for these metadata_ids to avoid constraint violations
					for metadata_id, _, _ in vector_values:
						try:
							conn.execute("DELETE FROM embeddings_index WHERE rowid = ?", (metadata_id,))
						except sqlite3.OperationalError as e:
							self._debug_log(f"Error deleting from embeddings_index: {e}")
					
					# Now insert the new entries
					try:
						conn.executemany("""
						INSERT OR REPLACE INTO embeddings_index (rowid, embedding, metadata_id)
						VALUES (?, ?, ?)
						""", vector_values)
					except sqlite3.OperationalError as e:
						self._debug_log(f"Error inserting into vector index: {e}")
				else:
					# Create the table
					conn.execute(f"""
					CREATE VIRTUAL TABLE embeddings_index USING vec0(
						embedding float[{self.dimension}],
						metadata_id INTEGER
					)
					""")
					
					conn.executemany("""
					INSERT INTO embeddings_index (rowid, embedding, metadata_id)
					VALUES (?, ?, ?)
					""", vector_values)
					
					if self.debug:
						self._debug_log(f"Created vector index table and inserted {len(vector_values)} items")
			except sqlite3.OperationalError as e:
				self._debug_log(f"Error working with vector index: {e}")
			
			# Commit the entire transaction
			conn.commit()
			self._debug_log(f"Batch stored {successful} documents in single transaction")
			return successful
			
		except Exception as e:
			conn.rollback()
			self._debug_log(f"Error in batch store: {e}")
			if self.debug:
				import traceback
				traceback.print_exc()
			return successful
		
		
		

	def delete_by_file_path(self, file_path: str) -> int:
		"""
		Delete all documents associated with a specific file path.
		
		Args:
			file_path: Path of the file whose documents should be deleted
			
		Returns:
			Number of documents deleted
		"""
		conn = self._ensure_connection()
		
		try:
			# Start transaction
			conn.execute("BEGIN")
			
			# Find metadata IDs to delete
			cursor = conn.execute("SELECT id FROM metadata WHERE file_path = ?", (file_path,))
			metadata_ids = [row[0] for row in cursor.fetchall()]
			
			if not metadata_ids:
				conn.commit()
				return 0
			
			# Delete from embeddings_index
			try:
				for metadata_id in metadata_ids:
					conn.execute("DELETE FROM embeddings_index WHERE metadata_id = ?", (metadata_id,))
			except sqlite3.OperationalError:
				# Table might not exist
				pass
			
			# Delete from embeddings
			for metadata_id in metadata_ids:
				conn.execute("DELETE FROM embeddings WHERE metadata_id = ?", (metadata_id,))
			
			# Delete from content
			for metadata_id in metadata_ids:
				conn.execute("DELETE FROM content WHERE metadata_id = ?", (metadata_id,))
			
			# Delete from metadata
			cursor = conn.execute("DELETE FROM metadata WHERE file_path = ?", (file_path,))
			deleted_count = cursor.rowcount
			
			# Commit transaction
			conn.commit()
			self._debug_log(f"Deleted {deleted_count} documents for file: {file_path}")
			return deleted_count
			
		except Exception as e:
			conn.rollback()
			self._debug_log(f"Error deleting documents: {e}")
			if self.debug:
				traceback.print_exc()
			return 0
	




	
	def clear_project(self, project: str) -> int:
		"""Clear all documents for a specific project much more efficiently."""
		conn = self._ensure_connection()
		
		try:
			# Start transaction
			conn.execute("BEGIN")
			
			# Get count before deletion for return value
			cursor = conn.execute("SELECT COUNT(*) FROM metadata WHERE project = ?", (project,))
			count = cursor.fetchone()[0]
			
			# Delete all at once using JOIN
			conn.execute("""
			DELETE FROM content WHERE metadata_id IN 
			(SELECT id FROM metadata WHERE project = ?)
			""", (project,))
			
			conn.execute("""
			DELETE FROM embeddings WHERE metadata_id IN 
			(SELECT id FROM metadata WHERE project = ?)
			""", (project,))
			
			# Try to delete from vector index if it exists
			try:
				conn.execute("""
				DELETE FROM embeddings_index WHERE metadata_id IN 
				(SELECT id FROM metadata WHERE project = ?)
				""", (project,))
			except sqlite3.OperationalError:
				pass
			
			# Finally delete from metadata
			conn.execute("DELETE FROM metadata WHERE project = ?", (project,))
			
			# Commit transaction
			conn.commit()
			self._debug_log(f"Cleared project {project}: deleted {count} documents")
			return count
			
		except Exception as e:
			conn.rollback()
			self._debug_log(f"Error clearing project: {e}")
			if self.debug:
				import traceback
				traceback.print_exc()
			return 0
	
	
	

	
	
	
	def search_similar(self, query_vector: List[float], 
		  top_k: int = 3, project: Optional[str] = None,
		  filters: Optional[Dict[str, Any]] = None) -> List[Document]:
		"""
		Search for documents similar to the query vector using sqlite-vec.
		"""
		if not query_vector:
			self._debug_log("Cannot search with empty query vector")
			return []
		
		conn = self._ensure_connection()
		
		# Add cosine similarity function for potential fallback
		add_cosine_similarity_function(conn)
		
		try:
			# Convert query vector to binary data
			query_blob = serialize_f32(query_vector)
			
			# Check if the vector index table exists
			table_exists = conn.execute(
				"SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_index'"
			).fetchone()
			
			if not table_exists:
				self._debug_log("Vector index table 'embeddings_index' not found, falling back to direct search")
				raise ValueError("No embeddings_index table found")
			
			# Check if search function exists or try to create it
			try:
				# Try to use the search function
				base_query = """
				SELECT m.id, m.file_path, m.file_name, m.project, m.embedding_model, 
					m.embedding_type, m.chunk_index, m.total_chunks, m.chunk_size, 
					m.paragraphs, m.last_modified, m.json_metadata, c.content, 
					e.id as embedding_id, distance
				FROM embeddings_index_search(?, ?) as search
				JOIN metadata m ON m.id = search.metadata_id
				JOIN content c ON c.metadata_id = m.id
				JOIN embeddings e ON e.metadata_id = m.id
				"""
				
				params = [query_blob, top_k]
				
				# Add filters
				where_clauses = []
				
				# Add project filter if specified
				if project:
					where_clauses.append("m.project = ?")
					params.append(project)
				
				# Add additional filters if specified
				if filters:
					for key, value in filters.items():
						if key in ['file_path', 'file_name', 'embedding_model', 'embedding_type']:
							where_clauses.append(f"m.{key} = ?")
							params.append(value)
				
				# Add WHERE clause if we have any conditions
				if where_clauses:
					base_query += " WHERE " + " AND ".join(where_clauses)
				
				# Execute the query
				start_time = time.time()
				cursor = conn.execute(base_query, params)
				rows = cursor.fetchall()
			
				# Convert rows to Document objects
				results = []
				for row in rows:
					(metadata_id, file_path, file_name, doc_project, embedding_model, 
					embedding_type, chunk_index, total_chunks, chunk_size, 
					paragraphs, last_modified, json_metadata, content, 
					embedding_id, distance) = row
					
					# Build metadata dictionary
					metadata = {
						'file_path': file_path,
						'file_name': file_name,
						'project': doc_project,
						'embedding_model': embedding_model,
						'embedding_type': embedding_type,
						'chunk_index': chunk_index,
						'total_chunks': total_chunks,
						'chunk_size': chunk_size,
						'paragraphs': paragraphs,
						'last_modified': last_modified,
					}
					
					# Add any additional metadata from the JSON field
					if json_metadata:
						try:
							additional_metadata = json.loads(json_metadata)
							metadata.update(additional_metadata)
						except:
							pass
					
					# Convert L2 distance to cosine similarity 
					# (assumes normalized vectors) where 1.0 is identical and 0.0 is completely dissimilar
					# Formula: cosine_sim = 1.0 - (L2_distance^2 / 2.0)
					# For non-normalized vectors, we should normalize the distance differently
					l2_distance = float(distance)
					cosine_sim = 1.0 - (l2_distance**2 / 2.0)
					
					# Ensure it's in the right range
					cosine_sim = max(0.0, min(1.0, cosine_sim))
					
					# Add to metadata
					metadata['similarity'] = cosine_sim
					
					self._debug_log(f"L2 distance: {l2_distance:.4f}, calculated similarity: {cosine_sim:.4f}")
					
					# Get the embedding for this document
					embedding_cursor = conn.execute(
						"SELECT embedding FROM embeddings WHERE id = ?", 
						(embedding_id,)
					)
					embedding_row = embedding_cursor.fetchone()
					
					# Deserialize the embedding from binary to a list of floats
					embedding = None
					if embedding_row:
						try:
							embedding = deserialize_f32(embedding_row[0], self.dimension)
						except:
							self._debug_log(f"Error deserializing embedding for document {metadata_id}")
					
					# Create Document object
					doc = Document(content=content, metadata=metadata, embedding=embedding)
					results.append(doc)
				
				elapsed = time.time() - start_time
				self._debug_log(f"Search completed in {elapsed:.4f}s, found {len(results)} results")
				
				# Sort by similarity (highest first)
				results.sort(key=lambda doc: doc.metadata.get('similarity', 0), reverse=True)
				
				return results
				
			except Exception as e:
				self._debug_log(f"Vector search failed: {e}, using fallback method")
				# Fall back to direct cosine similarity search
				
				# Use the fallback method with cosine_similarity
				fallback_query = """
				SELECT m.id, m.file_path, m.file_name, m.project, m.embedding_model, 
					m.embedding_type, m.chunk_index, m.total_chunks, m.chunk_size, 
					m.paragraphs, m.last_modified, m.json_metadata, c.content, 
					e.id as embedding_id, cosine_similarity(e.embedding, ?) as similarity
				FROM metadata m
				JOIN content c ON c.metadata_id = m.id
				JOIN embeddings e ON e.metadata_id = m.id
				"""
				
				fallback_params = [query_blob]
				
				# Add project filter if specified
				where_clauses = []
				if project:
					where_clauses.append("m.project = ?")
					fallback_params.append(project)
				
				# Add WHERE clause if we have project filter
				if where_clauses:
					fallback_query += " WHERE " + " AND ".join(where_clauses)
				
				# Add order and limit
				fallback_query += " ORDER BY similarity DESC LIMIT ?"
				fallback_params.append(top_k)
				
				try:
					# Execute the fallback query
					self._debug_log("Executing fallback cosine similarity search")
					cursor = conn.execute(fallback_query, fallback_params)
					rows = cursor.fetchall()
				
					# Process results (similar to above)
					results = []
					for row in rows:
						# ... (same processing as above)
						(metadata_id, file_path, file_name, doc_project, embedding_model, 
						embedding_type, chunk_index, total_chunks, chunk_size, 
						paragraphs, last_modified, json_metadata, content, 
						embedding_id, similarity) = row
						
						# Build metadata dictionary
						metadata = {
							'file_path': file_path,
							'file_name': file_name,
							'project': doc_project,
							'embedding_model': embedding_model,
							'embedding_type': embedding_type,
							'chunk_index': chunk_index,
							'total_chunks': total_chunks,
							'chunk_size': chunk_size,
							'paragraphs': paragraphs,
							'last_modified': last_modified,
						}
						
						# Add any additional metadata from the JSON field
						if json_metadata:
							try:
								additional_metadata = json.loads(json_metadata)
								metadata.update(additional_metadata)
							except:
								pass
						
						# Add similarity as a metadata field - this should already be a proper similarity score
						metadata['similarity'] = float(similarity)
						
						self._debug_log(f"Fallback cosine similarity: {similarity:.4f}")
						
						# Get the embedding for this document
						embedding_cursor = conn.execute(
							"SELECT embedding FROM embeddings WHERE id = ?", 
							(embedding_id,)
						)
						embedding_row = embedding_cursor.fetchone()
						
						# Deserialize the embedding from binary to a list of floats
						embedding = None
						if embedding_row:
							try:
								embedding = deserialize_f32(embedding_row[0], self.dimension)
							except:
								self._debug_log(f"Error deserializing embedding for document {metadata_id}")
						
						# Create Document object
						doc = Document(content=content, metadata=metadata, embedding=embedding)
						results.append(doc)
					
					self._debug_log(f"Fallback search found {len(results)} results")
					return results
			
				except Exception as fallback_error:
					self._debug_log(f"Fallback search also failed: {fallback_error}")
					return []
		
		except Exception as some_other_error:
			self._debug_log(f"Complete search fail: {some_other_error}")
			return []
	
	
	
	def document_count(self, project: Optional[str] = None) -> int:
		"""
		Get the count of documents in the database.
		
		Args:
			project: Optional project to filter by
			
		Returns:
			Number of documents
		"""
		conn = self._ensure_connection()
		
		try:
			if project:
				cursor = conn.execute("SELECT COUNT(*) FROM metadata WHERE project = ?", (project,))
			else:
				cursor = conn.execute("SELECT COUNT(*) FROM metadata")
			
			return cursor.fetchone()[0]
			
			# Close connection
			conn.close()
			
		except Exception as e:
			self._debug_log(f"Error counting documents: {e}")
			return 0
	
	
	
	
	
	def get_all_documents(self, project: Optional[str] = None) -> List[Document]:
		"""
		Retrieve all documents from the database.
		
		Args:
			project: Optional project to filter by
			
		Returns:
			List of Document objects
		"""
		conn = self._ensure_connection()
		
		try:
			# Build the query
			query = """
			SELECT m.id, m.file_path, m.file_name, m.project, m.embedding_model, 
				   m.embedding_type, m.chunk_index, m.total_chunks, m.chunk_size, 
				   m.paragraphs, m.last_modified, m.json_metadata, c.content, e.embedding
			FROM metadata m
			JOIN content c ON c.metadata_id = m.id
			LEFT JOIN embeddings e ON e.metadata_id = m.id
			"""
			
			params = []
			
			# Add project filter if specified
			if project:
				query += " WHERE m.project = ?"
				params.append(project)
			
			# Execute the query
			cursor = conn.execute(query, params)
			rows = cursor.fetchall()
			
			# Convert rows to Document objects
			results = []
			for row in rows:
				(metadata_id, file_path, file_name, doc_project, embedding_model, 
				 embedding_type, chunk_index, total_chunks, chunk_size, 
				 paragraphs, last_modified, json_metadata, content, embedding_blob) = row
				
				# Build metadata dictionary
				metadata = {
					'file_path': file_path,
					'file_name': file_name,
					'project': doc_project,
					'embedding_model': embedding_model,
					'embedding_type': embedding_type,
					'chunk_index': chunk_index,
					'total_chunks': total_chunks,
					'chunk_size': chunk_size,
					'paragraphs': paragraphs,
					'last_modified': last_modified,
				}
				
				# Add any additional metadata from the JSON field
				if json_metadata:
					try:
						additional_metadata = json.loads(json_metadata)
						metadata.update(additional_metadata)
					except:
						pass
				
				# Deserialize the embedding from binary to a list of floats
				embedding = None
				if embedding_blob:
					try:
						embedding = deserialize_f32(embedding_blob, self.dimension)
					except:
						self._debug_log(f"Error deserializing embedding for document {metadata_id}")
				
				# Create Document object
				doc = Document(content=content, metadata=metadata, embedding=embedding)
				results.append(doc)
			
			self._debug_log(f"Retrieved {len(results)} documents")
			return results
			
		except Exception as e:
			self._debug_log(f"Error retrieving documents: {e}")
			if self.debug:
				traceback.print_exc()
			return []
	
	def get_projects(self) -> List[Dict[str, Any]]:
		"""
		Get a list of all projects with document counts.
		
		Returns:
			List of dictionaries with project info
		"""
		conn = self._ensure_connection()
		
		try:
			cursor = conn.execute("""
			SELECT project, COUNT(*) as doc_count, 
				   embedding_model, embedding_type
			FROM metadata
			GROUP BY project, embedding_model, embedding_type
			ORDER BY project
			""")
			
			rows = cursor.fetchall()
			
			# Organize results by project
			projects = {}
			for row in rows:
				project, count, model, embed_type = row
				if project not in projects:
					projects[project] = {
						'name': project,
						'total_documents': 0,
						'embedding_types': []
					}
				
				projects[project]['total_documents'] += count
				projects[project]['embedding_types'].append({
					'model': model,
					'type': embed_type,
					'count': count
				})
			
			return list(projects.values())
			
		except Exception as e:
			self._debug_log(f"Error getting projects: {e}")
			return []

# Add the cosine_similarity function to SQLite



	
	
def add_cosine_similarity_function(conn):
	"""Add a cosine similarity function to SQLite connection"""
	
	def cosine_similarity(blob1, blob2):
		"""Calculate cosine similarity between two serialized vectors"""
		# Get blob size to determine vector dimension
		blob1_size = len(blob1)
		dimension = blob1_size // 4  # 4 bytes per float
		
		# Deserialize blobs to vectors
		vec1 = struct.unpack(f"{dimension}f", blob1)
		vec2 = struct.unpack(f"{dimension}f", blob2)
		
		# Calculate dot product
		dot_product = sum(a * b for a, b in zip(vec1, vec2))
		
		# Calculate magnitudes
		magnitude1 = sum(a * a for a in vec1) ** 0.5
		magnitude2 = sum(b * b for b in vec2) ** 0.5
		
		# Avoid division by zero
		if magnitude1 < 1e-10 or magnitude2 < 1e-10:
			return 0.0
			
		# Calculate cosine similarity
		similarity = dot_product / (magnitude1 * magnitude2)
		
		# Ensure it's in the range [0, 1]
		# Theoretically cosine can be [-1, 1] but for document embeddings 
		# we typically expect positive values
		similarity = max(0.0, min(1.0, similarity))
		
		return similarity
	
	conn.create_function("cosine_similarity", 2, cosine_similarity)
	
	
	
	
	

# Helper functions for compatibility with the existing application

def get_db_path(index_dir: str, project: str) -> str:
	"""Get the database path for a project."""
	if project == "master":
		return os.path.join(index_dir, "master.db")
	else:
		# Create project subdirectory in the index directory
		project_dir = os.path.join(index_dir, project)
		os.makedirs(project_dir, exist_ok=True)
		return os.path.join(project_dir, f"{project}.db")

def document_count(index_dir: str, project: str) -> int:
	"""
	Get the count of documents for a project.
	
	Args:
		index_dir: Base index directory
		project: Project name
		
	Returns:
		Number of documents in the project
	"""
	db_path = get_db_path(index_dir, project)
	
	# Check if the database exists
	if not os.path.exists(db_path):
		return 0
	
	# Determine dimension (use default if not found)
	dimension = 384
	
	# Create vector store
	store = SQLiteVectorStore(db_path, dimension, False)
	
	# Get document count
	count = store.document_count(project)
	
	# Close the connection
	store.close()
	
	return count

def load_documents(index_dir: str, project: str, limit: int = None, debug: bool = False) -> List[Document]:
	"""
	Load documents for a project from SQLite database.
	Compatible with the existing application's load_index function.
	
	Args:
		index_dir: Base index directory
		project: Project name
		limit: Optional limit on the number of documents to load (useful for debugging)
		debug: Whether to enable debug output
		
	Returns:
		List of Document objects
	"""
	db_path = get_db_path(index_dir, project)
	if debug:
		print(f"[DEBUG] Database should be at: {db_path}")
	
	# Check if the database exists
	if not os.path.exists(db_path):
		if debug:
			print(f"[DEBUG] Database not found at: {db_path}")
		return []
	
	# Determine dimension (read from first document or use default)
	dimension = 384
	try:
		conn = sqlite3.connect(db_path)
		
		# Add custom function for cosine similarity
		add_cosine_similarity_function(conn)
		
		# Try to determine dimension from the first document
		cursor = conn.execute("SELECT embedding FROM embeddings LIMIT 1")
		result = cursor.fetchone()
		if result and result[0]:
			# Extract dimension from blob length (4 bytes per float)
			dimension = len(result[0]) // 4
		conn.close()
	except:
		# Use default dimension
		pass
	
	# Create vector store
	store = SQLiteVectorStore(db_path, dimension, debug)
	
	try:
		if limit is not None:
			# For debugging, just get a limited number of documents
			conn = store._ensure_connection()
			
			# Add custom function for cosine similarity
			add_cosine_similarity_function(conn)
			
			# Build a limited query with LIMIT clause
			query = f"""
			SELECT m.id, m.file_path, m.file_name, m.project, m.embedding_model, 
				m.embedding_type, m.chunk_index, m.total_chunks, m.chunk_size, 
				m.paragraphs, m.last_modified, m.json_metadata, c.content, e.embedding
			FROM metadata m
			JOIN content c ON c.metadata_id = m.id
			LEFT JOIN embeddings e ON e.metadata_id = m.id
			WHERE m.project = ?
			LIMIT {limit}
			"""
			
			cursor = conn.execute(query, (project,))
			rows = cursor.fetchall()
			
			# Convert rows to Document objects
			documents = []
			for row in rows:
				(metadata_id, file_path, file_name, doc_project, embedding_model, 
				embedding_type, chunk_index, total_chunks, chunk_size, 
				paragraphs, last_modified, json_metadata, content, embedding_blob) = row
				
				# Build metadata dictionary
				metadata = {
					'file_path': file_path,
					'file_name': file_name,
					'project': doc_project,
					'embedding_model': embedding_model,
					'embedding_type': embedding_type,
					'chunk_index': chunk_index,
					'total_chunks': total_chunks,
					'chunk_size': chunk_size,
					'paragraphs': paragraphs,
					'last_modified': last_modified,
				}
				
				# Add any additional metadata from the JSON field
				if json_metadata:
					try:
						additional_metadata = json.loads(json_metadata)
						metadata.update(additional_metadata)
					except:
						pass
				
				# Deserialize the embedding blob
				embedding = None
				if embedding_blob:
					try:
						embedding = deserialize_f32(embedding_blob, dimension)
					except:
						if debug:
							print(f"[DEBUG] Error deserializing embedding for document {metadata_id}")
				
				# Create Document object
				doc = Document(content=content, metadata=metadata, embedding=embedding)
				documents.append(doc)
				
			if debug:
				print(f"[DEBUG] Loaded {len(documents)} sample documents (limit={limit}) from SQLite database")
				
			return documents
		else:
			# Get all documents (typically not used with SQLite since we use vector search directly)
			documents = store.get_all_documents(project)
	finally:
		# Close the connection when done
		store.close()
	
	if debug:
		print(f"[DEBUG] Loaded {len(documents)} documents from SQLite database: {db_path}")
	
	return documents

def save_documents(documents: List[Document], index_dir: str, project: str, 
				  dimension: int = 384, debug: bool = False) -> bool:
	"""
	Save documents for a project to SQLite database.
	Compatible with the existing application's save_index function.
	
	Args:
		documents: List of Document objects to save
		index_dir: Base index directory
		project: Project name
		dimension: Embedding dimension
		debug: Whether to enable debug output
		
	Returns:
		True if successful, False otherwise
	"""
	db_path = get_db_path(index_dir, project)
	
	if debug:
		print(f"[DEBUG] Saving documents to SQLite DB: {db_path}")
		print(f"[DEBUG] Project: {project}, Dimension: {dimension}")
		print(f"[DEBUG] Document count: {len(documents)}")
		# Print first few document details
		for i, doc in enumerate(documents[:3]):
			if i == 0:
				print(f"[DEBUG] First document sample:")
				print(f"[DEBUG]   File: {doc.metadata.get('file_path', 'unknown')}")
				print(f"[DEBUG]   Embedding type: {doc.metadata.get('embedding_type', 'unknown')}")
				print(f"[DEBUG]   Embedding dims: {len(doc.embedding) if doc.embedding else 'None'}")
	
	try:
		# Create vector store
		store = SQLiteVectorStore(db_path, dimension, debug)
		
		# Batch store documents
		success_count = store.batch_store_documents(documents)
		
		# Close the connection when done
		store.close()
		
		if debug:
			print(f"[DEBUG] Saved {success_count}/{len(documents)} documents to SQLite database: {db_path}")
			
		return success_count == len(documents) or success_count > 0
	
	except Exception as e:
		if debug:
			print(f"[DEBUG] SQLite save error: {e}")
			import traceback
			print(traceback.format_exc())
		return False

def clear_project_index(index_dir: str, project: str, debug: bool = False) -> bool:
	"""
	Clear the index for a specific project.
	Compatible with the existing application's clear_index function.
	
	Args:
		index_dir: Base index directory
		project: Project name
		debug: Whether to enable debug output
		
	Returns:
		True if successful, False otherwise
	"""
	db_path = get_db_path(index_dir, project)
	
	if debug:
		print(f"[DEBUG] Database should be at: {db_path}")
	
	# Check if the database exists
	if not os.path.exists(db_path):
		if debug:
			print(f"[DEBUG] Database not found at: {db_path}")
		return True  # Already cleared
	
	# Determine dimension (read from first document or use default)
	dimension = 384
	try:
		conn = sqlite3.connect(db_path)
		cursor = conn.execute("SELECT embedding FROM embeddings LIMIT 1")
		result = cursor.fetchone()
		if result and result[0]:
			# Extract dimension from blob length (4 bytes per float)
			dimension = len(result[0]) // 4
		conn.close()
	except:
		# Use default dimension
		pass
	
	# Create vector store
	store = SQLiteVectorStore(db_path, dimension, debug)
	
	# Clear the project
	deleted_count = store.clear_project(project)
	
	# Close the connection when done
	store.close()
	
	if debug:
		print(f"[DEBUG] Cleared project '{project}' index: deleted {deleted_count} documents")
	
	return True

def discover_projects(index_dir: str) -> List[str]:
	"""
	Discover all projects with SQLite databases.
	Compatible with the existing application's discover_projects function.
	
	Args:
		index_dir: Base index directory
		
	Returns:
		List of project names
	"""
	projects = []
	
	# Check for master database
	master_db = os.path.join(index_dir, "master.db")
	if os.path.exists(master_db):
		projects.append("master")
	
	# Look for project databases
	try:
		for item in os.listdir(index_dir):
			item_path = os.path.join(index_dir, item)
			if os.path.isdir(item_path) and item != "backups":
				# Check if this directory has a database file
				project_db = os.path.join(item_path, f"{item}.db")
				if os.path.exists(project_db):
					projects.append(item)
	except:
		pass
	
	return projects

def search_documents(query_vector: List[float], index_dir: str, project: str, 
					top_k: int = 3, debug: bool = False) -> List[Document]:
	"""
	Search for documents similar to the query vector.
	
	Args:
		query_vector: Query embedding vector
		index_dir: Base index directory
		project: Project to search in, or "master" for all
		top_k: Number of results to return
		debug: Whether to enable debug output
		
	Returns:
		List of Document objects sorted by similarity
	"""
	db_path = get_db_path(index_dir, project)
	
	# Check if the database exists
	if not os.path.exists(db_path):
		if debug:
			print(f"[DEBUG] Database not found at: {db_path}")
		return []
	
	# Determine dimension
	dimension = len(query_vector)
	
	# Create vector store
	store = SQLiteVectorStore(db_path, dimension, debug)
	
	# Search for similar documents
	results = store.search_similar(query_vector, top_k, project)
	
	# Close the connection when done
	store.close()
	
	return results

def search_similar_documents(query_vector: List[float], index_dir: str, project: str, 
						   top_k: int = 3, debug: bool = False, 
						   rag_mode: str = "chunk") -> List[Document]:
	"""
	Search for documents similar to the query vector with support for different RAG modes.
	
	Args:
		query_vector: Query embedding vector
		index_dir: Base index directory
		project: Project to search in
		top_k: Number of results to return
		debug: Whether to enable debug output
		rag_mode: RAG mode ("chunk", "file", or "none")
		
	Returns:
		List of Document objects sorted by similarity
	"""
	if rag_mode.lower() == "none":
		return []
	
	db_path = get_db_path(index_dir, project)
	
	# Check if the database exists
	if not os.path.exists(db_path):
		if debug:
			print(f"[DEBUG] Database not found at: {db_path}")
		return []
	
	# Determine dimension
	dimension = len(query_vector)
	
	# Create vector store
	store = SQLiteVectorStore(db_path, dimension, debug)
	conn = store._ensure_connection()
	
	# Add cosine similarity function
	add_cosine_similarity_function(conn)
	
	try:
		# Different handling based on RAG mode
		if rag_mode.lower() == "file":
			# For file mode: we need to handle this differently
			# First get the top_k most similar documents to get the most relevant files
			top_results = store.search_similar(query_vector, top_k, project)
			
			# Extract the distinct file paths
			distinct_files = set()
			for doc in top_results:
				file_path = doc.metadata.get('file_path', '')
				if file_path:
					distinct_files.add(file_path)
			
			# If we found files, get all chunks from those files
			if distinct_files:
				# Build a query to get all chunks from the top files
				file_placeholders = ','.join(['?'] * len(distinct_files))
				query = f"""
				SELECT m.id, m.file_path, m.file_name, m.project, m.embedding_model, 
					   m.embedding_type, m.chunk_index, m.total_chunks, m.chunk_size, 
					   m.paragraphs, m.last_modified, m.json_metadata, c.content, e.embedding
				FROM metadata m
				JOIN content c ON c.metadata_id = m.id
				LEFT JOIN embeddings e ON e.metadata_id = m.id
				WHERE m.file_path IN ({file_placeholders})
				ORDER BY m.file_path, m.chunk_index
				"""
				
				# Execute the query
				cursor = conn.execute(query, list(distinct_files))
				rows = cursor.fetchall()
				
				# Convert rows to Document objects
				file_chunks = []
				for row in rows:
					(metadata_id, file_path, file_name, doc_project, embedding_model, 
					 embedding_type, chunk_index, total_chunks, chunk_size, 
					 paragraphs, last_modified, json_metadata, content, embedding_blob) = row
					
					# Build metadata dictionary
					metadata = {
						'file_path': file_path,
						'file_name': file_name,
						'project': doc_project,
						'embedding_model': embedding_model,
						'embedding_type': embedding_type,
						'chunk_index': chunk_index,
						'total_chunks': total_chunks,
						'chunk_size': chunk_size,
						'paragraphs': paragraphs,
						'last_modified': last_modified,
					}
					
					# Add any additional metadata from the JSON field
					if json_metadata:
						try:
							additional_metadata = json.loads(json_metadata)
							metadata.update(additional_metadata)
						except:
							pass
					
					# Add a similarity metric for compatibility (approximated)
					# Use the same similarity as the best match for the file
					for top_doc in top_results:
						if top_doc.metadata.get('file_path') == file_path:
							metadata['similarity'] = top_doc.metadata.get('similarity', 0)
							break
					
					# Deserialize the embedding
					embedding = None
					if embedding_blob:
						try:
							embedding = deserialize_f32(embedding_blob, dimension)
						except:
							if debug:
								print(f"[DEBUG] Error deserializing embedding for document {metadata_id}")
					
					# Create Document object
					doc = Document(content=content, metadata=metadata, embedding=embedding)
					file_chunks.append(doc)
				
				if debug:
					print(f"[DEBUG] File mode: found {len(file_chunks)} chunks from {len(distinct_files)} files")
				
				return file_chunks
			
			return top_results  # Fallback if no distinct files were found
		else:
			# Default "chunk" mode: just get top_k chunks
			return store.search_similar(query_vector, top_k, project)
			
	finally:
		# Make sure to close the connection
		store.close()


		
		
		
		

	
	
def verify_database_tables(db_path, debug=False):
	"""Verify that all tables are correctly populated and have the expected structure."""
	try:
		conn = sqlite3.connect(db_path)
		
		# Add cosine similarity function
		add_cosine_similarity_function(conn)
		
		# Check tables and counts
		tables = ['metadata', 'content', 'embeddings', 'embeddings_index']
		counts = {}
		
		for table in tables:
			try:
				cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
				count = cursor.fetchone()[0]
				counts[table] = count
			except sqlite3.OperationalError as e:
				counts[table] = f"Error: {e}"
		
		# Print results
		print(f"Database verification for: {db_path}")
		for table, count in counts.items():
			print(f"  Table {table}: {count}")
		
		# Check if we can query metadata
		try:
			cursor = conn.execute("SELECT DISTINCT project FROM metadata")
			projects = [row[0] for row in cursor.fetchall()]
			print(f"  Projects in database: {', '.join(projects)}")
			
			for project in projects:
				cursor = conn.execute("SELECT COUNT(*) FROM metadata WHERE project = ?", (project,))
				count = cursor.fetchone()[0]
				print(f"    Project '{project}': {count} documents")
		except sqlite3.OperationalError as e:
			print(f"  Error querying projects: {e}")
		
		# Verify vector search functionality
		try:
			# Try to access the vector search function
			conn.execute("SELECT * FROM embeddings_index_search LIMIT 0")
			print("  Vector search function exists and is accessible")
		except sqlite3.OperationalError as e:
			print(f"  Vector search function issue: {e}")
			
			# Try to create the search function
			try:
				conn.execute("""
				CREATE VIRTUAL TABLE embeddings_index_search 
				USING vec0(embeddings_index, embedding, metadata_id);
				""")
				print("  Created vector search function")
			except sqlite3.OperationalError as e:
				print(f"  Could not create vector search function: {e}")
		
		conn.close()
		return counts
	except Exception as e:
		print(f"Error verifying database: {e}")
		import traceback
		traceback.print_exc()
		return {}
		
	
	
	
			
		
		
		

def verify_sqlite_vec_installation():
	"""Verify that the sqlite-vec extension is properly installed and can be loaded."""
	import sqlite3
	
	print("Verifying sqlite-vec installation...")
	conn = sqlite3.connect(":memory:")
	
	try:
		# Enable extension loading
		conn.enable_load_extension(True)
		
		# Try different ways to load the extension
		extension_paths = [
			"sqlite_vec",                        # Default (depends on LD_LIBRARY_PATH)
			"/opt/homebrew/lib/sqlite-vec",      # MacOS Homebrew
			"/usr/local/lib/sqlite-vec",         # Common Unix path
			"/usr/lib/sqlite3/sqlite-vec",       # Linux
			"sqlite_vec.so"                      # Direct .so file
		]
		
		loaded = False
		error_messages = []
		
		# First try the Python API if available
		try:
			import sqlite_vec
			print("Found sqlite_vec Python package, trying to load...")
			sqlite_vec.load(conn)
			loaded = True
			print("Successfully loaded via Python API")
		except (ImportError, Exception) as e:
			error_messages.append(f"Python API load failed: {e}")
			print(f"Python API load failed: {e}")
			
			# Try direct extension loading
			for path in extension_paths:
				try:
					print(f"Trying to load extension from: {path}")
					conn.load_extension(path)
					loaded = True
					print(f"Successfully loaded from {path}")
					break
				except sqlite3.OperationalError as e:
					error_messages.append(f"Failed to load from {path}: {e}")
					print(f"  Failed: {e}")
		
		# Verify it's working
		if loaded:
			try:
				version = conn.execute("SELECT vec_version()").fetchone()[0]
				print(f"SQLite vector extension loaded successfully. Version: {version}")
				
				# Test vector functionality using struct packing
				try:
					print("Testing vector table creation with vec0...")
					# Create a simple test table
					conn.execute("CREATE VIRTUAL TABLE test_vec USING vec0(embedding float[4])")
					
					# Add custom function for cosine similarity
					add_cosine_similarity_function(conn)
					
					# Create a sample vector and serialize it
					sample_vec = [0.1, 0.2, 0.3, 0.4]
					serialized_vec = serialize_f32(sample_vec)
					
					# Insert into the table
					conn.execute("INSERT INTO test_vec (rowid, embedding) VALUES (1, ?)", (serialized_vec,))
					print("Successfully inserted a vector")
					
					# Test retrieval
					result = conn.execute("SELECT embedding FROM test_vec").fetchone()
					if result:
						retrieved_vec = deserialize_f32(result[0], 4)
						print(f"Retrieved vector: {retrieved_vec}")
						
						# Test cosine similarity
						query_vec = [0.2, 0.3, 0.4, 0.5] 
						query_serialized = serialize_f32(query_vec)
						sim_result = conn.execute("SELECT cosine_similarity(embedding, ?) FROM test_vec", 
												(query_serialized,)).fetchone()
						print(f"Cosine similarity test result: {sim_result[0]}")
					
					conn.execute("DROP TABLE test_vec")
					print("Successfully created, used, and dropped a vector table")
					return True
				except sqlite3.OperationalError as e:
					print(f"Error testing vector functionality with vec0: {e}")
					# If vec0 didn't work, we can't use vector search
					return False
			except sqlite3.OperationalError as e:
				print(f"Error testing vector functionality: {e}")
				return False
		else:
			print("Could not load sqlite-vec extension.")
			print("\nPossible solutions:")
			print("1. Install sqlite-vec: brew install sqlite-vec (MacOS) or follow installation instructions")
			print("2. Make sure the extension is in your library path")
			print("3. Install the Python package: pip install sqlite-vec")
			print("\nError details:")
			for msg in error_messages:
				print(f"- {msg}")
			return False
			
	except Exception as e:
		print(f"Error during verification: {e}")
		import traceback
		traceback.print_exc()
		return False
	finally:
		conn.close()