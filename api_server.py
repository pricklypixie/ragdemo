#!/usr/bin/env python3
"""
RAG Query API Server

Provides a REST API for the RAG Query application, making it accessible
through a web interface or other HTTP clients.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import traceback
import asyncio

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Import from shared functions
from shared_functions import (
	# Constants
	DEFAULT_INDEX_DIR, DEFAULT_DOCUMENT_DIR, DEFAULT_EMBEDDING_MODEL, 
	DEFAULT_EMBEDDING_TYPE, TOP_K_DOCUMENTS, MASTER_PROJECT, PROMPTS_DIR,
	DEFAULT_CHARS_PER_DIMENSION, LLM_CLAUDE, LLM_LOCAL, LLM_HF, LLM_OPENAI,
	DEFAULT_LLM_TYPE, DEFAULT_LOCAL_MODEL, DEFAULT_HF_MODEL, DEFAULT_CLAUDE_MODEL,
	DEFAULT_OPENAI_MODEL,
	
	# Classes
	Document, EmbeddingProviderCache,
	
	# Functions
	get_project_config_path, load_project_config_file, get_project_embedding_config,
	load_index, discover_projects, clear_index, index_project, get_index_path,
	save_embedding_config, is_command, get_model_name_for_llm_type,
	search_documents, get_response, display_search_results
)

# Import our embedding library
from embeddings import EmbeddingConfig






# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_api")

# Create FastAPI app
app = FastAPI(
	title="RAG Query API",
	description="API for querying documents using Retrieval Augmented Generation",
	version="1.0.0"
)

# Initialize templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create embedding provider cache
provider_cache = EmbeddingProviderCache(debug=False)

# Create a global dictionary to store project documents
project_documents = {}


# check if sqlite is available
SQLITE_AVAILABLE = False
try:
	from sqlite_storage import discover_projects as sqlite_discover_projects, document_count
	SQLITE_AVAILABLE = True
except ImportError:
	logger.warning("SQLite storage module not available")




# Request/Response models for API
class QueryRequest(BaseModel):
	query: str
	project: str = MASTER_PROJECT
	llm_type: Optional[str] = None
	model_name: Optional[str] = None
	rag_mode: Optional[str] = None
	rag_count: Optional[int] = None
	system_prompt: Optional[str] = None

class Document(BaseModel):
	content: str
	metadata: Dict[str, Any]
	similarity: Optional[float] = None

class QueryResponse(BaseModel):
	query: str
	project: str
	documents: List[Document]
	answer: str
	llm_type: str
	model_name: str
	timestamp: str
	elapsed_time: float

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
	"""Serve the main query interface HTML page."""
	# Check if index.html exists in templates directory, create if not
	index_path = templates_dir / "index.html"
	if not index_path.exists():
		with open(index_path, "w") as f:
			f.write(DEFAULT_INDEX_HTML)
	
	# Get list of available projects - try both methods
	sqlite_projects = []
	pickle_projects = []
	
	try:
		# First try SQLite
		sqlite_projects = discover_projects(DEFAULT_DOCUMENT_DIR, DEFAULT_INDEX_DIR, use_sqlite=True)
	except Exception as e:
		logger.warning(f"Error discovering SQLite projects: {e}")
	
	try:
		# Then try pickle files
		pickle_projects = discover_projects(DEFAULT_DOCUMENT_DIR, DEFAULT_INDEX_DIR, use_sqlite=False)
	except Exception as e:
		logger.warning(f"Error discovering pickle projects: {e}")
	
	# Combine unique projects from both methods
	all_projects = list(set(sqlite_projects + pickle_projects))
	
	# Sort projects (with "master" always first if present)
	if "master" in all_projects:
		all_projects.remove("master")
		all_projects = ["master"] + sorted(all_projects)
	else:
		all_projects = sorted(all_projects)
	
	return templates.TemplateResponse(
		"index.html", 
		{"request": request, "projects": all_projects}
	)



@app.get("/document/{project}/{filepath:path}")
async def get_document(project: str, filepath: str):
	"""Serve the content of a document."""
	try:
		# Ensure the path is safe (prevent directory traversal attacks)
		clean_filepath = filepath.lstrip("/")
		
		# Check if the filepath already includes the project name
		path_parts = clean_filepath.split('/')
		if path_parts and path_parts[0] == project:
			# The project name is already in the path, so don't add it again
			file_path_with_project = clean_filepath
		else:
			# Need to add the project name to the path
			file_path_with_project = clean_filepath
		
		# Determine the full file path
		if project == MASTER_PROJECT:
			full_path = os.path.join(DEFAULT_DOCUMENT_DIR, clean_filepath)
		else:
			# This is the key fix: check if the path already starts with the project name
			if clean_filepath.startswith(f"{project}/"):
				# Path already includes project, don't add it again
				full_path = os.path.join(DEFAULT_DOCUMENT_DIR, clean_filepath)
			else:
				# Path doesn't include project yet, add it
				full_path = os.path.join(DEFAULT_DOCUMENT_DIR, project, clean_filepath)
			
		logger.debug(f"Serving document: {full_path}")
			
		# Check if file exists and is within the document directory
		abs_path = os.path.abspath(full_path)
		abs_doc_dir = os.path.abspath(DEFAULT_DOCUMENT_DIR)
		
		# Security check to prevent directory traversal
		if not abs_path.startswith(abs_doc_dir):
			logger.error(f"Security violation: Path {abs_path} is outside document directory {abs_doc_dir}")
			raise HTTPException(status_code=403, detail="Access denied")
			
		# Check if file exists
		if not os.path.exists(abs_path):
			logger.error(f"Document not found: {abs_path}")
			raise HTTPException(status_code=404, detail="Document not found")
			
		# Read the file content
		with open(abs_path, 'r', encoding='utf-8') as f:
			content = f.read()
			
		# Return content with content-type based on file extension
		if abs_path.endswith('.md'):
			return HTMLResponse(marked_markdown_to_html(content))
		else:
			return HTMLResponse(f"<pre>{content}</pre>")
	except HTTPException:
		# Re-raise HTTP exceptions
		raise
	except Exception as e:
		logger.error(f"Error serving document: {str(e)}")
		raise HTTPException(status_code=500, detail=f"Error serving document: {str(e)}")



# @app.get("/document/{project}/{filepath:path}")
# 	async def get_document(project: str, filepath: str):
# 		"""Serve the content of a document."""
# 		try:
# 			# Ensure the path is safe (prevent directory traversal attacks)
# 			clean_filepath = filepath.lstrip("/")
# 			
# 			# Determine the full file path
# 			if project == MASTER_PROJECT:
# 				full_path = os.path.join(DEFAULT_DOCUMENT_DIR, clean_filepath)
# 			else:
# 				# Check if the file path already includes the project name as the first directory
# 				path_parts = clean_filepath.split('/')
# 				if path_parts and path_parts[0] == project:
# 					# The project name is already in the path, don't add it again
# 					full_path = os.path.join(DEFAULT_DOCUMENT_DIR, clean_filepath)
# 				else:
# 					# Add the project name to the path
# 					full_path = os.path.join(DEFAULT_DOCUMENT_DIR, project, clean_filepath)
# 				
# 			logger.debug(f"Serving document: {full_path}")
# 				
# 			# Check if file exists and is within the document directory
# 			abs_path = os.path.abspath(full_path)
# 			abs_doc_dir = os.path.abspath(DEFAULT_DOCUMENT_DIR)
# 			
# 			# Security check to prevent directory traversal
# 			if not abs_path.startswith(abs_doc_dir):
# 				logger.error(f"Security violation: Path {abs_path} is outside document directory {abs_doc_dir}")
# 				raise HTTPException(status_code=403, detail="Access denied")
# 				
# 			# Check if file exists
# 			if not os.path.exists(abs_path):
# 				logger.error(f"Document not found: {abs_path}")
# 				raise HTTPException(status_code=404, detail="Document not found")
# 				
# 			# Read the file content
# 			with open(abs_path, 'r', encoding='utf-8') as f:
# 				content = f.read()
# 				
# 			# Return content with content-type based on file extension
# 			if abs_path.endswith('.md'):
# 				return HTMLResponse(marked_markdown_to_html(content))
# 			else:
# 				return HTMLResponse(f"<pre>{content}</pre>")
# 		except HTTPException:
# 			# Re-raise HTTP exceptions
# 			raise
# 		except Exception as e:
# 			logger.error(f"Error serving document: {str(e)}")
# 			raise HTTPException(status_code=500, detail=f"Error serving document: {str(e)}")
			
			
			
			
			

def marked_markdown_to_html(markdown_text: str) -> str:
	"""Convert markdown to HTML with proper styling."""
	# Escape backticks to prevent JavaScript injection
	safe_markdown = markdown_text.replace('`', '\\`')
	
	return f"""
	<!DOCTYPE html>
	<html>
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>Document View</title>
		<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
		<style>
			body {{ 
				font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
				line-height: 1.6;
				color: #333;
				max-width: 800px;
				margin: 0 auto;
				padding: 20px;
			}}
			pre {{ 
				background-color: #f6f8fa; 
				padding: 16px;
				border-radius: 6px;
				overflow-x: auto;
			}}
			code {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; }}
			table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
			th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
			th {{ background-color: #f6f8fa; }}
			img {{ max-width: 100%; }}
			h1, h2, h3, h4, h5, h6 {{ margin-top: 24px; margin-bottom: 16px; }}
			blockquote {{ padding: 0 1em; color: #6a737d; border-left: 0.25em solid #dfe2e5; }}
		</style>
	</head>
	<body>
		<div id="content"></div>
		<script>
			document.getElementById('content').innerHTML = marked.parse(`{safe_markdown}`);
		</script>
	</body>
	</html>
	"""

@app.get("/projects")
async def list_projects():
	"""List all available projects."""
	try:
		# Get projects from both SQLite and pickle files
		sqlite_projects = []
		pickle_projects = []
		
		try:
			sqlite_projects = discover_projects(DEFAULT_DOCUMENT_DIR, DEFAULT_INDEX_DIR, use_sqlite=True)
		except Exception as e:
			logger.warning(f"Error discovering SQLite projects: {e}")
		
		try:
			pickle_projects = discover_projects(DEFAULT_DOCUMENT_DIR, DEFAULT_INDEX_DIR, use_sqlite=False)
		except Exception as e:
			logger.warning(f"Error discovering pickle projects: {e}")
		
		# Combine unique projects from both methods
		all_projects = list(set(sqlite_projects + pickle_projects))
		
		# Sort projects (with "master" always first if present)
		if "master" in all_projects:
			all_projects.remove("master")
			all_projects = ["master"] + sorted(all_projects)
		else:
			all_projects = sorted(all_projects)
		
		project_info = []
		
		for project in all_projects:
			# Get project configuration
			config = load_project_config_file(project, DEFAULT_DOCUMENT_DIR)
			
			# Extract embedding config
			indexing_config = config.get("indexing", {})
			embedding_type = indexing_config.get("embedding_type", DEFAULT_EMBEDDING_TYPE)
			embedding_model = indexing_config.get("model_name", DEFAULT_EMBEDDING_MODEL)
			
			# Extract RAG settings
			rag_settings = config.get("rag", {})
			llm_type = rag_settings.get("llm_type", DEFAULT_LLM_TYPE)
			llm_model = rag_settings.get("llm_model", get_model_name_for_llm_type(llm_type))
			rag_mode = rag_settings.get("rag_mode", "chunk")
			rag_count = rag_settings.get("rag_count", TOP_K_DOCUMENTS)
			
			# Get document count - try both SQLite and pickle
			doc_count = 0
			try:
				# Try SQLite first
				from sqlite_storage import document_count
				sqlite_count = document_count(DEFAULT_INDEX_DIR, project)
				doc_count += sqlite_count
			except Exception:
				pass
				
			try:
				# Then try pickle
				index_path, _ = get_index_path(DEFAULT_INDEX_DIR, project)
				if os.path.exists(index_path):
					with open(index_path, 'rb') as f:
						pickle_docs = pickle.load(f)
					doc_count += len(pickle_docs)
			except Exception:
				pass
			
			# Add to project info list
			project_info.append({
				"name": project,
				"documents": doc_count,
				"embedding": {
					"type": embedding_type,
					"model": embedding_model
				},
				"rag": {
					"llm_type": llm_type,
					"llm_model": llm_model,
					"mode": rag_mode,
					"count": rag_count
				}
			})
		
		return {"projects": project_info}
	
	except Exception as e:
		logger.error(f"Error listing projects: {str(e)}")
		logger.error(traceback.format_exc())
		raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")
			
			
			
			
			
			

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest = Body(...)):
	"""Process a RAG query and return results."""
	start_time = time.time()
	query_text = request.query.strip()
	project = request.project or MASTER_PROJECT
	
	if not query_text:
		raise HTTPException(status_code=400, detail="Query cannot be empty")
	
	logger.info(f"Received query request: project='{project}', query='{query_text}'")
	
	try:
		# Load project configuration
		project_config = load_project_config_file(project, DEFAULT_DOCUMENT_DIR)
		
		# Get embedding configuration
		embedding_config = get_project_embedding_config(project, DEFAULT_DOCUMENT_DIR)
		
		# Get RAG settings from project config, overridden by request params
		rag_settings = project_config.get("rag", {})
		llm_type = request.llm_type or rag_settings.get("llm_type", DEFAULT_LLM_TYPE)
		
		# Get model based on the LLM type
		if llm_type == LLM_CLAUDE:
			model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_CLAUDE_MODEL)
		elif llm_type == LLM_OPENAI:
			model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_OPENAI_MODEL)
		elif llm_type == LLM_LOCAL:
			model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_LOCAL_MODEL)
		elif llm_type == LLM_HF:
			model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_HF_MODEL)
		else:
			model_name = request.model_name or rag_settings.get("llm_model", get_model_name_for_llm_type(llm_type))
		
		rag_mode = request.rag_mode or rag_settings.get("rag_mode", "chunk")
		rag_count = request.rag_count or rag_settings.get("rag_count", TOP_K_DOCUMENTS)
		
		# Get system prompt from request or config
		system_prompt = request.system_prompt
		if not system_prompt:
			system_prompt = rag_settings.get("system_prompt")
			# If not in rag settings, check system section for model
			if not system_prompt:
				system_settings = project_config.get("system", {})
				if model_name in system_settings:
					system_prompt = system_settings[model_name].get("system_prompt")
		
		# Get API key
		api_key = None
		if llm_type == LLM_CLAUDE:
			api_key = os.environ.get("ANTHROPIC_API_KEY")
		elif llm_type == LLM_OPENAI:
			api_key = os.environ.get("OPENAI_API_KEY")
		
		# Load or get documents
		try:
			# Try SQLite first
			documents = search_documents(
				query_text, [], project, DEFAULT_DOCUMENT_DIR,
				embedding_config, rag_count, False, provider_cache,
				rag_mode, True, DEFAULT_INDEX_DIR
			)
		except:
			# Fall back to pickle files
			index_path, backup_dir = get_index_path(DEFAULT_INDEX_DIR, project)
			
			# Check if we already have loaded these documents
			if project not in project_documents:
				documents = load_index(index_path, backup_dir, False)
				project_documents[project] = documents
			else:
				documents = project_documents[project]
			
			# Search documents
			documents = search_documents(
				query_text, documents, project, DEFAULT_DOCUMENT_DIR,
				embedding_config, rag_count, False, provider_cache,
				rag_mode, False
			)
		
		# Convert Document objects to Pydantic schema
		pydantic_docs = []
		for doc in documents:
			# Extract similarity score
			similarity = doc.metadata.get('similarity', 0)
			
			# Create Pydantic document
			pydantic_docs.append(Document(
				content=doc.content,
				metadata=doc.metadata,
				similarity=similarity
			))
		
		# Get response from LLM
		answer = get_response(
			query_text, documents, api_key, project,
			llm_type, model_name, False, PROMPTS_DIR,
			rag_mode, DEFAULT_DOCUMENT_DIR, system_prompt
		)
		
		elapsed_time = time.time() - start_time
		logger.info(f"Query completed in {elapsed_time:.2f}s")
		
		# Prepare and return response
		response = QueryResponse(
			query=query_text,
			project=project,
			documents=pydantic_docs,
			answer=answer,
			llm_type=llm_type,
			model_name=model_name,
			timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
			elapsed_time=elapsed_time
		)
		
		return response
	
	except Exception as e:
		logger.error(f"Error processing query: {str(e)}")
		logger.error(traceback.format_exc())
		raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/stream_query")
async def stream_query(request: QueryRequest = Body(...)):
	"""Process a RAG query and stream the results, first documents then answer."""
	query_text = request.query.strip()
	project = request.project or MASTER_PROJECT
	
	if not query_text:
		raise HTTPException(status_code=400, detail="Query cannot be empty")
	
	logger.info(f"Received streaming query request: project='{project}', query='{query_text}'")
	
	# Create async generator to stream the response
	async def generate_stream():
		start_time = time.time()
		try:
			# Load project configuration
			project_config = load_project_config_file(project, DEFAULT_DOCUMENT_DIR)
			
			# Get embedding configuration
			embedding_config = get_project_embedding_config(project, DEFAULT_DOCUMENT_DIR)
			
			# Get RAG settings
			rag_settings = project_config.get("rag", {})
			llm_type = request.llm_type or rag_settings.get("llm_type", DEFAULT_LLM_TYPE)
			
			# Get model based on LLM type
			if llm_type == LLM_CLAUDE:
				model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_CLAUDE_MODEL)
			elif llm_type == LLM_OPENAI:
				model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_OPENAI_MODEL)
			elif llm_type == LLM_LOCAL:
				model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_LOCAL_MODEL)
			elif llm_type == LLM_HF:
				model_name = request.model_name or rag_settings.get("llm_model", DEFAULT_HF_MODEL)
			else:
				model_name = request.model_name or rag_settings.get("llm_model", get_model_name_for_llm_type(llm_type))
			
			rag_mode = request.rag_mode or rag_settings.get("rag_mode", "chunk")
			rag_count = request.rag_count or rag_settings.get("rag_count", TOP_K_DOCUMENTS)
			
			# Get system prompt
			system_prompt = request.system_prompt
			if not system_prompt:
				system_prompt = rag_settings.get("system_prompt")
				if not system_prompt:
					system_settings = project_config.get("system", {})
					if model_name in system_settings:
						system_prompt = system_settings[model_name].get("system_prompt")
			
			# Get API key
			api_key = None
			if llm_type == LLM_CLAUDE:
				api_key = os.environ.get("ANTHROPIC_API_KEY")
			elif llm_type == LLM_OPENAI:
				api_key = os.environ.get("OPENAI_API_KEY")
			
			# First phase: Search documents
			documents = []
			
			# Stream event indicating document search has started
			yield json.dumps({"event": "search_started"}) + "\n"
			
			try:
				# Try SQLite first
				documents = search_documents(
					query_text, [], project, DEFAULT_DOCUMENT_DIR,
					embedding_config, rag_count, False, provider_cache,
					rag_mode, True, DEFAULT_INDEX_DIR
				)
			except:
				# Fall back to pickle files
				index_path, backup_dir = get_index_path(DEFAULT_INDEX_DIR, project)
				
				if project not in project_documents:
					documents = load_index(index_path, backup_dir, False)
					project_documents[project] = documents
				else:
					documents = project_documents[project]
				
				documents = search_documents(
					query_text, documents, project, DEFAULT_DOCUMENT_DIR,
					embedding_config, rag_count, False, provider_cache,
					rag_mode, False
				)
			
			# Convert documents to serializable format with similarity
			doc_results = []
			for doc in documents:
				similarity = doc.metadata.get('similarity', 0)
				doc_results.append({
					"content": doc.content,
					"metadata": doc.metadata,
					"similarity": similarity
				})
			
			# Stream the document results
			yield json.dumps({"event": "documents", "data": doc_results}) + "\n"
			
			# Small delay to ensure frontend processes the documents
			await asyncio.sleep(0.1)
			
			# Second phase: Get LLM response
			yield json.dumps({"event": "llm_started"}) + "\n"
			
			# Get response from LLM
			answer = get_response(
				query_text, documents, api_key, project,
				llm_type, model_name, False, PROMPTS_DIR,
				rag_mode, DEFAULT_DOCUMENT_DIR, system_prompt
			)
			
			# Stream the answer
			elapsed_time = time.time() - start_time
			yield json.dumps({
				"event": "answer", 
				"data": {
					"answer": answer,
					"llm_type": llm_type,
					"model_name": model_name,
					"elapsed_time": elapsed_time
				}
			}) + "\n"
			
			# Final event to signal completion
			yield json.dumps({"event": "complete"}) + "\n"
			
		except Exception as e:
			logger.error(f"Error in streaming query: {str(e)}")
			logger.error(traceback.format_exc())
			# Stream error event
			yield json.dumps({"event": "error", "message": str(e)}) + "\n"
	
	return StreamingResponse(
		generate_stream(),
		media_type="text/event-stream"
	)

@app.post("/index")
async def index_project_api(project: str = Query(default=MASTER_PROJECT)):
	"""Start indexing the specified project."""
	try:
		logger.info(f"Starting indexing of project: {project}")
		
		success = index_project(
			project, 
			DEFAULT_DOCUMENT_DIR, 
			DEFAULT_INDEX_DIR, 
			debug=False,
			auto_adjust_chunks=True,
			chars_per_dimension=DEFAULT_CHARS_PER_DIMENSION,
			use_sqlite=True
		)
		
		if success:
			# Clear cached documents for this project
			if project in project_documents:
				del project_documents[project]
			
			# Get document count
			try:
				from sqlite_storage import document_count
				doc_count = document_count(DEFAULT_INDEX_DIR, project)
			except:
				doc_count = 0
				try:
					index_path, _ = get_index_path(DEFAULT_INDEX_DIR, project)
					with open(index_path, 'rb') as f:
						documents = json.load(f)
					doc_count = len(documents)
				except:
					pass
			
			return {"success": True, "message": f"Project '{project}' indexed successfully", "document_count": doc_count}
		else:
			raise HTTPException(status_code=500, detail=f"Failed to index project '{project}'")
	
	except Exception as e:
		logger.error(f"Error indexing project {project}: {str(e)}")
		logger.error(traceback.format_exc())
		raise HTTPException(status_code=500, detail=f"Error indexing project: {str(e)}")
		
		

@app.get("/status")
async def system_status():
	"""Return system status information."""
	status = {
		"status": "ok",
		"sqlite_available": SQLITE_AVAILABLE,
		"document_dir": os.path.abspath(DEFAULT_DOCUMENT_DIR),
		"index_dir": os.path.abspath(DEFAULT_INDEX_DIR),
		"api_keys": {
			"anthropic": "available" if os.environ.get("ANTHROPIC_API_KEY") else "missing",
			"openai": "available" if os.environ.get("OPENAI_API_KEY") else "missing"
		}
	}
	
	# Add project count
	try:
		all_projects = []
		if SQLITE_AVAILABLE:
			sqlite_projects = sqlite_discover_projects(DEFAULT_INDEX_DIR)
			all_projects.extend(sqlite_projects)
		
		pickle_projects = discover_projects(DEFAULT_DOCUMENT_DIR, DEFAULT_INDEX_DIR, use_sqlite=False)
		all_projects.extend(pickle_projects)
		all_projects = list(set(all_projects))
		
		status["projects"] = len(all_projects)
	except Exception as e:
		status["projects"] = "error: " + str(e)
	
	return status



# Default HTML template for index.html
DEFAULT_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>RAG Query Interface</title>
	<style>
		body {
			font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
			line-height: 1.6;
			color: #333;
			max-width: 1200px;
			margin: 0 auto;
			padding: 20px;
			background-color: #f9f9f9;
		}
		header {
			background-color: #2c3e50;
			color: white;
			padding: 10px 20px;
			border-radius: 5px;
			margin-bottom: 20px;
			display: flex;
			justify-content: space-between;
			align-items: center;
		}
		h1 {
			margin: 0;
			font-size: 24px;
		}
		.container {
			background-color: white;
			border-radius: 5px;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
			padding: 20px;
			margin-bottom: 20px;
		}
		.form-group {
			margin-bottom: 15px;
		}
		label {
			display: block;
			margin-bottom: 5px;
			font-weight: bold;
		}
		select, textarea {
			width: 100%;
			padding: 10px;
			border: 1px solid #ddd;
			border-radius: 4px;
			box-sizing: border-box;
			font-family: inherit;
			font-size: 16px;
		}
		textarea {
			min-height: 100px;
			resize: vertical;
		}
		button {
			background-color: #3498db;
			color: white;
			border: none;
			padding: 10px 15px;
			border-radius: 4px;
			cursor: pointer;
			font-size: 16px;
			transition: background-color 0.3s;
		}
		button:hover {
			background-color: #2980b9;
		}
		.button-row {
			display: flex;
			justify-content: space-between;
			align-items: center;
		}
		#index-button {
			background-color: #27ae60;
			margin-left: 10px;
		}
		#index-button:hover {
			background-color: #219653;
		}
		.settings-toggle {
			background-color: #7f8c8d;
			margin-left: auto;
			margin-right: 10px;
		}
		.settings-toggle:hover {
			background-color: #6c7a7d;
		}
		.advanced-settings {
			display: none;
			background-color: #f5f5f5;
			border: 1px solid #ddd;
			border-radius: 4px;
			padding: 15px;
			margin-bottom: 15px;
		}
		.response {
			margin-top: 30px;
		}
		.sources {
			background-color: #f8f9fa;
			border-left: 3px solid #3498db;
			padding: 10px 15px;
			margin-bottom: 20px;
			font-size: 14px;
		}
		.source-item {
			margin-bottom: 10px;
			padding-bottom: 10px;
			border-bottom: 1px solid #eee;
		}
		.source-item:last-child {
			border-bottom: none;
			margin-bottom: 0;
			padding-bottom: 0;
		}
		.answer {
			padding: 20px;
			background-color: #e8f4fc;
			border-radius: 5px;
			margin-bottom: 20px;
			line-height: 1.7;
		}
		.loading {
			display: none;
			text-align: center;
			padding: 20px;
			font-style: italic;
			color: #666;
		}
		.loading:after {
			content: "";
			animation: dots 1.5s infinite;
		}
		@keyframes dots {
			0%, 20% { content: "."; }
			40% { content: ".."; }
			60%, 100% { content: "..."; }
		}
		.source-header {
			font-weight: bold;
			margin-bottom: 5px;
			display: flex;
			justify-content: space-between;
		}
		.source-content {
			margin-left: 10px;
			border-left: 2px solid #ddd;
			padding-left: 10px;
			white-space: pre-line;
			overflow-wrap: break-word;
			max-height: 100px;
			overflow-y: auto;
		}
		.expand-button {
			background-color: #eee;
			border: none;
			padding: 2px 5px;
			font-size: 12px;
			cursor: pointer;
			border-radius: 3px;
		}
		.expanded .source-content {
			max-height: none;
		}
		.similarity-score {
			display: inline-block;
			padding: 2px 5px;
			background-color: #e8f4fc;
			border-radius: 3px;
			font-size: 12px;
			margin-left: 10px;
		}
		.info-bar {
			margin-top: 20px;
			font-size: 14px;
			color: #666;
		}
		.project-info {
			display: flex;
			align-items: center;
		}
		.project-badge {
			background-color: #3498db;
			color: white;
			padding: 5px 10px;
			border-radius: 3px;
			font-size: 12px;
			margin-left: 10px;
		}
		a {
			color: #3498db;
			text-decoration: none;
		}
		a:hover {
			text-decoration: underline;
		}
		.answer img {
			max-width: 100%;
			height: auto;
			margin: 10px 0;
			border-radius: 5px;
		}
		pre {
			background-color: #f8f9fa;
			padding: 10px;
			border-radius: 4px;
			overflow-x: auto;
			font-family: 'Courier New', Courier, monospace;
		}
		code {
			background-color: #f0f0f0;
			padding: 2px 4px;
			border-radius: 3px;
			font-family: 'Courier New', Courier, monospace;
		}
		.answer table {
			border-collapse: collapse;
			width: 100%;
			margin: 10px 0;
		}
		.answer th, .answer td {
			border: 1px solid #ddd;
			padding: 8px;
			text-align: left;
		}
		.answer th {
			background-color: #f2f2f2;
		}
		.answer tr:nth-child(even) {
			background-color: #f9f9f9;
		}
		.doc-link {
			display: inline-flex;
			align-items: center;
			margin-left: 5px;
			color: #3498db;
			transition: color 0.2s;
		}
		.doc-link:hover {
			color: #2980b9;
		}
	</style>
</head>
<body>
	<header>
		<h1>RAG Query Interface</h1>
		<div class="project-info">
			<span>Project:</span>
			<span class="project-badge" id="current-project">{{ projects[0] if projects else "master" }}</span>
		</div>
	</header>
	
	<div class="container">
		<form id="query-form">
			<div class="form-group">
				<label for="project-select">Select Project:</label>
				<select id="project-select" name="project">
					{% for project in projects %}
					<option value="{{ project }}">{{ project }}</option>
					{% endfor %}
				</select>
			</div>
			
			<button type="button" class="settings-toggle" id="settings-toggle">Advanced Settings</button>
			
			<div class="advanced-settings" id="advanced-settings">
				<div class="form-group">
					<label for="llm-type">LLM Type:</label>
					<select id="llm-type" name="llm_type">
						<option value="">Use Project Default</option>
						<option value="claude">Claude</option>
						<option value="openai">OpenAI</option>
						<option value="local">Local</option>
						<option value="hf">Hugging Face</option>
					</select>
				</div>
				
				<div class="form-group">
					<label for="model-name">Model Name:</label>
					<select id="model-name" name="model_name">
						<option value="">Use Project Default</option>
						<optgroup label="Claude Models">
							<option value="claude-3-5-haiku-20241022">claude-3-5-haiku-20241022</option>
							<option value="claude-3-sonnet-20240229">claude-3-sonnet-20240229</option>
							<option value="claude-3-opus-20240229">claude-3-opus-20240229</option>
						</optgroup>
						<optgroup label="OpenAI Models">
							<option value="gpt-4o-mini">gpt-4o-mini</option>
							<option value="gpt-4o">gpt-4o</option>
							<option value="gpt-4">gpt-4</option>
							<option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
						</optgroup>
						<optgroup label="Local Models">
							<option value="mistral-7b-instruct-v0">mistral-7b-instruct-v0</option>
							<option value="orca-2-7b">orca-2-7b">orca-2-7b">orca-2-7b</option>
							<option value="orca-2-13b">orca-2-13b">orca-2-13b">orca-2-13b</option>
							<option value="DeepSeek-R1-Distill-Qwen-14B-Q4_0">DeepSeek-R1-Distill-Qwen-14B-Q4_0</option>
							<option value="mlx-community/gemma-3-4b-it-bf16">mlx-community/gemma-3-4b-it-bf16</option>
							<option value="mlx-community/OLMo-2-0325-32B-Instruct-4bit">OLMo-2-0325-32B-Instruct-4bit (requires llm-mlx)</option>
						</optgroup>
						<optgroup label="Hugging Face Models">
							<option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama (1.1B)</option>
							<option value="mistralai/Mistral-7B-v0.1">Mistral (7B)</option>
						</optgroup>
					</select>
				</div>
				
				<div class="form-group">
					<label for="rag-mode">RAG Mode:</label>
					<select id="rag-mode" name="rag_mode">
						<option value="">Use Project Default</option>
						<option value="chunk">Chunk</option>
						<option value="file">File</option>
						<option value="none">None</option>
					</select>
				</div>
				
				<div class="form-group">
					<label for="rag-count">Document Count:</label>
					<select id="rag-count" name="rag_count">
						<option value="">Use Project Default</option>
						<option value="1">1</option>
						<option value="2">2</option>
						<option value="3">3</option>
						<option value="5">5</option>
						<option value="10">10</option>
					</select>
				</div>
				
				<div class="form-group">
					<label for="system-prompt">System Prompt:</label>
					<textarea id="system-prompt" name="system_prompt" placeholder="Leave empty to use project default"></textarea>
				</div>
			</div>
			
			<div class="form-group">
				<label for="query-input">Your Question:</label>
				<textarea id="query-input" name="query" placeholder="Enter your question here..."></textarea>
			</div>
			
			<div class="button-row">
				<button type="submit" id="submit-button">Send Query</button>
				<button type="button" id="index-button">Re-Index Project</button>
			</div>
		</form>
	</div>
	
	<div class="loading" id="loading">Processing your query</div>
	
	<div class="response" id="response">
		<!-- Response will be inserted here -->
	</div>

	<script>
		document.addEventListener('DOMContentLoaded', function() {
			const queryForm = document.getElementById('query-form');
			const projectSelect = document.getElementById('project-select');
			const settingsToggle = document.getElementById('settings-toggle');
			const advancedSettings = document.getElementById('advanced-settings');
			const indexButton = document.getElementById('index-button');
			const loading = document.getElementById('loading');
			const response = document.getElementById('response');
			const currentProjectBadge = document.getElementById('current-project');
			
			// Update current project badge when project is selected
			projectSelect.addEventListener('change', function() {
				currentProjectBadge.textContent = this.value;
			});
			
			// Toggle advanced settings
			settingsToggle.addEventListener('click', function() {
				if (advancedSettings.style.display === 'block') {
					advancedSettings.style.display = 'none';
					settingsToggle.textContent = 'Advanced Settings';
				} else {
					advancedSettings.style.display = 'block';
					settingsToggle.textContent = 'Hide Advanced Settings';
				}
			});
			
			// Handle form submission with streaming
			queryForm.addEventListener('submit', async function(e) {
				e.preventDefault();
				
				const formData = new FormData(queryForm);
				const query = formData.get('query').trim();
				
				if (!query) {
					alert('Please enter a query');
					return;
				}
				
				// Prepare request data
				const requestData = {
					query: query,
					project: formData.get('project')
				};
				
				// Add optional parameters if set
				if (formData.get('llm_type')) requestData.llm_type = formData.get('llm_type');
				if (formData.get('model_name')) requestData.model_name = formData.get('model_name');
				if (formData.get('rag_mode')) requestData.rag_mode = formData.get('rag_mode');
				if (formData.get('rag_count')) requestData.rag_count = parseInt(formData.get('rag_count'));
				if (formData.get('system_prompt')) requestData.system_prompt = formData.get('system_prompt');
				
				// Show loading indicator
				loading.style.display = 'block';
				loading.textContent = 'Searching for relevant documents...';
				response.innerHTML = '';
				
				try {
					// Set up the response structure in advance
					response.innerHTML = `
						<div id="documents-section" style="display:none;">
							<h2>Sources</h2>
							<div class="sources" id="sources-container"></div>
						</div>
						<div id="answer-section" style="display:none;">
							<h2>Answer</h2>
							<div class="answer" id="answer-container"></div>
						</div>
						<div class="info-bar" id="info-bar" style="display:none;"></div>
					`;
					
					const sourcesContainer = document.getElementById('sources-container');
					const documentsSection = document.getElementById('documents-section');
					const answerContainer = document.getElementById('answer-container');
					const answerSection = document.getElementById('answer-section');
					const infoBar = document.getElementById('info-bar');
					
					// Make the streaming request
					const response = await fetch('/stream_query', {
						method: 'POST',
						headers: {
							'Content-Type': 'application/json'
						},
						body: JSON.stringify(requestData)
					});
					
					if (!response.ok) {
						throw new Error(`Server error: ${response.status}`);
					}
					
					// Set up the reader for the stream
					const reader = response.body.getReader();
					const decoder = new TextDecoder();
					let buffer = '';
					
					while (true) {
						const { done, value } = await reader.read();
						
						if (done) {
							break;
						}
						
						// Decode the received chunk and add to buffer
						buffer += decoder.decode(value, { stream: true });
						
						// Process complete lines in the buffer
						let lines = buffer.split('\\n');
						buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer
						
						for (const line of lines) {
							if (line.trim() === '') continue;
							
							try {
								const event = JSON.parse(line);
								
								switch (event.event) {
									case 'search_started':
										loading.textContent = 'Searching for relevant documents...';
										break;
										
									case 'documents':
										// Display the documents
										loading.textContent = 'Documents found. Generating answer...';
										documentsSection.style.display = 'block';
										
										if (event.data && event.data.length > 0) {
											let documentHTML = '';
											
											event.data.forEach((doc, index) => {
												const similarity = doc.similarity ? Math.round(doc.similarity * 100) : 0;
												const filePath = doc.metadata.file_path || 'Unknown';
												const fileName = filePath.split('/').pop();
												const projectName = doc.metadata.project || requestData.project;
												
												// Create document link
												const docLink = `/document/${encodeURIComponent(projectName)}/${encodeURIComponent(filePath)}`;
												
												documentHTML += `
													<div class="source-item" id="source-${index}">
														<div class="source-header">
															<div>
																<strong>Source ${index + 1}:</strong> ${fileName}
																<a href="${docLink}" target="_blank" class="doc-link" title="Open document in new tab">
																	<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
																		<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
																		<polyline points="15 3 21 3 21 9"></polyline>
																		<line x1="10" y1="14" x2="21" y2="3"></line>
																	</svg>
																</a>
																<span class="similarity-score">${similarity}% match</span>
															</div>
															<button class="expand-button" onclick="toggleSource('source-${index}')">Expand</button>
														</div>
														<div class="source-content">${doc.content}</div>
													</div>
												`;
											});
											
											sourcesContainer.innerHTML = documentHTML;
										} else {
											sourcesContainer.innerHTML = '<p>No relevant documents found.</p>';
										}
										break;
										
									case 'llm_started':
										loading.textContent = 'Generating answer...';
										break;
										
									case 'answer':
										// Display the answer and metadata
										answerSection.style.display = 'block';
										answerContainer.innerHTML = marked.parse(event.data.answer);
										
										// Show the info bar with metadata
										infoBar.style.display = 'block';
										infoBar.innerHTML = `
											<div>Answered by: ${event.data.llm_type}/${event.data.model_name}</div>
											<div>Response time: ${event.data.elapsed_time.toFixed(2)} seconds</div>
										`;
										break;
										
									case 'error':
										throw new Error(event.message);
										
									case 'complete':
										loading.style.display = 'none';
										break;
								}
							} catch (err) {
								console.error('Error processing event:', err, 'Event data:', line);
							}
						}
					}
					
				} catch (error) {
					response.innerHTML = `<div class="container" style="background-color: #ffeeee; border-left: 3px solid #e74c3c;">
						<h3 style="color: #e74c3c;">Error</h3>
						<p>${error.message}</p>
					</div>`;
					
					loading.style.display = 'none';
				}
			});
			
			// Handle index button
			indexButton.addEventListener('click', async function() {
				const project = projectSelect.value;
				
				if (confirm(`Are you sure you want to re-index project "${project}"? This might take some time.`)) {
					try {
						// Show loading indicator
						loading.style.display = 'block';
						loading.textContent = `Indexing project "${project}"...`;
						
						const res = await fetch(`/index?project=${encodeURIComponent(project)}`, {
							method: 'POST'
						});
						
						if (!res.ok) {
							const errorText = await res.text();
							throw new Error(errorText);
						}
						
						const data = await res.json();
						alert(`Project "${project}" indexed successfully. ${data.document_count} documents indexed.`);
						
					} catch (error) {
						alert(`Error indexing project: ${error.message}`);
					} finally {
						loading.style.display = 'none';
						loading.textContent = 'Processing your query';
					}
				}
			});
		});
		
		// Function to toggle source visibility
		function toggleSource(id) {
			const source = document.getElementById(id);
			const button = source.querySelector('.expand-button');
			
			if (source.classList.contains('expanded')) {
				source.classList.remove('expanded');
				button.textContent = 'Expand';
			} else {
				source.classList.add('expanded');
				button.textContent = 'Collapse';
			}
		}
	</script>
	
	<!-- Add Marked.js for Markdown parsing -->
	<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</body>
</html>
"""

def main():
	# Check if the static directory exists, if not create it and add default files
	static_dir = Path(__file__).parent / "static"
	static_dir.mkdir(exist_ok=True)
	
	# Create templates directory if it doesn't exist
	templates_dir = Path(__file__).parent / "templates"
	templates_dir.mkdir(exist_ok=True)
	
	# Create index.html if it doesn't exist
	index_path = templates_dir / "index.html"
	if not index_path.exists():
		with open(index_path, "w") as f:
			f.write(DEFAULT_INDEX_HTML)
	
	# Check SQLite availability
	sqlite_available = False
	try:
		from sqlite_storage import verify_sqlite_vec_installation
		sqlite_available = verify_sqlite_vec_installation()
	except ImportError:
		logger.warning("sqlite_storage module not found, will use pickle files")
	
	if sqlite_available:
		logger.info("SQLite with vector search is available")
	else:
		logger.info("SQLite with vector search is not available, will use pickle files")
	
	# Set up argument parser
	import argparse
	parser = argparse.ArgumentParser(description="RAG Query API Server")
	parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
	parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
	parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
	parser.add_argument("--debug", action="store_true", help="Enable debug mode")
	args = parser.parse_args()
	
	# Configure logging based on debug flag
	if args.debug:
		logger.setLevel(logging.DEBUG)
		logger.debug("Debug mode enabled")
	
	# Start the server
	logger.info(f"Starting API server at http://{args.host}:{args.port}")
	logger.info(f"Open your browser to http://{args.host}:{args.port}/ to use the web interface")
	
	uvicorn.run(
		"api_server:app", 
		host=args.host, 
		port=args.port, 
		reload=args.reload,
		log_level="info"
	)

if __name__ == "__main__":
	main()