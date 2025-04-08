#!/usr/bin/env python3
"""
RAG Query Application with Project Support and Colorful Interface

This application:
1. Loads document indexes created by the document_indexer
2. Supports querying specific projects or the master index
3. Retrieves relevant documents based on the query
4. Sends the query and context to various LLMs for answering
5. Supports on-demand indexing of projects
6. Features a colorful terminal interface
7. Logs prompts to JSON files when in debug mode
"""

import os
import sys
import json
import argparse
import re
import time
import readline
import traceback
import atexit
import pickle
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime


# Import from our shared functions library
from shared_functions import (
	# Constants
	DEFAULT_INDEX_DIR, DEFAULT_DOCUMENT_DIR, DEFAULT_EMBEDDING_MODEL, 
	DEFAULT_EMBEDDING_TYPE, TOP_K_DOCUMENTS, MASTER_PROJECT, PROMPTS_DIR,
	DEFAULT_CHARS_PER_DIMENSION, LLM_CLAUDE, LLM_LOCAL, LLM_HF, LLM_OPENAI,
	DEFAULT_LLM_TYPE, DEFAULT_LOCAL_MODEL, DEFAULT_HF_MODEL, DEFAULT_CLAUDE_MODEL,
	DEFAULT_OPENAI_MODEL, COLORS_AVAILABLE, HIGHLIGHT_COLOR, RESET_COLOR, QUERY_COLOR, SYSTEM_COLOR, ANSWER_COLOR,
	
	# Classes
	Document, EmbeddingProviderCache,
	
	# Functions
	get_project_config_path, load_project_config_file, get_project_embedding_config,
	load_index, discover_projects, clear_index, index_project, get_index_path,
	save_embedding_config, is_command, get_model_name_for_llm_type,
	search_documents, get_response, display_search_results, batch_process, read_query_from_file, print_debug, print_system, print_error, ensure_directory_structure
)

# Import our embedding library
from embeddings import EmbeddingConfig

# Set environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_FALLBACK_POLICY"] = "0" 

# Color scheme
# QUERY_COLOR = Fore.GREEN
# ANSWER_COLOR = Fore.CYAN
# DEBUG_COLOR = Fore.YELLOW
# ERROR_COLOR = Fore.RED
# SYSTEM_COLOR = Fore.MAGENTA
# HIGHLIGHT_COLOR = Fore.WHITE + Style.BRIGHT
# RESET_COLOR = Style.RESET_ALL

class CommandHistory:
	"""Manages command history for interactive mode."""
	
	def __init__(self, history_dir="logs/history", max_size=1000):
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
		from datetime import datetime
		
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

# def print_debug(message: str) -> None:
# 	"""Print debug message in debug color."""
# 	print(f"{DEBUG_COLOR}[DEBUG] {message}{RESET_COLOR}")
# 
# def print_error(message: str) -> None:
# 	"""Print error message in error color."""
# 	print(f"{ERROR_COLOR}Error: {message}{RESET_COLOR}")
# 
# def print_system(message: str) -> None:
# 	"""Print system message in system color."""
# 	print(f"{SYSTEM_COLOR}{message}{RESET_COLOR}")

def print_help_info(current_project: str, current_llm_type: str, current_model: str, 
				   current_rag_mode: str, current_rag_count: int,
				   current_system_prompt: str = None) -> None:     
	"""
	Print help information about available commands.
	"""
	
	print_system(f"Current Project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
	print_system(f"Current LLM: {HIGHLIGHT_COLOR}{current_llm_type}:{current_model}{RESET_COLOR}")
	print_system(f"Current RAG Mode: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
	print_system(f"Document Count: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
	
	# Add system prompt info if available
	if current_system_prompt:
		# If the prompt is long, truncate it for display
		display_prompt = current_system_prompt
		if len(display_prompt) > 50:
			display_prompt = display_prompt[:47] + "..."
		print_system(f"System Prompt: {HIGHLIGHT_COLOR}\"{display_prompt}\"{RESET_COLOR}")

	print_system("\nAvailable Commands:")
	print_system("  help                     Show this help information")
	print_system("  exit, quit               End the session")
	
	# Project commands
	print_system("\nProject Commands:")
	print_system("  projects                 List all available projects")
	print_system("  project <name>           Switch to a different project")
	print_system("  config                   Show current project configuration")
	
	# Index commands
	print_system("\nIndex Commands:")
	print_system("  index                    Re-index the current project")
	print_system("  index clear              Clear the current project's index")
	
	print_system("\nSQLite Commands:")
	print_system("  sqlite verify           Checks Sqlite is installed correctly")
	print_system("  sqlite inspect          Shows key information about SQLite projct DB")

	# RAG commands
	print_system("\nRAG Commands:")
	print_system("  rag mode <mode>          Set RAG mode (chunk, file, none)")
	print_system("  rag count <number>       Set number of documents to retrieve")
	print_system("  defaults save            Save current settings as defaults")
	print_system("  defaults read            Load default settings to current configuration")

	# LLM commands
	print_system("\nLLM Commands:")
	print_system("  models                   List available LLM models")
	print_system("  llm claude [model_name]  Use Claude API (with optional model)")
	print_system("  llm openai [model_name]  Use OpenAI API (with optional model)")
	print_system("  llm local [model_name]   Use a local model via llm library")
	print_system("  llm hf [model_name]      Use a Hugging Face model")
	
	# Add system prompt command to the help info
	print_system("\nSystem Prompt Commands:")
	print_system("  system prompt \"<prompt>\"  Set the system prompt for the current LLM")
	print_system("  system prompt show        Show the current system prompt")
	print_system("  system prompt clear       Clear the current system prompt")
		
	# History commands
	print_system("\nHistory Commands:")
	print_system("  history                  Show command history")
	print_system("  history clear            Clear command history")
	print_system("  history save             Save history to a file")
	
	print_system("\nFor any other input, the application will treat it as a query")
	print_system("and search for relevant documents to help answer it.")   

def interactive_mode(documents: List[Document], api_key: str, project: str, 
					document_dir: str, index_dir: str, 
					embedding_config: Optional[EmbeddingConfig] = None,
					debug: bool = False, prompts_dir: str = PROMPTS_DIR,
					llm_type: str = LLM_CLAUDE, local_model: str = DEFAULT_LOCAL_MODEL,
					hf_model: str = DEFAULT_HF_MODEL, claude_model: str = DEFAULT_CLAUDE_MODEL,
					history_dir: str = "history",
					rag_count: Optional[int] = None,
					system_prompt: Optional[str] = None,
					use_sqlite: bool = False) -> None:
	"""Run the application in interactive mode."""
	print_system(f"RAG Query Application - Interactive Mode (Project: {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR})")
	print_system("Type 'help' to see available commands")
	
	# Load complete project configuration
	project_config = load_project_config_file(project, document_dir)
	
	# Initialize variables
	current_project = project
	current_documents = documents
	current_embedding_config = embedding_config or get_project_embedding_config(project, document_dir, debug)
	
	# Get RAG settings from project config
	rag_settings = project_config.get("rag", {})
	current_llm_type = rag_settings.get("llm_type", llm_type)
	current_local_model = rag_settings.get("llm_model", local_model) if current_llm_type == LLM_LOCAL else local_model
	current_hf_model = rag_settings.get("llm_model", hf_model) if current_llm_type == LLM_HF else hf_model
	current_claude_model = rag_settings.get("llm_model", claude_model) if current_llm_type == LLM_CLAUDE else claude_model
	current_openai_model = rag_settings.get("llm_model", DEFAULT_OPENAI_MODEL) if current_llm_type == LLM_OPENAI else DEFAULT_OPENAI_MODEL
	current_rag_mode = rag_settings.get("rag_mode", "chunk")
	
	# Define get_current_model_name function BEFORE using it
	def get_current_model_name():
		if current_llm_type == LLM_LOCAL:
			return current_local_model
		elif current_llm_type == LLM_HF:
			return current_hf_model
		elif current_llm_type == LLM_CLAUDE:
			return current_claude_model
		elif current_llm_type == LLM_OPENAI:
			return current_openai_model
		else:
			return "unknown"    
	
	# Initialize system prompt from:
	# 1. Command line argument
	# 2. Project config
	# 3. System settings for the current model

	current_system_prompt = None

	if system_prompt:
		current_system_prompt = system_prompt
	elif "system_prompt" in rag_settings:
		current_system_prompt = rag_settings.get("system_prompt")
	else:
		# Try to get from model-specific system prompts
		current_model = get_current_model_name()
		system_settings = project_config.get("system", {})
		if current_model in system_settings:
			current_system_prompt = system_settings[current_model].get("system_prompt")
	
	if debug and current_system_prompt:
		print_debug(f"Using system prompt: \"{current_system_prompt}\"")
	
	if rag_count is not None:
		current_rag_count = rag_count
	else:
		current_rag_count = rag_settings.get("rag_count", TOP_K_DOCUMENTS)
			
	# Add function to save system prompt to project config
	def save_system_prompt(prompt: str, save_to_defaults: bool = False):
		"""Save system prompt to the project configuration."""
		project_config = load_project_config_file(current_project, document_dir)
		
		# Ensure rag section exists
		if "rag" not in project_config:
			project_config["rag"] = {}
		
		# Save to rag settings
		project_config["rag"]["system_prompt"] = prompt
		
		# If requested, also save to defaults
		if save_to_defaults:
			if "defaults" not in project_config:
				project_config["defaults"] = {}
			project_config["defaults"]["system_prompt"] = prompt
			
			# Save to system settings for the current model
			current_model = get_current_model_name()
			if "system" not in project_config:
				project_config["system"] = {}
			
			if current_model not in project_config["system"]:
				project_config["system"][current_model] = {}
			
			project_config["system"][current_model]["system_prompt"] = prompt
		
		# Save the updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Saved system prompt to project config")
			if save_to_defaults:
				print_debug(f"Also saved to defaults and system settings for model: {current_model}")
	
	# Modify save_current_settings_as_defaults function to include system prompt
	def save_current_settings_as_defaults():
		"""
		Save the current RAG settings as defaults in the project configuration.
		"""
		
		if debug:
			print_debug("save_current_settings_as_defaults")
		
		# Get the current model name based on LLM type
		current_model = get_current_model_name()
		
		# Get current project config
		project_config = load_project_config_file(current_project, document_dir)
		
		# Ensure we have a defaults section
		if "defaults" not in project_config:
			project_config["defaults"] = {}
		
		# Copy current RAG settings to defaults
		project_config["defaults"] = {
			"llm_type": current_llm_type,
			"llm_model": current_model,
			"rag_mode": current_rag_mode,
			"rag_count": current_rag_count
		}
		
		# Add system prompt if available
		if current_system_prompt:
			project_config["defaults"]["system_prompt"] = current_system_prompt
			
			# Also save to system section for the current model
			if "system" not in project_config:
				project_config["system"] = {}
				
			if current_model not in project_config["system"]:
				project_config["system"][current_model] = {}
				
			project_config["system"][current_model]["system_prompt"] = current_system_prompt
		
		# Save updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Saved current settings as defaults to {config_path}")
			print_debug(f"  llm_type: {current_llm_type}")
			print_debug(f"  llm_model: {current_model}")
			print_debug(f"  rag_mode: {current_rag_mode}")
			print_debug(f"  rag_count: {current_rag_count}")
			if current_system_prompt:
				print_debug(f"  system_prompt: \"{current_system_prompt}\"")
		
	# Modify load_defaults_to_current_settings to include system prompt
	def load_defaults_to_current_settings():
		"""
		Load default settings from project configuration and apply them to current RAG settings.
		
		Returns:
			Dictionary containing the loaded default settings or empty dict if none found
		"""
		# Get current project config
		project_config = load_project_config_file(current_project, document_dir)
		
		# Check if we have defaults
		if "defaults" not in project_config or not project_config["defaults"]:
			if debug:
				print_debug("No default settings found in project configuration")
			return {}
		
		# Get defaults
		defaults = project_config["defaults"]
		
		# Copy defaults to current RAG settings
		project_config["rag"] = dict(defaults)
		
		# Save updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Loaded default settings to current RAG settings:")
			print_debug(f"  llm_type: {defaults.get('llm_type')}")
			print_debug(f"  llm_model: {defaults.get('llm_model')}")
			print_debug(f"  rag_mode: {defaults.get('rag_mode')}")
			print_debug(f"  rag_count: {defaults.get('rag_count')}")
			if "system_prompt" in defaults:
				print_debug(f"  system_prompt: \"{defaults.get('system_prompt')}\"")
		
		return defaults
			
	# Function to save current RAG settings to project config
	def save_current_rag_settings():
		# Get current project config
		project_config = load_project_config_file(current_project, document_dir)
		
		# Get current model based on LLM type
		model_to_save = get_current_model_name()
		
		# Update RAG settings
		project_config["rag"] = {
			"llm_type": current_llm_type,
			"llm_model": model_to_save,
			"rag_mode": current_rag_mode,
			"rag_count": current_rag_count
		}
		
		# Add system prompt if available
		if current_system_prompt:
			project_config["rag"]["system_prompt"] = current_system_prompt
		
		# Save updated config
		config_path = get_project_config_path(current_project, document_dir, use_legacy=False)
		os.makedirs(os.path.dirname(config_path), exist_ok=True)
		
		with open(config_path, 'w') as f:
			json.dump(project_config, f, indent=2)
		
		if debug:
			print_debug(f"Saved updated RAG settings to {config_path}")
			print_debug(f"  llm_type: {current_llm_type}")
			print_debug(f"  llm_model: {model_to_save}")
			print_debug(f"  rag_mode: {current_rag_mode}")
			print_debug(f"  rag_count: {current_rag_count}")
			if current_system_prompt:
				print_debug(f"  system_prompt: \"{current_system_prompt}\"")
	
	# Initialize history
	history = CommandHistory(history_dir="logs/history" if history_dir == "history" else history_dir)	

	# Initialize embedding provider cache
	provider_cache = EmbeddingProviderCache(debug=debug)
	
	# Print the initial help info
	print_help_info(current_project, current_llm_type, get_current_model_name(), current_rag_mode, current_rag_count)
	
	while True:
		try:
			# Print the prompt with the current project and LLM highlighted
			current_model = get_current_model_name()
			prompt = f"\n{QUERY_COLOR}Enter your question [{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{QUERY_COLOR}] [{HIGHLIGHT_COLOR}{current_llm_type}:{current_model}{RESET_COLOR}{QUERY_COLOR}]: {RESET_COLOR}"
			
			# Use print and input separately to ensure proper scrolling behavior
			print(prompt, end='', flush=True)
			query = input().strip()
			
			# The rest of the function continues below...
			
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
				print_help_info(current_project, current_llm_type, get_current_model_name(), 
							  current_rag_mode, current_rag_count)
				continue
				
			# verify sqlite install
			elif query.lower() == 'sqlite verify':
				from sqlite_storage import verify_sqlite_vec_installation
				verify_sqlite_vec_installation()
				continue
				
			elif query.lower() == 'sqlite inspect':
				if use_sqlite:
					from sqlite_storage import verify_database_tables, get_db_path
					db_path = get_db_path(index_dir, current_project)
					if os.path.exists(db_path):
						print_system(f"Inspecting SQLite database for project: {current_project}")
						verify_database_tables(db_path, debug=True)
					else:
						print_system(f"SQLite database not found for project: {current_project}")
						print_system(f"Expected at: {db_path}")
				else:
					print_system("Not using SQLite storage")
				continue

			
			# Handle special commands
			elif query.lower() == 'history':
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
				
			# Add to the interactive_mode function where it handles commands
			elif query.lower() == 'defaults save':
				# Save current settings as defaults
				save_current_settings_as_defaults()
				
				current_model = get_current_model_name()
				print_system(f"Saved current settings as defaults:")
				print_system(f"  LLM Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				print_system(f"  LLM Model: {HIGHLIGHT_COLOR}{current_model}{RESET_COLOR}")
				print_system(f"  RAG Mode: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
				print_system(f"  Document Count: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
				if current_system_prompt:
					print_system(f"  System Prompt: {HIGHLIGHT_COLOR}\"{current_system_prompt}\"{RESET_COLOR}")
				continue
			
			elif query.lower() == 'defaults read':
				# Load defaults to current settings
				defaults = load_defaults_to_current_settings()
				
				if defaults:
					# Update current variables with defaults
					current_llm_type = defaults.get('llm_type', current_llm_type)
					
					# Update the appropriate model variable based on LLM type
					model_name = defaults.get('llm_model', '')
					if current_llm_type == LLM_LOCAL:
						current_local_model = model_name or current_local_model
					elif current_llm_type == LLM_HF:
						current_hf_model = model_name or current_hf_model
					elif current_llm_type == LLM_CLAUDE:
						current_claude_model = model_name or current_claude_model
					elif current_llm_type == LLM_OPENAI:
						current_openai_model = model_name or current_openai_model
					
					current_rag_mode = defaults.get('rag_mode', current_rag_mode)
					current_rag_count = defaults.get('rag_count', current_rag_count)
					
					# First check if there's a system prompt in defaults
					if "system_prompt" in defaults:
						current_system_prompt = defaults.get("system_prompt")
					else:
						# If not, check if there's a system prompt for the current model
						current_model = get_current_model_name()
						system_settings = project_config.get("system", {})
						if current_model in system_settings:
							model_settings = system_settings[current_model]
							if "system_prompt" in model_settings:
								current_system_prompt = model_settings["system_prompt"]
					
					print_system(f"Loaded default settings:")
					print_system(f"  LLM Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
					print_system(f"  LLM Model: {HIGHLIGHT_COLOR}{get_current_model_name()}{RESET_COLOR}")
					print_system(f"  RAG Mode: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
					print_system(f"  Document Count: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
					if current_system_prompt:
						print_system(f"  System Prompt: {HIGHLIGHT_COLOR}\"{current_system_prompt}\"{RESET_COLOR}")
				else:
					print_system("No default settings found in project configuration")
				continue            
			
			# Handle other special commands
			elif query.lower() == 'projects':
				projects = discover_projects(document_dir, index_dir, use_sqlite)
				print_system("\nAvailable Projects:")
				for p in projects:
					marker = "*" if p == current_project else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{p}{RESET_COLOR}")
				continue
			
			elif query.lower() == 'config':
				# Enhanced config command to show all settings
				print_system("\nCurrent Project Configuration:")
				print_system(f"Project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
				
				# Show embedding config
				print_system("\nIndexing Configuration:")
				print_system(f"  Embedding Type: {current_embedding_config.embedding_type}")
				print_system(f"  Embedding Model: {current_embedding_config.model_name}")
				print_system(f"  Embedding Dimensions: {current_embedding_config.dimensions}")
				
				# Show RAG config
				print_system("\nRAG Configuration:")
				print_system(f"  LLM Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				print_system(f"  LLM Model: {HIGHLIGHT_COLOR}{get_current_model_name()}{RESET_COLOR}")
				print_system(f"  RAG Mode: {current_rag_mode}")
				print_system(f"  Document Count: {current_rag_count}")
				
				# Show config file path
				config_path = get_project_config_path(current_project, document_dir)
				if os.path.exists(config_path):
					print_system(f"\nConfig File: {config_path}")
				else:
					legacy_path = get_project_config_path(current_project, document_dir, use_legacy=True)
					if os.path.exists(legacy_path):
						print_system(f"\nLegacy Config File: {legacy_path}")
					else:
						print_system("\nConfig File: Not found (using defaults)")
				continue
			
			# Set RAG count command
			elif query.lower().startswith('rag count '):
				try:
					count = int(query[10:].strip())
					if count < 1:
						print_error("RAG count must be at least 1")
					else:
						current_rag_count = count
						print_system(f"Set RAG document count to: {HIGHLIGHT_COLOR}{current_rag_count}{RESET_COLOR}")
						save_current_rag_settings()
				except ValueError:
					print_error("Invalid RAG count. Please specify a number.")
				continue
			
			# Set RAG mode command
			elif query.lower().startswith('rag mode '):
				mode = query[9:].strip().lower()
				if mode in ["chunk", "file", "none"]:
					current_rag_mode = mode
					print_system(f"Set RAG mode to: {HIGHLIGHT_COLOR}{current_rag_mode}{RESET_COLOR}")
					save_current_rag_settings()
				else:
					print_error(f"Invalid RAG mode: {mode}")
					print_system("Valid modes are: chunk, file, none")
				continue
			
				
			elif query.lower() == 'index clear':
				# Ask for confirmation
				confirm = input(f"{SYSTEM_COLOR}Are you sure you want to clear the index for project '{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{SYSTEM_COLOR}'? This cannot be undone. (y/n): {RESET_COLOR}").strip().lower()
				
				if confirm == 'y':
					print_system(f"\nClearing index for project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
					
					# Clear the index using the appropriate method
					if use_sqlite:
						from sqlite_storage import clear_project_index
						success = clear_project_index(index_dir, current_project, debug)
					else:
						success = clear_index(current_project, index_dir, use_sqlite, debug)
					
					if success:
						# Reset the current documents to an empty list
						current_documents = []
						print_system(f"Project '{HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}{SYSTEM_COLOR}' index cleared successfully")
						print_system(f"The index is now empty")
				
				continue
	
			
			# create or update the index
			elif query.lower() == 'index':
				print_system(f"\nIndexing project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
				success = index_project(
					current_project, 
					document_dir, 
					index_dir, 
					debug=debug,
					auto_adjust_chunks=True,
					chars_per_dimension=DEFAULT_CHARS_PER_DIMENSION,
					use_sqlite=use_sqlite
				)
				
				if success:
					# Reload the project index using the appropriate method
					if use_sqlite:
						from sqlite_storage import load_documents
						current_documents = load_documents(index_dir, current_project, debug=debug)
					else:
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
			
			# Handle system prompt commands
			elif query.lower() == 'system prompt show':
				if current_system_prompt:
					print_system(f"Current system prompt:")
					print_system(f"{HIGHLIGHT_COLOR}\"{current_system_prompt}\"{RESET_COLOR}")
				else:
					print_system("No system prompt is currently set")
				continue
				
			elif query.lower() == 'system prompt clear':
				current_system_prompt = None
				save_system_prompt(None)
				print_system("System prompt cleared")
				save_current_rag_settings()
				continue
				
			elif query.lower().startswith('system prompt "') or query.lower().startswith('system prompt \''):
				# Extract the prompt text between quotes
				match = re.match(r'system prompt ["\'](.*)["\']$', query)
				if match:
					new_prompt = match.group(1)
					current_system_prompt = new_prompt
					save_system_prompt(new_prompt)
					print_system(f"System prompt set to:")
					print_system(f"{HIGHLIGHT_COLOR}\"{new_prompt}\"{RESET_COLOR}")
					save_current_rag_settings()
				else:
					print_error("Invalid system prompt format. Use: system prompt \"your prompt here\"")
				continue

			
				
			# Add to the models command in interactive_mode
			elif query.lower() == 'models':
				# List both llm and Hugging Face models
				print_system("\nCurrent LLM Settings:")
				print_system(f"  Type: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR}")
				
				if current_llm_type == LLM_LOCAL:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
				elif current_llm_type == LLM_HF:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
				elif current_llm_type == LLM_CLAUDE:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_claude_model}{RESET_COLOR}")
				elif current_llm_type == LLM_OPENAI:
					print_system(f"  Model: {HIGHLIGHT_COLOR}{current_openai_model}{RESET_COLOR}")
				
				# Add OpenAI model listing
				print_system("\nAvailable OpenAI models (--llm openai):")
				openai_models = [
					"o3-mini",
					"gpt-3.5-turbo",
					"gpt-4-turbo",
					"gpt-4",
					"gpt-4o",
					"gpt-4o-mini"
				]
				for model in openai_models:
					marker = "*" if model == current_openai_model and current_llm_type == LLM_OPENAI else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{model}{RESET_COLOR}")
				
				# Add Claude model listing
				print_system("\nAvailable Claude models (--llm claude):")
				claude_models = [
					"claude-3-5-haiku-20241022",
					"claude-3-sonnet-20240229",
					"claude-3-opus-20240229",
					"claude-3-haiku-20240307",
					"claude-3-sonnet-20240229"
				]
				for model in claude_models:
					marker = "*" if model == current_claude_model and current_llm_type == LLM_CLAUDE else " "
					print_system(f"{marker} {HIGHLIGHT_COLOR}{model}{RESET_COLOR}")

				
				# Try Simon Willison's llm library first
				llm_found = False
				try:
					import llm
					llm_found = True
					
					# Try different methods to get models
					try:
						models = llm.get_models()
						model_names = [getattr(m, 'model_id', str(m)) for m in models]
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
					if model_arg:
						current_claude_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_claude_model}{RESET_COLOR}")
					else:
						current_claude_model = DEFAULT_CLAUDE_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_claude_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				elif llm_choice == LLM_OPENAI:
					current_llm_type = LLM_OPENAI
					if model_arg:
						current_openai_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_openai_model}{RESET_COLOR}")
					else:
						current_openai_model = DEFAULT_OPENAI_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_openai_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				elif llm_choice == LLM_LOCAL:
					current_llm_type = LLM_LOCAL
					if model_arg:
						current_local_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
					else:
						current_local_model = DEFAULT_LOCAL_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_local_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				elif llm_choice == LLM_HF:
					current_llm_type = LLM_HF
					if model_arg:
						current_hf_model = model_arg
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
					else:
						current_hf_model = DEFAULT_HF_MODEL  # Reset to default if not specified
						print_system(f"Changed LLM to: {HIGHLIGHT_COLOR}{current_llm_type}{RESET_COLOR} with default model: {HIGHLIGHT_COLOR}{current_hf_model}{RESET_COLOR}")
					
					# Save changes to config
					save_current_rag_settings()
				
				else:
					print_error(f"Unknown LLM type: {llm_choice}")
					print_system("Valid options are:")
					print_system("  llm claude [model_name]")
					print_system("  llm openai [model_name]")
					print_system("  llm local [model_name]")
					print_system("  llm hf [model_name]")
				
				continue

			
			elif query.lower().startswith('project '):
				new_project = query[8:].strip()
				
				# Get the index path based on the new project
				index_path, backup_dir = get_index_path(index_dir, new_project)
				
				# Determine whether to use SQLite or pickle files
				index_exists = False
				if use_sqlite:
					# For SQLite, check if the database file exists
					from sqlite_storage import get_db_path, document_count
					db_path = get_db_path(index_dir, new_project)
					index_exists = os.path.exists(db_path)
					
					if index_exists:
						# Check if there are documents
						docs_count = document_count(index_dir, new_project)
						index_exists = docs_count > 0
						if debug:
							print_debug(f"SQLite database exists: {os.path.exists(db_path)}, contains {docs_count} documents")

					
				else:
					# For pickle files, check if the index file exists
					index_exists = os.path.exists(index_path)
					if index_exists:
						# Check if it's not empty
						try:
							with open(index_path, 'rb') as f:
								documents = pickle.load(f)
							index_exists = len(documents) > 0
						except:
							index_exists = False

				
				if not index_exists:
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
						success = index_project(
							new_project, 
							document_dir, 
							index_dir, 
							debug=debug, 
							use_sqlite=use_sqlite
						)
				
						if success:
							current_project = new_project
							
							# Load the documents using the appropriate method
							if use_sqlite:
								from sqlite_storage import load_documents
								current_documents = load_documents(index_dir, current_project, debug=debug)
							else:
								current_documents = load_index(index_path, backup_dir, debug)
							
							# Load the project configuration
							project_config = load_project_config_file(current_project, document_dir)
							
							# Update embedding config
							current_embedding_config = get_project_embedding_config(current_project, document_dir, debug)
							
							# Update RAG settings from project config
							rag_settings = project_config.get("rag", {})
							current_llm_type = rag_settings.get("llm_type", current_llm_type)
							current_local_model = rag_settings.get("llm_model", current_local_model)
							current_hf_model = rag_settings.get("llm_model", current_hf_model) if current_llm_type == LLM_HF else current_hf_model
							current_rag_mode = rag_settings.get("rag_mode", current_rag_mode)
							current_rag_count = rag_settings.get("rag_count", current_rag_count)
							
							# Preload the embedding provider for the new project
							provider_cache.get_provider(current_project, document_dir, current_embedding_config)
							
							print_system(f"Switched to project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
					continue
				
				# In the interactive_mode function, in the part that handles 'project' command:
				# Load the new project's documents or get document count
				if use_sqlite:
					from sqlite_storage import document_count, load_documents
					doc_count = document_count(index_dir, new_project)
				
					if doc_count > 0:
						current_project = new_project
						# Don't set current_documents to an empty list
						# Instead, load a sample for compatibility
						current_documents = load_documents(index_dir, new_project, limit=50, debug=debug)
					
						if debug:
							print_debug(f"Loaded {len(current_documents)} sample documents for project {new_project}")
							print_debug(f"Total documents in database: {doc_count}")
						
						# Load the project configuration
						project_config = load_project_config_file(new_project, document_dir)
						
						# Update embedding config
						current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
						
						# Update RAG settings from project config
						rag_settings = project_config.get("rag", {})
						current_llm_type = rag_settings.get("llm_type", current_llm_type)
						current_local_model = rag_settings.get("llm_model", current_local_model)
						current_hf_model = rag_settings.get("llm_model", current_hf_model) if current_llm_type == LLM_HF else current_hf_model
						current_rag_mode = rag_settings.get("rag_mode", current_rag_mode)
						current_rag_count = rag_settings.get("rag_count", current_rag_count)
						
						# Preload the embedding provider for the new project
						provider_cache.get_provider(current_project, document_dir, current_embedding_config)
						
						print_system(f"Switched to project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
						print_system(f"Using SQLite storage with {doc_count} documents")
						print_system(f"Embedding Type: {current_embedding_config.embedding_type}")
						print_system(f"Embedding Model: {current_embedding_config.model_name}")
						print_system(f"Embedding Dimensions: {current_embedding_config.dimensions}")
						print_system(f"LLM Type: {current_llm_type}")
						print_system(f"RAG Mode: {current_rag_mode}")
						print_system(f"Document Count: {current_rag_count}")
					else:
						print_system(f"No documents found in project: {HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}")
						print_system(f"Add .txt or .md documents to the project folder and run 'index'")
				
				else:
					# Original pickle file loading
					new_documents = load_index(index_path, backup_dir, debug)
				
					if new_documents:
						current_project = new_project
						current_documents = new_documents
						
						# Load the project configuration
						project_config = load_project_config_file(new_project, document_dir)
						
						# Update embedding config
						current_embedding_config = get_project_embedding_config(new_project, document_dir, debug)
						
						# Update RAG settings from project config
						rag_settings = project_config.get("rag", {})
						current_llm_type = rag_settings.get("llm_type", current_llm_type)
						current_local_model = rag_settings.get("llm_model", current_local_model)
						current_hf_model = rag_settings.get("llm_model", current_hf_model) if current_llm_type == LLM_HF else current_hf_model
						current_rag_mode = rag_settings.get("rag_mode", current_rag_mode)
						current_rag_count = rag_settings.get("rag_count", current_rag_count)
						
						# Preload the embedding provider for the new project
						provider_cache.get_provider(current_project, document_dir, current_embedding_config)
						
						print_system(f"Switched to project: {HIGHLIGHT_COLOR}{current_project}{RESET_COLOR}")
						print_system(f"Embedding Type: {current_embedding_config.embedding_type}")
						print_system(f"Embedding Model: {current_embedding_config.model_name}")
						print_system(f"Embedding Dimensions: {current_embedding_config.dimensions}")
						print_system(f"LLM Type: {current_llm_type}")
						print_system(f"RAG Mode: {current_rag_mode}")
						print_system(f"Document Count: {current_rag_count}")
					else:
						print_system(f"No documents found in project: {HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}")
						print_system(f"Add .txt or .md documents to the project folder: {HIGHLIGHT_COLOR}{new_project}{RESET_COLOR}")
				
				continue

			# Regular query - search for relevant documents
			# Echo the query if it's not a command
			if not is_command(query):
				# Add a blank line after the query for better readability
				print()
				
				# In 'none' RAG mode, we skip the document search
				if current_rag_mode.lower() == "none":
					relevant_docs = []
					if debug:
						print_debug(f"Using RAG mode 'none' - skipping document search")
				else:
					print_system("Searching for relevant documents...")
					
					# Pass the provider cache to the search function
					relevant_docs = search_documents(
						query, current_documents, current_project, 
						document_dir, current_embedding_config, 
						top_k=current_rag_count,
						debug=debug, provider_cache=provider_cache,
						rag_mode=current_rag_mode,
						use_sqlite=use_sqlite,
						index_dir=index_dir
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
				elif current_llm_type == LLM_CLAUDE:
					model_name = current_claude_model
				elif current_llm_type == LLM_OPENAI:
					model_name = current_openai_model
				
				# Ask the selected LLM
				print_system(f"Generating answer with {current_llm_type}:{model_name} (RAG mode: {current_rag_mode})...")
				
				# Pass the system prompt to get_response
				answer = get_response(
					query, relevant_docs, api_key, current_project,
					current_llm_type, model_name, debug, prompts_dir,
					current_rag_mode, document_dir, current_system_prompt
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

def main():
	"""Main entry point for the query application."""
	parser = argparse.ArgumentParser(description="RAG Query Application with Project Support")
	
	parser.add_argument("--api-key", type=str, help="API key for selected LLM (Claude or OpenAI)")
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
						help="LLM to use: 'claude', 'openai', 'local' (Simon Willison's llm), or 'hf' (Hugging Face)")
	parser.add_argument("--local-model", type=str, default=DEFAULT_LOCAL_MODEL,
						help="Local model to use when --llm=local (default: gpt4all)")
	parser.add_argument("--hf-model", type=str, default=DEFAULT_HF_MODEL,
						help="Hugging Face model to use when --llm=hf")
	parser.add_argument("--history-dir", type=str, default="history",
						help="Directory to store command history")
	
	# Add the new command line argument for rag-count
	parser.add_argument("--rag-count", type=int,
						help=f"Number of documents to retrieve (default: {TOP_K_DOCUMENTS})")
							
	parser.add_argument("--model", type=str, default=None,
					   help="Model to use for the selected LLM")
					   
	# add argument for system prompt
	parser.add_argument("--system-prompt", type=str, 
						help="System prompt to use for the LLM")
	
	# for working with sqlite                    
	parser.add_argument("--use-sqlite", action="store_true",
						help="Use SQLite with vector search instead of pickle files")
						
	# for batch processing	
	# In the argument parser section of main(), add these new arguments:
	parser.add_argument("--batch-models", type=str,
						help="Comma-separated list of models to run in batch mode")
	parser.add_argument("--batch-output", type=str,
						help="Output JSON file for batch results (default: batch_results_<timestamp>.json)")
	parser.add_argument("--prompt-file", type=str,
						help="File containing the prompt/query to use")


	args = parser.parse_args()
	
	# Create required directory structure
	ensure_directory_structure()

	
	# Update API key handling
	api_key = args.api_key
	if args.llm.lower() == LLM_CLAUDE:
		api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
		if not api_key:
			print_error("Anthropic API key is required to use Claude.")
			print_error("Please provide it via --api-key or set the ANTHROPIC_API_KEY environment variable.")
	elif args.llm.lower() == LLM_OPENAI:
		api_key = api_key or os.environ.get("OPENAI_API_KEY")
		if not api_key:
			print_error("OpenAI API key is required to use OpenAI models.")
			print_error("Please provide it via --api-key or set the OPENAI_API_KEY environment variable.")
	
	# Set prompts directory from args
	if args.prompts_dir != PROMPTS_DIR:
		prompts_dir = args.prompts_dir
		os.makedirs(prompts_dir, exist_ok=True)
		if args.debug:
			print_debug(f"Prompt logs will be saved to: {os.path.abspath(prompts_dir)}")
	else:
		prompts_dir = PROMPTS_DIR
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
	
	# Check if document directory exists
	if not os.path.exists(args.document_dir):
		print_system(f"Document directory not found: {args.document_dir}")
		print_system("Document directory created.")
		os.makedirs(args.document_dir, exist_ok=True)
		print_system("Please add your projects and files in the 'documents' directory.")
	
	# Create the index directory if it doesn't exist
	os.makedirs(args.index_dir, exist_ok=True)
	
	# Create prompts directory if in debug mode
	if args.debug:
		os.makedirs(PROMPTS_DIR, exist_ok=True)
		print_debug(f"Prompt logs will be saved to: {os.path.abspath(PROMPTS_DIR)}")
	
	# Just list projects if requested
	if args.list_projects:
		projects = discover_projects(args.document_dir, args.index_dir, args.use_sqlite)
		if not projects:
			print_system("No indexed projects found.")
			return
			
		print_system("\nAvailable Projects:")
		for project in projects:
			# Get the document count for this project
			if args.use_sqlite:
				try:
					from sqlite_storage import document_count
					docs_count = document_count(args.index_dir, project)
					print_system(f"  {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR} ({docs_count} documents in SQLite)")
				except:
					print_system(f"  {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR} (error accessing SQLite database)")
			else:
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
					
				except:
					print_system(f"  {HIGHLIGHT_COLOR}{project}{RESET_COLOR}{SYSTEM_COLOR} (error loading index)")
			
			# Show config file path and info
			config_path = get_project_config_path(project, args.document_dir)
			if os.path.exists(config_path):
				try:
					config = load_project_config_file(project, args.document_dir)
					indexing_config = config.get("indexing", {})
					rag_config = config.get("rag", {})
					print_system(f"    Config: Embedding={indexing_config.get('embedding_type', 'unknown')}/{indexing_config.get('model_name', 'unknown')}, LLM={rag_config.get('llm_type', 'unknown')}")
				except:
					print_system(f"    Config: Error loading {config_path}")
			else:
				print_system(f"    Config: Not found (using defaults)")
					
		return
	
	# Create embedding configuration from command line args
	embedding_config = None
	if args.embedding_model or args.embedding_type:
		embedding_config = EmbeddingConfig(
			embedding_type=args.embedding_type or DEFAULT_EMBEDDING_TYPE,
			model_name=args.embedding_model or DEFAULT_EMBEDDING_MODEL
		)
	
	# To clear the index
	if args.index_clear:
		print_system(f"Clearing index for project: {HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}")
		success = clear_index(args.project, args.index_dir, args.use_sqlite, args.debug)
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
		success = index_project(
			args.project, 
			args.document_dir, 
			args.index_dir, 
			debug=args.debug,
			auto_adjust_chunks=True,
			chars_per_dimension=DEFAULT_CHARS_PER_DIMENSION,
			use_sqlite=args.use_sqlite
		)
		if not success:
			print_error("Indexing failed. Exiting.")
			sys.exit(1)
							
	# Get index path for the specified project
	index_path, backup_dir = get_index_path(args.index_dir, args.project)
	
	# Check if the index exists (differently for SQLite vs pickle files)
	index_exists = False
	if args.use_sqlite:
		# For SQLite, check if the database file exists
		try:
			from sqlite_storage import get_db_path
			db_path = get_db_path(args.index_dir, args.project)
			index_exists = os.path.exists(db_path)
			if args.debug:
				print_debug(f"Checking for SQLite database at: {db_path}")
				print_debug(f"SQLite database exists: {index_exists}")
		except ImportError as e:
			print_error(f"Error importing sqlite_storage module: {e}")
			print_error("Make sure sqlite_storage.py is in your current directory")
			sys.exit(1)
	else:
		# For pickle files, check if the index file exists
		index_exists = os.path.exists(index_path)
		if args.debug:
			print_debug(f"Checking for pickle index at: {index_path}")
			print_debug(f"Pickle index exists: {index_exists}")
	
	# Now use index_exists for the rest of the logic
	if not index_exists:
		print_system(f"Index for project '{HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}{SYSTEM_COLOR}' not found")
		storage_type = "SQLite database" if args.use_sqlite else "pickle index"
		
		# Ask if user wants to create it
		create = input(f"{SYSTEM_COLOR}Would you like to create and index project '{HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}{SYSTEM_COLOR}' using {storage_type}? (y/n): {RESET_COLOR}").strip().lower()
		if create == 'y':
			# Create the project directory if needed
			if args.project != MASTER_PROJECT:
				project_dir = os.path.join(args.document_dir, args.project)
				os.makedirs(project_dir, exist_ok=True)
				print_system(f"Created project directory: {project_dir}")
			
			# Index the project
			success = index_project(args.project, args.document_dir, args.index_dir, args.debug, use_sqlite=args.use_sqlite)
			if not success:
				print_error("Indexing failed. Exiting.")
				sys.exit(1)
		else:
			# List available projects
			projects = discover_projects(args.document_dir, args.index_dir, args.use_sqlite)
			if projects:
				print_system("\nAvailable Projects:")
				for project in projects:
					print_system(f"  {HIGHLIGHT_COLOR}{project}{RESET_COLOR}")
			sys.exit(1)
	
	# Print application info
	print_system(f"RAG Query Application with Project Support")
	print_system(f"Python version: {sys.version}")
	print_system(f"Project: {HIGHLIGHT_COLOR}{args.project}{RESET_COLOR}")
	if args.use_sqlite:
		from sqlite_storage import get_db_path
		db_path = get_db_path(args.index_dir, args.project)
		print_system(f"Index location: {db_path}")
	else:
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
		import anthropic
		print_system(f"Anthropic SDK version: {anthropic.__version__}")
	except (ImportError, AttributeError):
		print_system("Anthropic SDK version: unknown")
	
	# Load document index for the project
	if args.use_sqlite:
		from sqlite_storage import load_documents, document_count
		# With SQLite, we don't load everything into memory at once
		# Just get a count to show to the user
		doc_count = document_count(args.index_dir, args.project)
		print_system(f"Using SQLite storage with {doc_count} documents")
		
		# We'll create an empty list as a placeholder
		documents = []
		if doc_count > 0:
			documents = load_documents(args.index_dir, args.project, limit=50, debug=args.debug)
			if args.debug:
				print_debug(f"Loaded {len(documents)} sample documents for compatibility")
		else:
			print_system("No documents found in the SQLite database.")
			
	else:
		# Use existing pickle file functions
		documents = load_index(index_path, backup_dir, args.debug)
	
	if not documents and not args.use_sqlite:
		print_error(f"No documents found in the project index. Please add documents and run `index`.")
	
	# Display information about the embeddings in the index
	if args.debug and documents:
		embedding_types = {}
		for doc in documents:
			emb_type = doc.metadata.get('embedding_type', 'unknown')
			emb_model = doc.metadata.get('embedding_model', 'unknown')
			key = f"{emb_type}/{emb_model}"
			embedding_types[key] = embedding_types.get(key, 0) + 1
		
		print_debug("\nEmbedding types in index:")
		for key, count in embedding_types.items():
			print_debug(f"  {key}: {count} documents")
	
	# Load project configuration to get rag_count
	project_config = load_project_config_file(args.project, args.document_dir)
	rag_settings = project_config.get("rag", {})
	
	# Determine the rag_count to use, in order of precedence:
	# 1. Command line argument (--rag-count)
	# 2. Project configuration
	# 3. Default value (TOP_K_DOCUMENTS)
	rag_count = args.rag_count
	if rag_count is None:
		rag_count = rag_settings.get("rag_count", TOP_K_DOCUMENTS)
	
	if args.debug:
		print_debug(f"Using rag_count: {rag_count}")
		if args.rag_count is not None:
			print_debug("Source: Command line argument")
		elif "rag_count" in rag_settings:
			print_debug("Source: Project configuration")
		else:
			print_debug("Source: Default value")
			
	# Handle prompt file
	if args.prompt_file:
		if args.query:
			print_error("Cannot use both --query and --prompt-file")
			sys.exit(1)
		
		if args.debug:
			print_debug(f"Reading query from file: {args.prompt_file}")
		
		args.query = read_query_from_file(args.prompt_file)
		print_system(f"Using query from file: {args.prompt_file}")
	
	
	# In the main() function, after loading documents and before the if args.query block:
	
	# Handle prompt file
	if args.prompt_file:
		if args.query:
			print_error("Cannot use both --query and --prompt-file")
			sys.exit(1)
		
		if args.debug:
			print_debug(f"Reading query from file: {args.prompt_file}")
		
		args.query = read_query_from_file(args.prompt_file)
		print_system(f"Using query from file: {args.prompt_file}")
	
	# Replace the if args.query block with this:
	if args.query:
		# Single query mode
		# Echo the query
		print(f"{QUERY_COLOR}{args.query}{RESET_COLOR}\n")
		
		# Get RAG mode from project config
		rag_mode = rag_settings.get("rag_mode", "chunk")
		
		# Get system prompt from args or project config
		system_prompt = args.system_prompt
		if not system_prompt:
			system_prompt = rag_settings.get("system_prompt")
			
			# If no system prompt in rag settings, check system section
			if not system_prompt:
				model_name = args.model or get_model_name_for_llm_type(args.llm)
				system_settings = project_config.get("system", {})
				if model_name in system_settings:
					system_prompt = system_settings[model_name].get("system_prompt")
		
		# In 'none' RAG mode, we skip document search
		if rag_mode.lower() == "none":
			relevant_docs = []
			if args.debug:
				print_debug(f"Using RAG mode 'none' - skipping document search")
		else:
			print_system("Searching for relevant documents...")
			
			# Search for relevant documents
			relevant_docs = search_documents(
				args.query, documents, args.project, args.document_dir, 
				embedding_config, rag_count, args.debug, provider_cache,
				rag_mode, args.use_sqlite, args.index_dir
			)
		
		# Check if batch mode is requested
		if args.batch_models:
			# Parse model list
			models = [m.strip() for m in args.batch_models.split(',')]
			if args.debug:
				print_debug(f"Batch mode: processing {len(models)} models")
				print_debug(f"Models: {', '.join(models)}")
			
			# Map model names to LLM types
			llm_types = {}
			
			# Process in batch mode
			batch_results = batch_process(
				args.query, models, relevant_docs, api_key, args.project, llm_types,
				args.debug, prompts_dir, rag_mode, args.document_dir, system_prompt,
				args.batch_output
			)
			
			# Print summary
			print_system("\nBatch processing complete")
			print_system(f"Models processed: {len(models)}")
			print_system(f"Results saved to: {args.batch_output or batch_results.get('output_file', 'batch_results.json')}")
			
		else:
			# Regular single-model processing
			# Check if we should use the model specified in command line args
			model_name = args.model
			if not model_name:
				# If not provided, get from project config
				if args.llm == LLM_CLAUDE:
					model_name = rag_settings.get("llm_model", DEFAULT_CLAUDE_MODEL)
				elif args.llm == LLM_OPENAI:
					model_name = rag_settings.get("llm_model", DEFAULT_OPENAI_MODEL)
				elif args.llm == LLM_LOCAL:
					model_name = rag_settings.get("llm_model", DEFAULT_LOCAL_MODEL)
				elif args.llm == LLM_HF:
					model_name = rag_settings.get("llm_model", DEFAULT_HF_MODEL)
				else:
					model_name = get_model_name_for_llm_type(args.llm)
			
			# Get response from LLM
			print_system(f"Generating answer with {args.llm}:{model_name} (RAG mode: {rag_mode})...")
			
			# Get response
			answer = get_response(
				args.query, relevant_docs, api_key, args.project,
				args.llm, model_name, args.debug, prompts_dir,
				rag_mode, args.document_dir, system_prompt
			)
			
			# Print the answer
			print(f"\n{ANSWER_COLOR}Answer:{RESET_COLOR}")
			print(f"{ANSWER_COLOR}{answer}{RESET_COLOR}")
	
	
	else:
		# Interactive mode
		interactive_mode(
			documents, api_key, args.project, args.document_dir, args.index_dir,
			embedding_config, args.debug, prompts_dir, args.llm,
			args.local_model, args.hf_model, args.model or DEFAULT_CLAUDE_MODEL,
			args.history_dir, rag_count, args.system_prompt, args.use_sqlite
		)


if __name__ == "__main__":
	main()