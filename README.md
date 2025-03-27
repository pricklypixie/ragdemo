# RAG Query Application

A versatile Retrieval Augmented Generation (RAG) application that allows you to query your document collection using various Large Language Models (LLMs). This application indexes documents locally, creates embeddings, and uses those to perform semantic search before passing relevant context to an LLM to answer your questions.

## Features

- **Multiple LLM Support**: Use Claude API, OpenAI API, or local models via MLX or Hugging Face
- **Efficient Document Storage**: Choose between pickle files (simple) or SQLite with vector search (scalable)
- **Paragraph-Based Chunking**: Creates natural document chunks while maintaining context
- **Project Organization**: Organize documents in subdirectories as separate projects
- **Local Embeddings**: Generate embeddings locally using sentence-transformers
- **Multiple Interfaces**: Command-line, interactive terminal, or web-based interface
- **RAG Modes**: Choose between chunk-based, file-based, or no RAG
- **Configurable System Prompts**: Customize the LLM's behavior with system prompts

## Requirements

- Python 3.10+ (tested with 3.11.5)
- Apple Silicon Mac for local model support
- API keys (optional):
  - Anthropic API key (for Claude models)
  - OpenAI API key (for GPT models)
- Documents in TXT or MD format

## Installation

1. Create a virtual environment:

```bash
conda create -n ragdemo python=3.11
conda activate ragdemo
```

2. Clone the repository:

```bash
git clone https://github.com/pricklypixie/ragdemo.git

# to test the development branch
# git clone -b development https://github.com/pricklypixie/ragdemo.git

cd ragdemo
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. (optional) For better performance with large document collections, install SQLite with vector search extension:

```bash
# macOS (with Homebrew)
brew install sqlite-vss

# Or install the Python package
pip install sqlite-vec
```

5. (optional) Set up API keys:

```bash
# For Claude API
export ANTHROPIC_API_KEY="your-api-key"

# For OpenAI API
export OPENAI_API_KEY="your-api-key"
```

6. For local models support, install Simon Willison's LLM library:

This should happen as part of the install using requirements.txt.

Check it is working with the following:

```bash 
llm -m Llama-3 "hello. who are you?"
```

The model will download and you should see a reply.

If this doesn't work:

```bash
pip install llm llm-gpt4all
```

## Quick Start

### 1. Add your documents

Place documents in the `documents` directory:
- Files directly in the `documents` folder belong to the "master" project
- Subdirectories form separate projects (e.g., `documents/project1/file.txt`)

```
documents/
├── file1.txt
├── file2.md
├── project1/
│   ├── doc1.txt
│   └── doc2.md
└── project2/
    ├── doc3.txt
    └── doc4.md
```

The application works with text documents only that use .txt or .md as a suffix. If you have documents in other formats, you will need to extract the text and save it as a .txt or .md file.

For more than casual experimentation, best to use project folders and keep the root ("master") documents folder just for projects.

### 2. Launch the interactive terminal

```bash
python rag_query.py
```

### 3. Index and query your documents

Basic workflow in interactive mode:
```
# List available projects
projects

# Switch to a project
project project1

# Index the current project
index

# Ask a question
What information is contained in these documents?
```

The question won't actually work that well, better to ask something directly related to the content of your documents.

### 4. Get help anytime

```
help
```

## Interfaces

### Command Line (Single Query)

```bash
# Basic query using default settings
python rag_query.py --query "What information is in my documents?"

# Query a specific project using Claude
python rag_query.py --project project1 --llm claude --query "Summarize this content"

# Query with OpenAI
python rag_query.py --llm openai --model gpt-4o-mini --query "Analyze these documents"

# Query with a local model
python rag_query.py --llm local --model orca-2-13b --query "Explain the key concepts"
```

As above, best to test with queries related to the documents, rather than general ones.

### Interactive Terminal

```bash
# Start interactive mode
python rag_query.py

# With debug information
python rag_query.py --debug

# Starting with a specific project
python rag_query.py --project project1
```

### Web Interface

```bash
# Start the web server
python api_server.py

# With custom host and port
python api_server.py --host 0.0.0.0 --port 8080
```

Then open your browser to http://localhost:8000/ (or your custom host/port)

## Document Indexing

The indexer can be used as a standalone tool:

```bash
# Index all documents
python document_indexer.py

# Index a specific project
python document_indexer.py --project project1

# Index with SQLite storage
python document_indexer.py --use-sqlite

# Index a single file
python document_indexer.py --file documents/file1.txt

# List available projects
python document_indexer.py --list-projects
```

## Interactive Mode Commands

Interactive mode is launched with:

```bash
python rag_query.py
``` in

In interactive mode, you can use these commands:

### Project Management
- `projects` - List all available projects
- `project <name>` - Switch to a different project
- `config` - Show current project configuration
- `index` - Index the current project
- `index clear` - Clear the current project's index

### LLM Selection
- `models` - List all available LLM models
- `llm claude [model_name]` - Use Claude API
- `llm openai [model_name]` - Use OpenAI API
- `llm local [model_name]` - Use a local model
- `llm hf [model_name]` - Use a Hugging Face model

### RAG Settings
- `rag mode <mode>` - Set RAG mode (chunk, file, none)
- `rag count <number>` - Set number of documents to retrieve

### System Prompts
- `system prompt "prompt"` - Set the system prompt
- `system prompt show` - Show the current system prompt
- `system prompt clear` - Clear the system prompt

### Defaults
- `defaults save` - Save current settings as defaults
- `defaults read` - Load default settings

### Other
- `history` - Show command history
- `history clear` - Clear command history
- `history save` - Save history to a file
- `exit` or `quit` - Exit the application

### SQLite Utilities
- `sqlite verify` - Check if SQLite vector extension is properly installed
- `sqlite inspect` - Show details about the current SQLite database


In interactive mode, `help` will list the available commands.

## Configuration

### Project Configuration

Each project can have its own configuration in a `project_config.json` file. This is a sample.

```json
{
  "indexing": {
    "embedding_type": "sentence_transformers",
    "model_name": "all-MiniLM-L6-v2",
    "api_key": null,
    "additional_params": {}
  },
  "rag": {
    "llm_type": "local",
    "llm_model": "mlx-community/Ministral-8B-Instruct-2410-bf16",
    "rag_mode": "chunk",
    "rag_count": 2,
    "system_prompt": "You are a helpful assistant."
  },
  "defaults": {
    "llm_type": "local",
    "llm_model": "mlx-community/Ministral-8B-Instruct-2410-bf16",
    "rag_mode": "chunk",
    "rag_count": 2,
    "system_prompt": "You are a helpful assistant."
  },
  "system": {
    "orca-2-13b": {
      "system_prompt": "You are a helpful assistant that provides detailed information."
    },
    "claude-3-5-haiku-20241022": {
      "system_prompt": "You are a concise, accurate assistant."
    }
  }
}
```

You can edit this manually, however all settings except for the embedding_type can be managed in interactive mode.

### Supported Embedding Models

1. **sentence_transformers** (default)
   - Local embedding generation using sentence-transformers
   - Recommended models: 
     - "all-MiniLM-L6-v2" (384 dimensions, faster)
     - "all-mpnet-base-v2" (768 dimensions, more accurate)
   - No API key required

2. **openai**
   - OpenAI's embedding API
   - Models: "text-embedding-3-small", "text-embedding-3-large"
   - Requires OpenAI API key

### Supported LLM Models

1. **Claude API** (`--llm claude`)
   - Models: "claude-3-5-haiku-20241022", "claude-3-sonnet-20240229", "claude-3-opus-20240229"
   - Requires Anthropic API key

2. **OpenAI API** (`--llm openai`)
   - Models: "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"
   - Requires OpenAI API key

3. **Local Models** (`--llm local`)
   - Requires Simon Willison's llm library
   - Install models with: `llm install <model_name>` (in the same python environment use use to launch rag_query.py)
   - Popular models:
     - "mlx-community/Ministral-8B-Instruct-2410-bf16"
     - "orca-2-13b"
     - "DeepSeek-R1-Distill-Qwen-14B-Q4_0"

4. **Hugging Face Models** (`--llm hf`)
   - Direct integration with Hugging Face models
   - Examples: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mistralai/Mistral-7B-v0.1"

## RAG Modes

- **chunk** (default): Retrieves the most relevant document chunks
- **file**: Retrieves entire files containing the most relevant chunks
- **none**: Doesn't provide document context (uses LLM's knowledge only)

## SQLite Vector Storage

For large document collections, SQLite with vector search is recommended:

```bash
# Index using SQLite storage
python document_indexer.py --use-sqlite

# Query using SQLite storage
python rag_query.py --use-sqlite
```

## How It Works

1. **Document Indexing**:
   - Documents are split into paragraphs
   - Paragraphs are grouped into chunks with overlap
   - Each chunk is embedded using the configured model
   - Embeddings and chunks are stored in pickle files or SQLite database

2. **Query Processing**:
   - User query is embedded using the same model
   - Semantic search finds relevant document chunks
   - Relevant context is sent to the selected LLM
   - LLM provides an answer based on the provided context

## Troubleshooting

- **Memory Issues**: For large document collections, use SQLite storage (`--use-sqlite`)
- **Local Model Performance**: Local models work best on Apple Silicon Macs with sufficient RAM
- **API Key Issues**: Check your environment variables are set correctly
- **Installation Problems**: Make sure all dependencies are installed

## Tips and Tricks

- Use `--debug` flag to see detailed information about what's happening
- Adjust `rag count` to control how many documents are retrieved
- Try different embedding models for different types of content
- Use system prompts to control the style and tone of responses
- For large projects, SQLite storage is more efficient than pickle files

## Next steps

There is still some development to do:

- Adding project management to the web interdace (e.g., to add documents)

## License

This project is licensed under the MIT License - see the LICENSE file for details.