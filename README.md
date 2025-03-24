# RAG CLI Application with Claude

A robust command-line Retrieval Augmented Generation (RAG) application that uses Claude and sentence transformers. This application indexes documents with a paragraph-based chunking approach and allows answering questions based on your document collection.

## Features

- **Paragraph-Based Chunking**: Creates natural document chunks while maintaining context
- **Project Support**: Organize documents in subdirectories as separate projects
- **Local Embeddings**: Uses sentence-transformers for generating embeddings locally
- **Claude Integration**: Leverages Claude's powerful AI to answer questions
- **Interactive Mode**: Switch between projects and ask multiple questions in a session

## Requirements

- Python 3.11.5
- Anthropic API key (optional)
- OpenAI API key (optional)
- Huggingface API key (optional)
- Documents in TXT or MD format

## Installation


1. Create virtual environment

```bash
conda create -n ragdemo
conda activate ragdemo
```

2. Clone the repository:

```bash
git clone https://github.com/pricklypixie/ragdemo.git
# for development branch
# git clone -b development https://github.com/pricklypixie/ragdemo.git
cd ragdemo
```


3. Install dependencies:

```bash
conda install python==3.11.5
pip install -r requirements.txt
```

4. Set up your Anthropic API key as an environment variable (optional - local models work)

```bash
# On Linux/macOS
export ANTHROPIC_API_KEY="your-api-key"

# On Windows
set ANTHROPIC_API_KEY=your-api-key
```

5. For using OpenAI embeddings or other Sentence Transformer embeddings, set the appropriate keys:

```bash
export HF_TOKEN=your-huggingface-token
export OPENAI_API_KEY=your-openai-key


6. To use on device models (optional - test first)

```bash
# Then install an LLM model provider (e.g., GPT4All)
## Not needed, in requirements file (Double check not needed).
llm install gpt4all

# Option 2: Install transformers
## In requirements file
pip install transformers torch
```

### To work on development branch

```bash
conda create -n ragdemo
conda activate ragdemo

git clone -b development https://github.com/pricklypixie/ragdemo.git
cd ragdemo

# when some new development changes are made
git pull


## Usage

### Setting Up Your Documents

Place your documents in the `documents` directory:

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

- Files in the root directory belong only to the "master" project
- Files in subdirectories belong to their respective project and the master project

### Indexing Documents

Index all documents:

```bash
python document_indexer.py
```

Index with debugging information:

```bash
python document_indexer.py --debug
```

Index a specific project:

```bash
python document_indexer.py --project project1
```

Index a single file:

```bash
python document_indexer.py --file documents/file1.txt
```

List available projects:

```bash
python document_indexer.py --list-projects
```

### Querying the Index

Interactive mode:

```bash
python rag_query.py
```

Query a specific project:

```bash
python rag_query.py --project project1
```

Single query mode:

```bash
python rag_query.py --query "What information is in my documents?"
```

Debug mode:

```bash
python rag_query.py --debug
```

### Interactive Commands

When in interactive mode, you can use these commands:

- `projects` - List all available projects
- `project <name>` - Switch to a different project
- `exit` or `quit` - Exit the application

To manage the LLM used to answer the queries:
- `models` - List all available LLMs
- `llm claude` - Use Claude API using local api key
- `llm local [model_name]` - Use a local model with llm library
- `llm hf [model_name]` - Use a Hugging Face model

- `system prompt "prompt"` - Set the system prompt
- `system prompt show` - Show the current system prompt
- `system prompt clear` - Clear the system prompt

To manage defaults (useful when experimenting with different RAG settings):
- `defaults save` - Save the defaults for a project (loaded when moving to the project)
- `defaults read` - Read the faults for a project

## Configuration

You can customize these parameters:

- `--index-dir`: Directory to store the index (default: `document_index`)
- `--document-dir`: Directory containing documents (default: `documents`)
- `--embedding-model`: SentenceTransformer model (default: `all-MiniLM-L6-v2`)
- `--max-chunk-size`: Maximum chunk size in characters (default: 1500)

## Project-Specific Embedding Configuration

Each project can have its own embedding configuration by adding an `embedding_config.json` file to the project directory:

```json
{
  "embedding_type": "sentence_transformers",
  "model_name": "all-MiniLM-L6-v2",
  "api_key": null,
  "additional_params": {}
}

### Supported Embedding Types

1. **sentence_transformers** (default)
   - Local embedding generation using sentence-transformers
   - Recommended models: "all-MiniLM-L6-v2", "all-mpnet-base-v2"
   - No API key required

2. **openai**
   - OpenAI's embedding API
   - Recommended models: "text-embedding-3-small", "text-embedding-3-large"
   - Requires OpenAI API key (set via OPENAI_API_KEY environment variable)
   
For experimentation, there we see no benefit from running remote embedding models, unless there is a particular need for much larger embedding sizes.

### Example Configurations

#### For multilingual documents:
```json
// A HuggingFace User Access Tokens may be required to download
// some sentence_transformers models
// https://huggingface.co/sentence-transformers
// 
{
  "embedding_type": "sentence_transformers",
  "model_name": "paraphrase-multilingual-mpnet-base-v2"
}
```

#### For OpenAI embeddings:
```json
{
  "embedding_type": "openai",
  "model_name": "text-embedding-3-small"
}
```

Note: Combining documents with different embedding models in the same index is supported. The system will handle the differences during search (untested).

## Mac-Specific Notes

This application has been optimized for macOS, including Apple Silicon Macs. It uses CPU for embeddings to avoid Metal compatibility issues.

## How It Works

1. **Document Indexing**:
   - Documents are split into paragraphs
   - Paragraphs are grouped into chunks with overlap between chunks
   - Each chunk is embedded using sentence-transformers
   - Embeddings and chunks are stored in separate project indexes

2. **Query Processing**:
   - User query is embedded using the same model
   - Semantic search finds relevant document chunks
   - Claude is prompted with the query and relevant chunks
   - Claude provides an answer based on the provided context
   
### Creating a New Project:
   
```bash
python rag_query.py --project new_project --index
```

This will:
1. Create the project directory if it doesn't exist
2. Create a default embedding configuration if none exists
3. Index all files in the project directory
4. Save the index for future queries

### Querying with Detailed Document Information:

```bash
python rag_query.py --project sample --query "What is the main topic?" --debug
```

## Experimenting with different local models

Easiest to experiment with the LLM.

For example, to use the excellent OLMo 2 model from the Allen institute (https://allenai.org/blog/olmo2-32B):

```bash
llm install llm-mlx
llm mlx download-model mlx-community/OLMo-2-0325-32B-Instruct-4bit
```

And then within the application:

llm local mlx-community/OLMo-2-0325-32B-Instruct-4bit


## Troubleshooting

- **Installation Issues**: Make sure you have the right Python version and all dependencies installed
- **API Key Issues**: Verify your Anthropic API key is set correctly
- **Memory Issues**: For large document collections, consider indexing projects separately

## Next Steps

   - Create way to access the relevant source documents
   
## Issues

   - Search can return the same document three times (if three chunks from the same document have the highest scores). Should be changed to return the three highest scoring documents.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


MIT License

Copyright (c) 2023 Prickly Pixie Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.