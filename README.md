# RAG CLI Application with Claude

A robust command-line Retrieval Augmented Generation (RAG) application that uses Claude and sentence transformers. This application indexes documents with a paragraph-based chunking approach and allows answering questions based on your document collection.

## Features

- **Paragraph-Based Chunking**: Creates natural document chunks while maintaining context
- **Project Support**: Organize documents in subdirectories as separate projects
- **Local Embeddings**: Uses sentence-transformers for generating embeddings locally
- **Claude Integration**: Leverages Claude's powerful AI to answer questions
- **Interactive Mode**: Switch between projects and ask multiple questions in a session

## Requirements

- Python 3.7+
- Anthropic API key
- Documents in TXT or MD format

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pricklypixie/ragdemo.git
cd ragdemo
```

3. Create virtual environment

```bash
conda create -n ragdemo
conda activate ragdemo
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your Anthropic API key as an environment variable:

```bash
# On Linux/macOS
export ANTHROPIC_API_KEY="your-api-key"

# On Windows
set ANTHROPIC_API_KEY=your-api-key
```

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

## Configuration

You can customize these parameters:

- `--index-dir`: Directory to store the index (default: `document_index`)
- `--document-dir`: Directory containing documents (default: `documents`)
- `--embedding-model`: SentenceTransformer model (default: `all-MiniLM-L6-v2`)
- `--max-chunk-size`: Maximum chunk size in characters (default: 1500)

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

## Troubleshooting

- **Installation Issues**: Make sure you have the right Python version and all dependencies installed
- **API Key Issues**: Verify your Anthropic API key is set correctly
- **Memory Issues**: For large document collections, consider indexing projects separately

## Next Steps

   - Create way to access the relevant source documents
   - Use alternative embedding / indexing models
   - Have choice of LLMs for answer questions

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