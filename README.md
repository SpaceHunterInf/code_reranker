# Code Reranker

A straightforward implementation of CodeRAG for question answering task.


## Features

- **Natural Language Queries**: Ask questions in plain English about code functionality
- **Repository Indexing**: Automatically indexes and builds search capabilities for any GitHub repository. Support Sentence Transformer and Google GenAI API for document embeddings.
- **Query Expansion and Summary**: Support VLLM backend for local inference on expanding queries and code document summarization. API support can be added easily.
- **Evaluation Mode**: Measure search performance using standard metrics like Recall@K
- **Interactive Mode**: Run as an interactive CLI tool for on-demand queries

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

To use Code Reranker interactively to query a repository:

```bash
python main.py --repo https://github.com/viarotel-org/escrcpy
```

You'll be prompted to enter natural language queries, and the tool will return the most relevant code files.

### Evaluation Mode

To evaluate the performance of Code Reranker against a test set at Recall@10 with provided testing example:

```bash
python main.py --config eval_config.json --eval escrcpy-commits-generated.json 
```
Achieved Average Recall@10: 0.7412 within 54 s with a RTX Quadro 8000 48GB, with `gtr-t5-xl` as baseline

### Command-Line Arguments

- `--repo`: GitHub repository URL to analyze (default: https://github.com/viarotel-org/escrcpy)
- `--eval`: Path to evaluation data JSON file
- `--k`: Number of results to return (default: 10)
- `--config`: Path to configuration file
- `--update`: Force update of repository and index
- `--model`: Embedding model to use

## Configuration

By default we use `gtr-t5-xl` for document embeddings and `meta-llama/Meta-Llama-3-8B-Instruct` for query expansion and summarization
You can customize Code Reranker through a configuration file. Create a JSON file with the following structure:

```json
{
    "github_url": "https://github.com/your/repository",
    "embedding_model": "model_name",
    "update": false
}
```

Use the `--config` flag to specify your configuration file. Below are options for config keys:
- `github_url` repo url, if you don't want to feed it by cmd
- `embedding_provider` pick `sentence_transformer` or `google`
- `embedding_model` e.g., `sentence-transformers/gtr-t5-xl` or `microsoft/codebert-base`
- `save_dir` cache dir
- `update` `true` if you want to pull the directory and update index
- `helper_model` model name for code summarisation and query expantion 
- `query_expansion` `true` if you would like to expand queries
- `summary` `true` if you want a summary to be generated for every file retrieved
- `helper_type` `vllm` default for query and summary generation. Take a look at `helper_util/vllm_helper.py` to get more vllm specific hyperparameters.
- `chat_mode` `true` if you want to use chat_template for helper llm.


`config.json` will give you 

## How It Works

1. Code Reranker clones the target repository
2. It processes the code files and generates embeddings using the specified model
3. When you ask a question, it:
   - Converts your question to an embedding vector
   - Finds the most similar code files using vector similarity
   - Returns a ranked list of relevant files
4. It saves the embedding and index cache until you pass ` "update": true ` in the config file.

## Examples

`python main.py --config config.json -k 1`

Sample output:
##
```
Enter your question (Ctrl+C to exit): How does the SelectDisplay component handle the device options when retrieving display IDs?
Processed prompts: 100%|███████████████████████████████████████████| 1/1 [00:14<00:00, 14.20s/it, est. speed input: 4.01 toks/s, output: 36.05 toks/s]
Expanded queries: ['How does the SelectDisplay component handle the device options when retrieving display IDs?', '1. "SelectDisplay component device options"', '2. "Retrieve display IDs device options"']
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.61it/s]
Processed prompts: 100%|██████████████████████████████████████████| 1/1 [00:01<00:00,  1.62s/it, est. speed input: 43.94 toks/s, output: 35.28 toks/s]
File: src/components/PreferenceForm/components/SelectDisplay/index.vue
Summary: <|start_header_id|>assistant<|end_header_id|>

This is a Vue.js component file named `SelectDisplay.vue` located in the `PreferenceForm` directory within the `src/components` directory. The component likely renders a dropdown select field with options, allowing users to make a selection from a list of values.


Top 1 relevant files:
1. src/components/PreferenceForm/components/SelectDisplay/index.vue
```