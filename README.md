# Semantic Search Engine

A lightweight, extensible toolkit for preparing and handling textual data for semantic search and downstream NLP
tasks.  
The repository bundles a collection of utilities for:

* Converting Doccano exports to formats ready for sequence‑ or token‑classification models.
* Transforming plain‑text corpora into JSON/JSONL datasets, with optional chunking, language detection and
  preprocessing.
* Managing model configurations (e.g., denoiser models) and interacting with AWS S3 storage.
* A minimal Django entry‑point (`manage.py`) for optional web or API components.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Command‑line Tools](#commandline-tools)
    - [Doccano Converter](#doccano-converter)
    - [Any‑Text‑to‑JSON Converter](#anytexttojson-converter)
6. [AWS Handler](#aws-handler)
7. [Running the Django Management Script](#running-the-django-management-script)
8. [License](#license)

---

## Project Structure

```
semantic-search-engine/
├── configs/                # Configuration files (e.g. AWS credentials)
├── scripts/                # Helper scripts (optional)
├── semantic_search_engine/ # Core package (models, utils, handlers)
│   ├── __init__.py
│   ├── constants.py
│   ├── aws_handler.py
│   └── models.json
├── doccano_converter.py   # Doccano export → dataset conversion
├── any_text_to_json.py    # Directory → JSON/JSONL conversion
├── manage.py              # Django entry point (optional)
└── README.md              # ← you are reading it!
```

---

## Prerequisites

| Tool                           | Minimum Version                    |
|--------------------------------|------------------------------------|
| Python                         | **3.11** (rc1 is fine)             |
| pip                            | latest                             |
| (Optional) Django              | 4.x                                |
| (Optional) boto3               | for AWS integration                |
| (Optional) radlab_data package | required by the conversion scripts |

All other dependencies are listed in `requirements.txt` (or can be installed via `pip install -r requirements.txt`).

---

## Installation

```shell script
# Clone the repository
cd semantic-search-engine
git clone https://github.com/radlab-dev-group/semantic-search-engine.git

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

If you plan to use the Django utilities:

```shell script
pip install django
```

---

## Configuration

### AWS

The `AwsHandler` expects a JSON configuration file placed in `configs/` (default filenames are defined in
`semantic_search_engine/constants.py`).  
A minimal example (`configs/aws_config.json`):

```json
{
  "aws": {
    "REGION_NAME": "us-east-1",
    "ENDPOINT_URL": "https://s3.amazonaws.com",
    "ACCESS_KEY_ID": "<your-access-key>",
    "SECRET_ACCESS_KEY": "<your-secret>",
    "STORAGE_BUCKET_NAME": "my-semsearch-bucket"
  }
}
```

The handler validates that all required fields are present and will raise an assertion error if anything is missing.

### Model Configuration

`semantic_search_engine/models.json` holds model metadata, e.g.:

```json
{
  "denoiser": {
    "model_name": "radlab/polish-denoiser-t5-base",
    "model_path": "/mnt/data2/llms/models/radlab-open/denoiser/radlab-denoiser-plt5-base-v2",
    "device": "cuda:0"
  }
}
```

You can edit the JSON to point to a different model or device.

---

## Command‑line Tools

Both conversion utilities share a similar CLI pattern based on `argparse`.  
Run `python <script>.py -h` for the full help message.

### Doccano Converter

Converts Doccano exports into datasets suitable for **sequence classification** or **token classification** tasks.

#### Example – Sequence Classification

```shell script
python doccano_converter.py \
    -I /path/to/doccano/export \
    -e .jsonl \
    --show-class-labels-histogram \
    -O prepared_datasets/seq_class/20231208 \
    --sequence-classification
```

#### Example – Token Classification

```shell script
python doccano_converter.py \
    -I /path/to/doccano/export \
    -e .jsonl \
    --show-class-labels-histogram \
    -O prepared_datasets/token_class/20231208 \
    --token-classification
```

Key flags:

| Flag                            | Description                                     |
|---------------------------------|-------------------------------------------------|
| `-I` / `--input-dir`            | Directory containing Doccano export files       |
| `-e` / `--dataset-extension`    | Extension of source files (default: `.jsonl`)   |
| `-O` / `--output-dir`           | Destination for the prepared dataset            |
| `--sequence-classification`     | Produce a dataset for text‑level classification |
| `--token-classification`        | Produce a dataset for token‑level labeling      |
| `--show-class-labels-histogram` | Prints a histogram of class label frequencies   |

### Any‑Text‑to‑JSON Converter

Walks through a directory of plain text files and writes a unified JSON (or JSONL) file. Supports optional chunking,
overlapping tokens, language detection, and cleaning.

#### Basic usage

```shell script
python any_text_to_json.py \
    -d /path/to/texts \
    -o /path/to/output/dataset.json
```

#### Advanced options

| Option                              | Description                                                                  |
|-------------------------------------|------------------------------------------------------------------------------|
| `--proper-pages`                    | Merge all texts belonging to the same page into a single entry               |
| `--merge-document-pages`            | Merge all pages of a document into one record                                |
| `--clear-texts`                     | Apply `radlab_data` cleaning pipeline before saving                          |
| `--split-to-max-tokens-in-chunk N`  | Split each document into chunks of *N* tokens                                |
| `--overlap-tokens-between-chunks M` | Overlap *M* tokens between consecutive chunks (requires the previous option) |
| `--check-text-language`             | Detect language of each document and store it in metadata                    |
| `--processes K`                     | Parallelise processing with *K* worker processes                             |

The script automatically detects whether the output file ends with `.json` (single JSON object) or any other extension (
treated as JSONL, one record per line).

---

## AWS Handler

`semantic_search_engine/aws_handler.py` provides a thin wrapper around **boto3** for common S3 operations:

* `mkdir(path)` – create a “directory” (zero‑byte object with a trailing slash).
* `rm(path)` – delete an object or a pseudo‑directory.
* `add_file_from_buffer(buffer, dest_path)` – upload raw data.
* `add_file_from_path(local_path, dest_dir)` – read a local file and upload it.
* `ls(dir=None, extensions=None)` – list objects, optionally filtered by file extensions (`json`, `jsonl`).
* `load_file(file_path, file_type=None)` – download and deserialize JSON/JSONL files.

Typical usage pattern:

```python
from semantic_search_engine.aws_handler import AwsHandler

aws = AwsHandler()  # Loads config from configs/aws_config.json
aws.mkdir('datasets/')  # Create a bucket “folder”
aws.add_file_from_path('data.json', 'datasets/')
files = aws.ls('datasets/', extensions=['json'])
print('Bucket contents:', files)

# Load a JSON file back into Python
data = aws.load_file('datasets/data.json')
```

All methods expose a `last_error` attribute for troubleshooting and automatically log failures.

---

## Running the Django Management Script

If the project includes a Django component, the standard `manage.py` entry point is provided:

```shell script
export DJANGO_SETTINGS_MODULE=main.settings   # Adjust if your settings module differs
python manage.py runserver                     # Start the development server
python manage.py migrate                       # Apply migrations
```

The script simply forwards the command‑line arguments to Django’s `execute_from_command_line`.

---

## License

This project is licensed under the **Apache 2.0 License** – see the `LICENSE` file for details.

---  

*Happy coding! 🚀*  