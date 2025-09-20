# Project Overview

This project is a Python-based implementation of a Transformer language model from scratch, as part of the CS336 course. It includes implementations of core components like Byte Pair Encoding (BPE) for tokenization, and various attention mechanisms (Scaled Dot Product, Multi-head Self Attention, RoPE). The project is structured as a Python package `cs336_basics` with a comprehensive test suite.

## Building and Running

The project uses `uv` for environment management.

### Setup

1.  **Install `uv`:**
    Follow the instructions [here](https://github.com/astral-sh/uv) to install `uv`.

2.  **Download data:**
    ```sh
    mkdir -p data
    cd data
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
    wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
    gunzip owt_train.txt.gz
    wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
    gunzip owt_valid.txt.gz
    cd ..
    ```

### Running Tests

To run the unit tests, use the following command:

```sh
uv run pytest
```

## Development Conventions

*   **Environment Management:** The project uses `uv` to manage dependencies and ensure a consistent development environment.
*   **Testing:** The project has a comprehensive test suite using `pytest`. Tests are located in the `tests/` directory and are structured to test individual components of the language model. The tests use a snapshot-based approach to verify the correctness of the implementations.
*   **Coding Style:** The code is well-structured, uses type hints, and follows the PEP 8 style guide. The `ruff` linter is used to enforce code quality.
*   **Core Libraries:** The project uses `torch` for tensor operations and `einops` for more expressive tensor manipulations.
