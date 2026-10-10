# User Guide

## Requirements

- Python 3.14 or newer
- Poetry

## Installing the Project

Clone the repository and enter its directory:

```bash
git clone https://github.com/BoddyGus/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

Install the project dependencies:

```bash
poetry install
```

## Running Tests

Run all tests from the project root:

```bash
poetry run pytest
```

Run a specific test file by replacing <test-file> with its path:

```bash
poetry run pytest src/tests/<test-file>
```

For example:
```bash
poetry run pytest src/tests/test_network.py
```

## Running Training

Run the training from the project root:
```bash
poetry run python -m digits.train
```
