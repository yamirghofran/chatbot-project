# Marimo Examples

Interactive [Marimo](https://marimo.io/) notebook examples demonstrating different use cases.

## Installation

```bash
uv pip install marimo torch torchvision polars altair
```

## Examples

### Basic Example - MNIST Neural Network

`basic_marimo_example.py`

Trains a neural network on MNIST using PyTorch. Dataset downloads automatically on first run.

```bash
marimo edit examples/marimo/basic_marimo_example.py
```

### Amazon Furniture Dataset - Polars Tutorial

`amazon_data_marimo_example.py`

Demonstrates Polars operations using an Amazon product dataset.

**Setup required:** This example requires a CSV dataset that is not included in the repository.

Place your CSV file at:
```
data/furniture_amazon_dataset_sample.csv
```

Expected columns:
- asin, title, brand, price, availability, categories, color, material, country_of_origin, date_first_available

```bash
marimo edit examples/marimo/amazon_data_marimo_example.py
```
