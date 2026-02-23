# Examples

Example notebooks and model artifacts for the BookDB project.

## Structure

```
examples/
├── marimo/           # Interactive Marimo notebook examples
│   ├── basic_marimo_example.py      # MNIST neural network tutorial
│   └── amazon_data_marimo_example.py # Polars data analysis tutorial
├── models/           # Model training/experiment notebooks
│   └── model1.ipynb  # Template/experimental notebook
└── template.ipynb    # Jupyter notebook template (legacy)
```

## Marimo Notebooks

See [marimo/README.md](marimo/README.md) for details on running interactive Marimo notebooks.

> **Note:** This project uses [Marimo](https://marimo.io/) instead of Jupyter notebooks. See [`.agents/docs/marimo.md`](../.agents/docs/marimo.md) for local documentation.

## Running Marimo Examples

```bash
# From project root
marimo edit examples/marimo/basic_marimo_example.py

# Or run the Amazon data analysis
marimo edit examples/marimo/amazon_data_marimo_example.py
```

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| `marimo/basic_marimo_example.py` | MNIST neural network tutorial demonstrating Marimo features |
| `marimo/amazon_data_marimo_example.py` | Polars data analysis tutorial with Amazon product data |
| `models/model1.ipynb` | Template/experimental Jupyter notebook |
