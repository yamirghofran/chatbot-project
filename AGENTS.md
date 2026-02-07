## Repo Structure and Rules

- `main` branch is only production code
- `dev` is the branch mostly used

## Tools

- Python
- [uv](https://docs.astral.sh/uv/) instead of pip.
  - Use `uv sync` to sync the dependencies with the remote repo whenever you pull.
- [marimo](https://docs.marimo.io/) notebooks instead of Jupyter notebooks. Local docs/instructions at [local marimo docs](.agents/docs/marimo.md)
- [polars](https://docs.pola.rs/) instead of Pandas
- [zensical](https://zensical.org/docs/get-started/) for documentation
