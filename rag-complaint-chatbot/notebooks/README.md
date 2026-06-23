
These files are written in [Jupytext "light" format](https://jupytext.readthedocs.io/)
— plain `.py` files with `# %%` cell markers — so they're easy to diff/review
in git, but open and run exactly like a notebook in VS Code or Jupyter.

- `01_eda_preprocessing.py` — Task 1: EDA and preprocessing
- `02_chunking_embedding_indexing.py` — Task 2: sampling, chunking, embedding, indexing

## Opening as notebooks

```bash
pip install jupytext
jupytext --to notebook 01_eda_preprocessing.py
jupytext --to notebook 02_chunking_embedding_indexing.py
```

Or in VS Code: open the `.py` file directly — the Jupytext extension (or
VS Code's native cell-marker support) will let you run cells with
`Shift+Enter` as if it were a `.ipynb` file, no conversion needed.
