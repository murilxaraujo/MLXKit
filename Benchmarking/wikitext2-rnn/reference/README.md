# WikiText-2 vanilla RNN — PyTorch reference

[`train.py`](train.py) is a standalone benchmark runner modeled on upstream
`src/rnn_pl.py`, but uses a reproducible tokenizer/vocabulary manifest and a
train-only vocabulary. The environment pins Python 3.11, NumPy 1.26.4,
NLTK 3.9.1, and PyTorch 2.6.0.

From the repository root:

```sh
conda env create -f Benchmarking/wikitext2-rnn/reference/environment.yml
Benchmarking/wikitext2-rnn/scripts/run-reference.sh
```

The runner selects PyTorch MPS when available, records the active backend, and
falls back to CPU otherwise. See the workload README for dataset provenance and
source deviations.
