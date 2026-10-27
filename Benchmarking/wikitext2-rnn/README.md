# WikiText-2 — vanilla RNN language model

This paired PyTorch/MLXKit workload follows the vanilla RNN in
[`liux2/RNN-on-wikitext2`](https://github.com/liux2/RNN-on-wikitext2), pinned to
revision `b2726e6bf08ddca5580e1fced26c57c1fd8e3657`. It benchmarks next-token
language modeling on that repository's train, validation, and test text splits.
The upstream repository is MIT-licensed; see [`UPSTREAM-LICENSE.txt`](UPSTREAM-LICENSE.txt).

## Model and protocol

- Model: embedding (100 dimensions), embedding-output dropout (`p=0.25`), two
  tanh RNN layers (100 hidden units each), and a linear vocabulary projection.
  Hidden state is initialized to zero at each batch and is not carried between
  windows. The model is not an LSTM.
- Data: the three `wiki.{train,valid,test}.txt` files from the pinned upstream
  revision. Their Git blob SHAs and downloaded-file SHA-256 checksums are
  recorded in the prepared-data manifest and each run report.
- Tokenization follows the source's NLTK sentence/word tokenizer, lowercase,
  seeded sentence shuffle, `<s>`/`</s>` boundary markers, and terminal
  punctuation handling. The vocabulary is built from the training split only;
  unseen validation/test words map to `<unk>`.
- Sequences: split-local, non-overlapping windows of 31 tokens; 30 input tokens
  predict the next 30 tokens. Incomplete final windows are dropped. Batch size
  is 20 and the final incomplete batch is dropped, matching the source.
- Optimizer: SGD, learning rate `0.5`, no momentum, and no gradient clipping.
  The source's vanilla RNN has no configured gradient clipping (the separate
  LSTM script uses clipping). Epoch count is 20 by default.
- Initialization: shared seeded float32 weights. The embedding uses uniform
  `[-0.1, 0.1]`; PyTorch-style recurrent and output parameters use uniform
  `[-1/sqrt(100), +1/sqrt(100)]`. Both implementations load the same files.
- Quality: token-averaged cross-entropy and perplexity (`exp(loss)`) on
  validation and test sequences. Evaluation excludes incomplete final batches,
  as the source data module does.

### Reproducibility and source corrections

The upstream `rnn_pl.py` shuffles sentences and its data loader shuffles
training windows. Here those shuffles are seeded: preprocessing is fixed for a
given seed and each training epoch has a distinct deterministic window
permutation. Both runners use MLXKit's SplitMix64 Fisher–Yates index shuffle,
so they consume the same training-window order. The upstream script also
accumulates the vocabulary from train, validation, and test, which leaks
held-out information. This benchmark uses a train-only vocabulary with an
explicit `<unk>` ID. These corrections make the benchmark reproducible and keep
evaluation data held out.

The MLX implementation uses a batch/time/token tensor layout and custom
unrolled tanh recurrent layers, with separate input and recurrent projections
to match PyTorch's two bias vectors per layer. PyTorch uses its native
`nn.RNN` implementation. Dropout random streams and backend kernels differ, so
identical starting weights do not imply identical training trajectories.

## Setup and run

Requirements: Apple-silicon Mac, macOS 27 or later, Xcode/Swift 6.4 or later,
and Conda. Create the Python environment from the repository root:

```sh
conda env create -f Benchmarking/wikitext2-rnn/reference/environment.yml
```

Run each implementation separately with the same run ID and options:

```sh
RUN_ID=wikitext2-rnn-pair-01 Benchmarking/wikitext2-rnn/scripts/run-reference.sh
RUN_ID=wikitext2-rnn-pair-01 Benchmarking/wikitext2-rnn/scripts/run-mlxkit.sh
```

The first run downloads and prepares the pinned corpus under
`Benchmarking/wikitext2-rnn/data/`. Reports, logs, and per-epoch CSVs are saved
under `Benchmarking/wikitext2-rnn/results/<run-id>/`. The reference uses MPS
when available and otherwise records CPU; MLX uses its default device.

Both runners accept `--epochs`, `--seed`, `--batch-size`, `--warmup-steps`, and
`--max-train-batches`. Use a small bounded check before full training:

```sh
RUN_ID=wikitext2-rnn-smoke Benchmarking/wikitext2-rnn/scripts/run-reference.sh \
  --epochs 1 --max-train-batches 2
RUN_ID=wikitext2-rnn-smoke Benchmarking/wikitext2-rnn/scripts/run-mlxkit.sh \
  --epochs 1 --max-train-batches 2
```

The bounded limit is for execution checks only; do not treat it as a full
training result. Reports record token throughput, median/P90 steady-state batch
latency, validation/test loss and perplexity, peak process RSS, and framework
memory counters where available. PyTorch MPS allocation counters and MLX active
memory are different measurements; compare process RSS for a like-for-like
memory view. Timed device steps are synchronized; dataset preparation and
initial model evaluation are outside the training timer.

```sh
Benchmarking/wikitext2-rnn/scripts/cleanup.sh
```

Add `--data` to remove the downloaded corpus and processed token cache as well.
The Conda environment and NLTK resources are never removed.

## Implementations

- [`reference/`](reference/README.md): PyTorch 2.6.0 reference.
- [`mlxkit/`](mlxkit/README.md): `WikiText2RNNBenchmark` executable.
- [`scripts/`](scripts/): source download, token preparation, shared
  initialization, runners, and cleanup.
