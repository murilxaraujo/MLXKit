# MNIST CNN — PyTorch reference

The executable is [`train.py`](train.py). Its Conda environment pins Python
3.11, NumPy 1.26.4, and PyTorch 2.6.0 in [`environment.yml`](environment.yml).
PyTorch's MPS backend is selected when available; the report records the actual
device and falls back to CPU when necessary.

From the repository root:

```sh
conda env create -f Benchmarking/mnist-cnn/reference/environment.yml
Benchmarking/mnist-cnn/scripts/run-reference.sh
```

The model follows the supplied notebook's convolution/pooling/dropout layout.
See the workload README for the canonical MNIST split and other explicit
adaptations made for reproducible pairing.
