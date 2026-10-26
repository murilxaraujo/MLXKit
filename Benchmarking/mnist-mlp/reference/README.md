# MNIST MLP — TensorFlow/Keras reference

The executable reference is [`train.py`](train.py). Its Conda environment pins
Python 3.11, NumPy 1.26.4, TensorFlow 2.16.2, Keras 3.3.3, and TensorFlow Metal
1.1.0 in [`environment.yml`](environment.yml).

From the repository root, create the environment and run the benchmark:

```sh
conda env create -f Benchmarking/mnist-mlp/reference/environment.yml
Benchmarking/mnist-mlp/scripts/run-reference.sh
```

The model retains the notebook's two 256-unit ReLU layers, softmax output,
sparse categorical cross-entropy, batch size 32, 10 epochs, and 48k/12k train
and validation split. Initial weights are loaded from the shared seeded
Glorot-uniform binary files produced by `scripts/prepare-initialization.py`.

The run report records exact package versions, physical and logical TensorFlow
devices, whether TensorFlow Metal is installed, metrics, timing, and memory.
The actual backend is reported; GPU use is not assumed.
