# CIFAR-10 — ResNet-18

**Status:** scaffold only. Dataset setup, exact ResNet-18 adaptation, reference
framework/version, run commands, and results are pending.

## Workload

- **Model:** ResNet-18 for 10-class image classification. Document any changes
  from the canonical ImageNet stem/classifier so both implementations match.
- **Dataset:** CIFAR-10; source and augmentation policy pending.
- **Quality:** test accuracy, with validation loss as an optional companion.
- **Performance:** training images/second, median step latency, and peak
  memory, measured as specified in [`../README.md`](../README.md).

Align image layout, normalization, augmentations, initialization, optimizer,
batch size, and training budget.

## Implementations

- [Reference implementation](reference/README.md)
- [MLXKit implementation](mlxkit/README.md)
