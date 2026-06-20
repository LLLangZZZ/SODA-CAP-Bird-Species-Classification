# SODA-CAP Bird Audio Classification

This repository is a cleaned project snapshot for the paper:

**SODA-CAP: Source-Oriented Data Augmentation and Class-Adaptive Complementary Policy for Robust Bird Audio Classification**

The code is kept as a method-level reference for the paper rather than a fully runnable training package. Large model checkpoints, prediction caches, intermediate tensors, dataset-cleaning utilities, and local data paths were intentionally removed before GitHub release.

## Method Summary

The paper studies robust classification of 36 resident bird species from noisy field recordings. Audio clips are converted into fixed-size log-Mel filter-bank features and classified with an AST-based model.

The central augmentation is **SODA**, which estimates bird-dominant and noise-dominant components in a spectrogram, perturbs the residual noise style, and recomposes the feature while preserving species-discriminative acoustic structure.

**SODA-CAP** keeps SODA as the dominant augmentation and assigns the remaining probability mass to class-specific auxiliary augmentations. The policy is built from interpretable validation behavior:

```text
Score(c, a) = rescue(c, a) - lambda * damage(c, a) + beta * gain(c, a)
```

where `rescue` measures how often an auxiliary augmentation fixes SODA errors, `damage` measures how often it breaks SODA-correct samples, and `gain` measures class-wise improvement over the baseline.

