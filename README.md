# SODA-CAP Bird Audio Classification

This repository is a cleaned project snapshot for the paper:

**SODA-CAP: Source-Oriented Data Augmentation and Class-Adaptive Complementary Policy for Robust Bird Audio Classification**

The code is kept as a method-level reference for the paper rather than a fully runnable training package. Large model checkpoints, prediction caches, intermediate tensors, dataset-cleaning utilities, and local data paths were intentionally removed before GitHub release.

## What Is Included

- `models/ast_model.py` - Audio Spectrogram Transformer classifier used as the spectrogram backbone.
- `augmentations.py` - SODA and auxiliary audio/spectrogram augmentation operators.
- `soda_cap/augment.py` - SODA-CAP training-time policy augmenter.
- `soda_cap/build_policy.py` - class-wise complementary policy construction from cached validation behavior.
- `soda_cap/plot_class_strategy_evolution*.py` - scripts used to visualize class-wise augmentation distributions.

## Method Summary

The paper studies robust classification of 36 resident bird species from noisy field recordings. Audio clips are converted into fixed-size log-Mel filter-bank features and classified with an AST-based model.

The central augmentation is **SODA**, which estimates bird-dominant and noise-dominant components in a spectrogram, perturbs the residual noise style, and recomposes the feature while preserving species-discriminative acoustic structure.

**SODA-CAP** keeps SODA as the dominant augmentation and assigns the remaining probability mass to class-specific auxiliary augmentations. The policy is built from interpretable validation behavior:

```text
Score(c, a) = rescue(c, a) - lambda * damage(c, a) + beta * gain(c, a)
```

where `rescue` measures how often an auxiliary augmentation fixes SODA errors, `damage` measures how often it breaks SODA-correct samples, and `gain` measures class-wise improvement over the baseline.

## Repository Layout

```text
.
├── augmentations.py
├── config/
│   ├── __init__.py
│   └── config.py
├── models/
│   ├── __init__.py
│   └── ast_model.py
├── soda_cap/
│   ├── augment.py
│   ├── build_policy.py
│   ├── plot_class_strategy_evolution.py
│   ├── plot_class_strategy_evolution_presentable.py
│   ├── plot_class_strategy_evolution_real.py
│   └── utils.py
└── requirements.txt
```

## Notes

This release intentionally does not include the original dataset, pretrained AST weights, trained models, or cached validation predictions. To reproduce experiments, users must provide their own audio dataset, regenerate validation prediction caches, and update local paths in `config/config.py`.
