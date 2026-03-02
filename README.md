# TRAC: Tensor-Train based Across-layer Compression for Parameter-Efficient Fine-Tuning

This is the official implementation repository for the paper **"TRAC: Tensor-Train based Across-layer Compression for Parameter-Efficient Fine-Tuning"**, accepted to **ICLR 2026**.

## Announcement regarding Code Release

We are currently in the process of refactoring and documenting the source code to ensure it is user-friendly, reproducible, and easy to integrate with existing frameworks.

**The source code of TRAC algorithm implementation will be released here.**

### Release Roadmap

- **Phase 1 (March 2026):** Release of the core TRAC algorithm implementation.
- **Phase 2 (April 2026):** Release of reproduction scripts for main experiments (e.g., GLUE benchmark).
- **Phase 3 (Ongoing):** Release of additional extensions, further optimizations, and support for more model architectures.

## Abstract

*Fine-tuning large pre-trained models under resource constraints remains challenging due to the massive number of parameters involved. Existing parameter-efficient tuning methods, such as low-rank adaptation (LoRA) and its variants, rely heavily on matrix factorization and often struggle in extremely low-parameter regimes. In this work, we propose TRAC, a novel fine-tuning framework that leverages **T**ensor-T**r**ain decomposition with **A**cross-layer **C**ompression. Specifically, TRAC represents each adaptation module as a compact sequence of tensor-train cores and allows certain cores to be frozen or shared across layers, thereby exploiting the inherent similarity and redundancy among layer weight matrices. To retain layer-specific flexibility, lightweight controllers are introduced, enabling shared tensor cores to adaptively modulate representations. We evaluate TRAC on diverse architectures, including Qwen, LLaMA, GPT, BERT, and ViT, across benchmarks covering text classification, text generation, and image classification. Experimental results demonstrate that TRAC achieves performance comparable to or better than LoRA and its variants, while substantially reducing trainable parameters and storage requirements.* 


## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{
ye2026trac,
title={{TRAC}: Tensor-Train based Across-layer Compression for Parameter-Efficient Fine-Tuning},
author={Bangguo Ye and Yuanwei Zhang and Xiaoqun Zhang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=tz5yPWZp9W}
}
