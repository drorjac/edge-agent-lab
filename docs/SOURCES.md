# Sources and licenses

## Measured dataset

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
*Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository.
[DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K).
[Dataset description and CC BY 4.0 license](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

Changes in this repository: zero-based labels; subject-held-out validation split
inside the official training partition; channel-wise normalization; synthetic
corruptions; and an independent procedural simulator. Original recordings are not
redistributed in Git. Derived metrics and example inference inputs retain the
above dataset attribution.

## Deployment APIs

- [Apple Core ML PyTorch conversion workflow](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)
- [Apple Core ML quantization overview](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)
- [PyTorch dynamic quantization API](https://pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)

The implementation uses published APIs, then measures actual local behavior. It
does not reproduce performance claims from these documents.

## Optional language model

[Qwen2.5-1.5B-Instruct-GGUF publisher repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), Apache-2.0.
The model fetcher records the exact publisher revision, file size, and LFS SHA256.
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python), MIT, runs the model
locally through grammar-constrained chat completions. The optional Ollama adapter
is independent of the measured in-process llama.cpp evaluation.

Repository code is MIT. Upstream dataset/model licenses govern those assets.

## Independent transfer data

Logacjov, A. & Ustad, A., *HAR70+*, UCI Machine Learning Repository,
[DOI 10.24432/C5CW3D](https://doi.org/10.24432/C5CW3D), CC BY 4.0.
The probe uses lower-back acceleration, a low-pass gravity estimate, zero-filled
absent gyroscope channels, homogeneous non-overlapping windows, and a reduced
activity-label mapping. These transformations and predefined orientation variants
are project modifications, not the dataset authors' benchmark protocol.
