# Positional Encoding is Not the Same as Context: A Study on Positional Encoding for Sequential Recommendation

This repository contains the code and resources used for the paper **"Positional Encoding is Not the Same as Context: A Study on Positional Encoding for Sequential Recommendation"**.

The experiments in this study utilize CARCA, the state-of-the-art (SOTA) model as of 2023, rewritten from TensorFlow to PyTorch for improved accessibility and performance. This analysis focuses on the importance of positional encodings in transformer models for sequential recommendation systems.

## Full Paper

The full version of the paper is available in the repository:
[Full Paper PDF](positional_encoding_SRS_full_version_17_01_25.pdf)

## Data

The datasets used in this study vary in the number of items, users, and the methods used to generate embeddings (e.g., image-based embeddings or text embeddings from SBERT). For reproducibility, especially considering seed consistency, we recommend using the following preprocessed datasets:

[Preprocessed Data](https://drive.google.com/drive/folders/1a_u52mIEUA-1WrwsNZZa-aoGJcMmVugs?usp=sharing)

## PyTorch Environment

To ensure compatibility and reproducibility, please use the following package versions:

- `matplotlib==3.3.4`
- `numpy==1.21.1`
- `pandas==1.1.5`
- `scikit-learn==1.3.0`
- `tqdm==4.65.0`
- `transformers==4.18.0`
- `torch==2.0.0`

## Running the Models

To run the model on the respective dataset using PyTorch, use the following command:

```bash
python main.py
```

Before running it you may select the model, encoding, dataset and hyperparameters.
