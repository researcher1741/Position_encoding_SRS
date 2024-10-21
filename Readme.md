# Positional encoding is not the same as context: A study on positional encoding for Sequential recommendation

This repository contains the code used for the paper "Positional encoding is not the same as context: A study on positional encoding for Sequential recommendation".
The experiments use CARCA, the state-of-the-art (SOTA) model as of 2023. The main model is exactly like CARCA but rewritten in pytorch. The original code was in tensorflow.
This analysis study the relevance of encodings in transformer models for sequential recommendation systems.

## Data

The datasets available online differ in the number of items, users, and the method used to generate embeddings 
(e.g., some are based on image embeddings, while others use text embeddings from SBERT). 
For reproducibility, especially considering seed consistency, we recommend using the following preprocessed dataset:

[Preprocessed Data](https://drive.google.com/drive/folders/1a_u52mIEUA-1WrwsNZZa-aoGJcMmVugs?usp=sharing)

## PyTorch Environment

To ensure compatibility, please use the following package versions:

- `matplotlib==3.3.4`
- `numpy==1.21.1`
- `pandas==1.1.5`
- `scikit_learn==1.3.0`
- `tqdm==4.65.0`
- `transformers==4.18.0`
- `torch==2.0.0`

## Running the Models

To run the model on the respective dataset using PyTorch, use the following command:

```bash
python main.py
