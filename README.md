
# new file:   cif3.py (modified the previous version)
#	new file:   tempCodeRunnerFile.py
#	new file:   test_cif3.py

# CIFv3: Context Isolation Framework for Semantic Embedding Enhancement

This repository evaluates the CIFv3 (Context Isolation Framework v3) , a deep residual architecture designed to refine high-dimensional sentence embeddings for improved semantic clustering and separability.

## 🚀 Purpose

The goal is to validate CIFv3’s ability to enhance contextual embeddings (e.g., from MiniLM) by evaluating:
- **Semantic cohesion** (how well similar items stay together)
- **Cluster separation** (how far apart different classes are)
- **Downstream clustering performance**

## 🧪 Evaluation Method  (new file:   test_cif3.py)


We apply **unsupervised clustering evaluation** using the following metrics:


- 🔹 **Adjusted Rand Index (ARI)**: Compares predicted clusters to ground truth labels
- 🔹 **Normalized Mutual Information (NMI)**: Measures mutual dependency between predicted clusters and true categories
- 🔹 **Inter-Category Distance**: Mean distance between class centroids to show global separability

All metrics are computed for:
- **Baseline embeddings** (MiniLM outputs)
- **CIFv3-enhanced embeddings**

## 📊 Dataset Support

- `20 Newsgroups` (text classification benchmark)
- `BBC News` (alternative available)
- Custom `.csv` or directory-based datasets

## 📦 Pipeline Overview

1. Load and preprocess dataset
2. Generate sentence embeddings using pretrained transformer
3. Pass embeddings through CIFv3 (deep residual enhancer)
4. Evaluate clustering quality before and after CIFv3
5. Visualize results (PCA plots + metric comparison)

## 🧰 Tech Stack

- Python (PyTorch, HuggingFace Transformers)
- Scikit-learn (clustering and metrics)
- Matplotlib (visualizations)



   
