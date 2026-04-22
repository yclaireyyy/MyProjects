# UGC Toxic Comment Classification

**COMP90051 Machine Learning Project**

## Project Overview

This project implements a novel fusion approach for robust toxic comment classification by combining:

- **BERT** (deep learning)
- **Logistic Regression** (traditional machine learning)
- **RAG-based retrieval** for probability fusion

The system is evaluated with **10-fold nested cross-validation** and tested under multiple **noise conditions** to assess robustness.

### Key Components

- **Baseline**: TF-IDF + GPU-accelerated Logistic Regression (cuML)
- **Deep Learning**: BERT fine-tuning with class-weighted training
- **Fusion**: RAG-enhanced probability fusion with hyperparameter optimization
- **Robustness Testing**: Multi-level noise evaluation (5%, 10%, 20%)

---

## System Requirements

### Hardware

- GPU with CUDA support (tested on **NVIDIA A100 80GB**)
- Minimum **32GB RAM** recommended
- At least **10GB free disk space**

### Software

- Python **3.8+**
- CUDA **11.6+** (for GPU acceleration)
- **Google Colab** (recommended) or local **Jupyter Notebook** environment

---

## Dependencies

### Core Libraries

- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `cuml >= 23.0`
- `cupy >= 12.0`
- `faiss-gpu >= 1.7.0`
- `sentence-transformers >= 2.2.0`
- `scikit-learn >= 1.2.0`
- `pandas >= 1.5.0`
- `numpy >= 1.23.0`
- `matplotlib >= 3.5.0`

### Installation

In the notebook, dependencies are installed automatically. For example:

```python
%pip install -q sentence-transformers faiss-cpu
