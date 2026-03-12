================================================================================
BERT+RAG Fusion Model for Toxic Comment Classification - README
COMP90051 Machine Learning Project
================================================================================

PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements a novel fusion approach combining BERT (deep learning) and Logistic Regression (traditional ML) with RAG-based retrieval for robust toxic comment classification. The system is evaluated using 10-fold nested cross-validation and tested under multiple noise conditions.

Key Components:
- Baseline: TF-IDF + GPU-accelerated Logistic Regression (cuML)
- Deep Learning: BERT fine-tuning with class-weighted training
- Fusion: RAG-enhanced probability fusion with hyperparameter optimization
- Robustness: Multi-level noise testing (5%, 10%, 20%)


SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------
Hardware:
- GPU with CUDA support (tested on NVIDIA A100 80GB)
- Minimum 32GB RAM recommended
- 10GB+ free disk space

Software:
- Python 3.8+
- CUDA 11.6+ (for GPU acceleration)
- Google Colab (recommended) or local Jupyter environment


DEPENDENCIES
--------------------------------------------------------------------------------
Core Libraries:
- torch >= 2.0.0
- transformers >= 4.30.0
- cuml >= 23.0 (GPU-accelerated ML)
- cupy >= 12.0 (GPU arrays)
- faiss-gpu >= 1.7.0 (vector similarity search)
- sentence-transformers >= 2.2.0
- scikit-learn >= 1.2.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.5.0

Installation (in notebook):
All dependencies are automatically installed when running the notebook cells.
Key installations are handled via:
    %pip install -q sentence-transformers faiss-cpu


DATASET REQUIREMENTS
--------------------------------------------------------------------------------
Required File: train.csv

Source: Kaggle Toxic Comment Classification Challenge
URL: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Expected Format:
- Columns: id, comment_text, toxic, severe_toxic, obscene, threat, 
           insult, identity_hate
- Minimum 10,000 instances (after cleaning)
- Text data in English

Placement:
- Upload to: archive/train.csv (auto-prompted during execution)
- Or manually create archive/ folder and place train.csv inside


HOW TO RUN THE CODE
--------------------------------------------------------------------------------

METHOD 1: Google Colab (Recommended)
-------------------------------------
1. Open Google Colab: https://colab.research.google.com/
2. Upload the notebook: COMP90051_Project.ipynb
3. Change runtime to GPU:
   - Runtime → Change runtime type → Hardware accelerator: GPU → A100
4. Run cells sequentially from top to bottom:
   - Cell 1 (C0): Global configuration
   - Cell 2 (C1): Common utilities
   - Cell 3 (C2.1): Data cleaner
   - Cell 4 (C2.2): Data loading (will prompt for train.csv upload)
   - Continue through remaining cells...
5. When prompted, upload train.csv file
6. Wait for execution (total ~2-4 hours for full pipeline)

METHOD 2: Local Jupyter
------------------------
1. Install Jupyter: pip install jupyter
2. Install CUDA toolkit and cuML:
   - Follow: https://docs.rapids.ai/install
3. Clone/download the notebook
4. Place train.csv in archive/ folder (or create it)
5. Launch: jupyter notebook
6. Open: BERT_LR_Fusion重构英文最终.ipynb
7. Run all cells: Cell → Run All


EXECUTION FLOW
--------------------------------------------------------------------------------
The notebook follows this pipeline:

Stage 0: Setup (C0)
- Configure paths, hyperparameters, random seeds
- Load tokenizer (bert-base-uncased)

Stage 1: Data Preparation (C1-C2)
- Load train.csv
- Clean data (remove duplicates, invalid entries, timestamps)
- Create binary label (is_toxic)
- Subsample to 30,000 instances (if larger)
- Generate fixed 10-fold splits

Stage 2: Noise Generation (C3)
- Create noise generator for robustness testing
- Cache noisy versions at p=0.05, 0.10, 0.20

Stage 3: Model Training (C4-C6)
- Train Logistic Regression baseline (TF-IDF + cuML)
- Fine-tune BERT with 3-fold inner CV for hyperparameter search
- Both use 10-fold outer CV
- Evaluate on clean and noisy test sets

Stage 4: Fusion (C6 continued)
- Build RAG cache (sentence embeddings + FAISS index)
- Grid search fusion hyperparameters (alpha, K, temperature)
- Evaluate fusion on clean and noisy test sets

Stage 5: Analysis (C7)
- Aggregate results across folds
- Generate visualizations:
  * Error bar plots (LR vs BERT vs Fusion)
  * Robustness curves (F1 vs noise level)
  * Hyperparameter frequency plots
- Statistical tests (Wilcoxon signed-rank)

Stage 6: Reproducibility (C8)
- Output system configuration and library versions


HYPERPARAMETER SETTINGS
--------------------------------------------------------------------------------
Logistic Regression:
- C: [0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
- Penalty: l2
- Max iterations: 1000

BERT:
- Learning rate: [3e-5, 5e-5, 7e-5]
- Batch size: 16
- Epochs: 2
- Warmup ratio: 0.1
- Weight decay: 0.01

Fusion:
- Alpha (BERT weight): [0.0, 0.2, 0.5, 0.8, 1.0]
- K (top-K neighbors): 32
- Temperature: [0.05, 0.10, 0.15]
- Tau (decision threshold): 0.5

Cross-Validation:
- Outer folds: 10 (fixed, stratified)
- Inner folds: 3 (for hyperparameter search)


OUTPUT FILES AND LOCATIONS
--------------------------------------------------------------------------------
Directory Structure:
├── data/
│   ├── cleaned_toxic_data.csv        # Cleaned full dataset
│   ├── subset_30k.csv                 # Subsampled data (30,000)
│   └── folds_10.json                  # Fixed 10-fold indices
│
├── unified/
│   └── by_fold/                       # Cached noisy data per fold
│
├── fusion_results/
│   ├── unified/
│   │   ├── by_fold/                   # Per-fold metrics (CSV)
│   │   ├── summary_stats.csv          # Aggregated results
│   │   └── summary_delta_f1.csv       # Robustness metrics
│   │
│   └── hp_logs/
│       └── hp_best_by_fold.csv        # Best hyperparameters per fold
│
├── figures/
│   ├── unified_f1_errorbars.png       # Model comparison plot
│   ├── robustness_curve.png           # Noise robustness plot
│   └── fusion_alpha_frequency.png     # Hyperparameter distribution
│
└── runs_bert/                         # BERT training logs (per fold)

Key Metrics Saved:
- F1-macro, Accuracy, ROC-AUC, AUPR
- Precision/Recall per class
- Standard deviations across folds


VERSION INFORMATION
--------------------------------------------------------------------------------
Code Version: Final Refactored (October 2025)
Compatible With: COMP90051 2025-2 Project Requirements
Tested On: Google Colab (A100 GPU), Python 3.12, PyTorch 2.8

Last Updated: October 17 2025