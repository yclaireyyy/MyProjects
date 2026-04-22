# Evaluating RAG-Enhanced Transformer Robustness for UGC Toxic Comment Classification Task Under Noisy Text

**Robust toxic comment classification under noisy text using Logistic Regression, BERT, and retrieval-augmented fusion**

This project studies whether retrieval augmentation can improve the robustness of Transformer-based toxic comment detection in noisy user-generated content. Using the Jigsaw Toxic Comment dataset, I built and compared three moderation pipelines — **Logistic Regression**, **fine-tuned BERT**, and a **BERT-RAG post-fusion model** — under both clean and corrupted text settings. The project combines **data cleaning**, **noise simulation**, **nested cross-validation**, and **retrieval-enhanced classification** to evaluate how much RAG actually helps when input quality degrades.

## Project Highlights

- Designed a **UGC noisy-text moderation prototype** around a clear research question: can retrieval augmentation improve Transformer robustness under text corruption?
- Built a **retrieval-enhanced moderation workflow** using **Sentence-BERT + FAISS + kNN**, integrated with BERT through a post-fusion classification pipeline.
- Constructed a full experimental setup with **10×3 nested cross-validation** and controlled noise injection at **5%, 10%, and 20%** intensity.
- Evaluated model behavior using **Macro-F1, AUPR, and ΔF1**, and identified why retrieval gains remain limited under high-noise, highly imbalanced settings. :contentReference[oaicite:1]{index=1}

## Problem Framing

Toxic comment detection is a core moderation task in online platforms, but model performance often drops when text becomes noisy — for example through typos, swapped characters, or corrupted wording. This project focuses on a practical moderation question:

> **Can retrieval augmentation improve the robustness of Transformer-based toxic comment classification under noisy text?** :contentReference[oaicite:2]{index=2}

To answer this, I compared:

- **Logistic Regression** as a traditional baseline
- **BERT** as the main Transformer classifier
- **BERT-RAG** as a retrieval-enhanced extension of BERT :contentReference[oaicite:3]{index=3}

## Method

### 1. Data Preparation
The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset and converts the original six toxicity labels into a **binary toxic / non-toxic task**. After cleaning, the dataset contains **151,443 valid samples**, and a **30,000-sample subset** is used for efficient experimentation while preserving class distribution. :contentReference[oaicite:4]{index=4}

### 2. Noise Simulation
To mimic real-world text corruption, I injected four types of noise:

- character replacement
- character swap
- character deletion
- word swap

Noise was applied at **5%, 10%, and 20%** intensity, and robustness was measured mainly through **ΔF1 = F1_clean - F1_noisy**. :contentReference[oaicite:5]{index=5}

### 3. Model Design
The project compares three model families:

- **Logistic Regression** with TF-IDF features
- **BERT-base-uncased** fine-tuned for binary classification
- **BERT-RAG post-fusion**, where comments are embedded with **all-MiniLM-L6-v2**, retrieved through **FAISS**, and fused with BERT predictions using tuned fusion weights and temperature settings :contentReference[oaicite:6]{index=6}

### 4. Evaluation Strategy
To make model comparison more reliable, the experiments use:

- **10-fold outer cross-validation**
- **3-fold inner cross-validation** for hyperparameter tuning

This nested CV design helps separate model selection from final evaluation. :contentReference[oaicite:7]{index=7}

## Key Results

On **clean data**, the retrieval-enhanced fusion model achieved the strongest performance:

- **Fusion**: Macro-F1 = **0.905 ± 0.013**
- **BERT**: Macro-F1 = **0.893 ± 0.011**
- **LR**: Macro-F1 = **0.850 ± 0.016** :contentReference[oaicite:8]{index=8}

Under **heavy noise (p = 0.20)**, the robustness advantage became much smaller:

- **Fusion**: ΔF1 = **0.047 ± 0.048**
- **BERT**: ΔF1 = **0.053 ± 0.055** :contentReference[oaicite:9]{index=9}

### Takeaway
RAG provided a **small but stable improvement on clean text**, but only **limited robustness gains under strong noise**. The report suggests that this is mainly because noisy input shifts retrieval embeddings and makes it harder to retrieve useful minority-class evidence in a highly imbalanced dataset. :contentReference[oaicite:10]{index=10}

## Why the Gains Were Limited

The project findings suggest that retrieval augmentation helps when the input remains relatively clean, but becomes less reliable when:

- noisy text distorts embedding quality
- nearest-neighbor retrieval becomes less accurate
- minority-class evidence is harder to recover under imbalance
- fusion uses fixed retrieval settings rather than adaptive gating or reranking :contentReference[oaicite:11]{index=11}

## Future Improvements

Based on the experimental results, the next iteration would focus on:

- **similarity gating**
- **dynamic fusion weights**
- **reranking models**
- **contrastive learning** for more noise-robust retrieval embeddings :contentReference[oaicite:12]{index=12}

## Technical Stack

- Python
- Scikit-learn
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS
- Jupyter / Colab

## Project Value

This project is not just a classification benchmark. It is a compact moderation-system prototype that combines:

- **rule-aware experimental thinking**
- **model comparison across complexity levels**
- **retrieval-enhanced architecture design**
- **robustness testing under noisy text**
- **metric-driven iteration planning**

It shows how a moderation pipeline can move from a pure classifier to a more structured **“model + retrieval + evaluation”** design, while also making clear where retrieval helps and where it still breaks down. :contentReference[oaicite:13]{index=13}
