# CIF3: Context Isolation Framework v3Add commentMore actions

CIF3 is a novel research framework and implementation for **explicit context isolation in language model embeddings**. It is designed to address the challenge of context leakage in large language models (LLMs) by enforcing clear boundaries between different contexts within the embedding space. This repository contains the code, experiments, and evaluation metrics for CIF3.

---

## 🚀 Key Features & Innovations

- **Context Gating Mechanism:** Dynamically modulates token representations to preserve context boundaries.
- **Security Layer with Multi-head Attention:** Reinforces intra-context coherence and suppresses inter-context interference.
- **Leakage Prevention System:** Penalizes context leakage during training using adversarial objectives.
- **Hybrid Boundary Score:** Combines Euclidean, Cosine, Mahalanobis, and KL Divergence metrics for robust context separation evaluation.
- **Residual Block Architecture:** Deep, scalable, and robust for large datasets.
- **Comprehensive Evaluation:** Includes clustering metrics (ARI, NMI), inter-category distance, and visualization (PCA, t-SNE).

---

## 📂 Project Structure

- `cif3.py` — Core CIF3 architecture, metrics, and visualization tools.
- `CIF_usingllmembeddings.py` — CIF implementation with LLM embeddings and hybrid boundary score.
- `test_cif3.py` — Main experimental pipeline, dataset loaders, and statistical evaluation.
- `CIF_onlargedata.py` — CIF for large-scale data scenarios(experimental).
- `CIF_Vs_baseline.py` — Baseline comparison and visualization(experimental).
- `requirements.txt` — Python dependencies.

---

## 🧑‍💻 How It Works

1. **Embedding Extraction:** Uses transformer models (e.g., MiniLM) to generate embeddings for input texts.
2. **Context Isolation:** Passes embeddings through the CIF3 model, which applies context gating, security attention, and leakage prevention.
3. **Boundary Scoring:** Computes hybrid scores to quantify context separation.
4. **Clustering & Visualization:** Evaluates context separation using ARI, NMI, and visualizes with PCA/t-SNE.
5. **Statistical Analysis:** Runs multiple trials and reports confidence intervals for all metrics.

---

## 📊 Example Results

- **Improved Context Separation:** CIF3 increases inter-category distance and hybrid boundary scores over standard LLM embeddings.
- **Better Clustering:** Higher ARI and NMI scores, indicating improved context grouping.
- **Clear Visual Separation:** PCA and t-SNE plots show distinct clusters for different contexts.

---

## 📦 Installation & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run experiments:**
   ```bash
   python test_cif3.py --dataset newsgroups --n_trials 5
   ```
   You can also use BBC News, CSV, or custom text datasets (see code for options).

---

## 📚 Datasets Supported
- 20 Newsgroups
- BBC News
- Custom CSV/text collections

---

## 📈 Evaluation Metrics
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Inter-category Distance
- Hybrid Boundary Score (Euclidean, Cosine, Mahalanobis, KL Divergence)

---

## 🛡️ Applications
- Privacy-sensitive information handling
- Multi-domain text processing
- Context-aware information retrieval
- Security-critical NLP systems

---



## 📬 Contact
For questions or issues, please open a GitHub issue or contact the author.

---

## 📑 License
This code is released under [License Name] for academic research only. See the LICENSE file for details.
