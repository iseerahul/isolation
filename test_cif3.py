# === CIFv3 with Clustering on Real-world Datasets ===

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os
import re
import requests
from io import StringIO
from tqdm import tqdm
import random
import argparse
import time
from cryptography.fernet import Fernet
import json
import base64
from scipy import stats

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Initialize tokenizer and model globally
tokenizer = None
model = None

def load_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load model and tokenizer globally"""
    global tokenizer, model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    return tokenizer, model

# Call load_model at startup
tokenizer, model = load_model()

@torch.no_grad()
def get_embedding(text, tokenizer, model, batch_size=32):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze()

def get_embeddings_batch(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# ========== 2️⃣ Deep CIFv3 Architecture with Residuals ==========
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.ff(x))

class CIFv3(nn.Module):
    def __init__(self, dim=384, depth=4):
        super().__init__()
        self.stack = nn.Sequential(*[ResidualBlock(dim) for _ in range(depth)])

    def forward(self, x):
        return self.stack(x)

# ========== 3️⃣ Dataset Loaders ==========
def load_newsgroups(n_samples=1000):
    """Load 20 Newsgroups dataset"""
    try:
        from sklearn.datasets import fetch_20newsgroups
        print("Loading 20 Newsgroups dataset...")
        newsgroups = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        
        # Limit samples if needed
        idx = random.sample(range(len(newsgroups.data)), min(n_samples, len(newsgroups.data)))
        texts = [newsgroups.data[i] for i in idx]
        labels = [newsgroups.target[i] for i in idx]
        categories = [newsgroups.target_names[label] for label in labels]
        
        # Clean texts (remove excessive whitespace, limit length)
        texts = [re.sub(r'\s+', ' ', text).strip()[:10000] for text in texts]
        
        return texts, labels, categories
    except Exception as e:
        print(f"Error loading newsgroups: {e}")
        return [], [], []

def load_bbc_news(url="http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip", n_samples=1000):
    """Load BBC News dataset"""
    try:
        import zipfile
        import tempfile
        print("Loading BBC News dataset...")
        
        # Download the dataset
        response = requests.get(url)
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            
            # Extract the dataset
            texts = []
            labels = []
            categories = []
            
            with zipfile.ZipFile(temp_file.name) as z:
                for filename in z.namelist():
                    if filename.startswith('bbc/') and not filename.endswith('/'):
                        category = filename.split('/')[1]
                        if category not in ('README.TXT', 'LICENSE.TXT'):
                            with z.open(filename) as f:
                                text = f.read().decode('utf-8', errors='ignore')
                                texts.append(text)
                                labels.append(category)
                                categories.append(category)
        
        # Convert categories to numeric labels
        unique_categories = list(set(categories))
        label_map = {cat: i for i, cat in enumerate(unique_categories)}
        numeric_labels = [label_map[cat] for cat in categories]
        
        # Limit samples if needed
        if len(texts) > n_samples:
            idx = random.sample(range(len(texts)), n_samples)
            texts = [texts[i] for i in idx]
            numeric_labels = [numeric_labels[i] for i in idx]
            categories = [categories[i] for i in idx]
        
        return texts, numeric_labels, categories
    except Exception as e:
        print(f"Error loading BBC News: {e}")
        return [], [], []

def load_csv_dataset(file_path, text_col, label_col=None, n_samples=1000):
    """Load dataset from CSV file"""
    try:
        print(f"Loading CSV dataset from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Ensure text column exists
        if text_col not in df.columns:
            print(f"Error: Column '{text_col}' not found in CSV.")
            return [], [], []
        
        texts = df[text_col].astype(str).tolist()
        
        # Handle labels if provided
        if label_col and label_col in df.columns:
            categories = df[label_col].astype(str).tolist()
            # Convert categories to numeric labels
            unique_categories = list(set(categories))
            label_map = {cat: i for i, cat in enumerate(unique_categories)}
            numeric_labels = [label_map[cat] for cat in categories]
        else:
            print("No label column specified or found. Using dummy labels.")
            numeric_labels = [0] * len(texts)
            categories = ["unknown"] * len(texts)
        
        # Limit samples if needed
        if len(texts) > n_samples:
            idx = random.sample(range(len(texts)), n_samples)
            texts = [texts[i] for i in idx]
            numeric_labels = [numeric_labels[i] for i in idx]
            categories = [categories[i] for i in idx]
        
        return texts, numeric_labels, categories
    except Exception as e:
        print(f"Error loading CSV dataset: {e}")
        return [], [], []

def load_text_files(directory, n_samples=1000):
    """Load text files from directory structure where each subdirectory is a category"""
    try:
        print(f"Loading text files from {directory}...")
        texts = []
        categories = []
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            category = os.path.basename(root)
            if category and files:  # Skip empty directories and root dir
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read()
                                texts.append(text)
                                categories.append(category)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
        
        # Convert categories to numeric labels
        unique_categories = list(set(categories))
        label_map = {cat: i for i, cat in enumerate(unique_categories)}
        numeric_labels = [label_map[cat] for cat in categories]
        
        # Limit samples if needed
        if len(texts) > n_samples:
            idx = random.sample(range(len(texts)), n_samples)
            texts = [texts[i] for i in idx]
            numeric_labels = [numeric_labels[i] for i in idx]
            categories = [categories[i] for i in idx]
        
        return texts, numeric_labels, categories
    except Exception as e:
        print(f"Error loading text files: {e}")
        return [], [], []

# ========== 4️⃣ Clustering and Evaluation ==========
def cluster_and_evaluate(embeddings, labels, title="", k=None, sample_size=1000):
    # Convert inputs to correct format
    if isinstance(embeddings, torch.Tensor):
        X = embeddings.detach().cpu().numpy()
    else:
        X = np.array(embeddings, dtype=np.float64)  # Ensure float64 for sklearn compatibility
    
    # Convert labels to numpy array for consistency
    labels_array = np.array(labels)
    
    # If data is too large, sample it for visualization
    if X.shape[0] > sample_size:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels_array[indices]
    else:
        X_sample = X
        labels_sample = labels_array
    
    # Determine optimal number of clusters if not provided
    if k is None:
        k = len(set(labels))
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    try:
        cluster_labels = kmeans.fit_predict(X)
    except ValueError as e:
        if "Buffer dtype mismatch" in str(e):
            # Try again with explicit dtype conversion
            X = np.array(X, dtype=np.float64)
            X_sample = np.array(X_sample, dtype=np.float64)
            cluster_labels = kmeans.fit_predict(X)
        else:
            raise
    
    # Apply dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    
    # Calculate metrics
    if len(set(labels)) > 1:  # Only calculate if we have ground truth labels
        rand_score = adjusted_rand_score(labels, cluster_labels)
        nmi_score = normalized_mutual_info_score(labels, cluster_labels)
    else:
        rand_score = nmi_score = 0
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot PCA visualization of clusters
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[labels_sample[i] for i in range(len(X_pca))], 
                cmap='rainbow', marker='o', alpha=0.6, s=50)
    plt.title(f"{title}\nARI: {rand_score:.3f}, NMI: {nmi_score:.3f}")
    plt.colorbar(label="True Categories")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
    
    # Also create a plot showing the predicted clusters
    plt.figure(figsize=(10, 8))
    # Get cluster predictions for visualization samples - fixed dtype issue
    sample_predictions = kmeans.predict(X_sample)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=sample_predictions, 
                cmap='rainbow', marker='o', alpha=0.6, s=50)
    plt.title(f"{title} (Predicted Clusters)")
    plt.colorbar(label="Predicted Clusters")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_predicted.png")
    plt.close()
    
    return {
        "adjusted_rand_index": rand_score,
        "normalized_mutual_info": nmi_score,
        "num_clusters": k
    }

# ========== 5️⃣ Metrics ==========
def euclidean(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.norm(a - b).item()
    else:
        return np.linalg.norm(a - b)

def cosine(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return 1 - F.cosine_similarity(a, b, dim=0).item()
    else:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return 1 - np.dot(a, b) / (a_norm * b_norm) if a_norm * b_norm != 0 else 1

def mahalanobis(a, b, cov_inv):
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    diff = a - b
    return np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))

def kl_div(a, b):
    if isinstance(a, torch.Tensor):
        p = F.softmax(a, dim=0).detach().numpy() + 1e-6
    else:
        p = np.exp(a) / np.sum(np.exp(a)) + 1e-6
        
    if isinstance(b, torch.Tensor):
        q = F.softmax(b, dim=0).detach().numpy() + 1e-6
    else:
        q = np.exp(b) / np.sum(np.exp(b)) + 1e-6
        
    return np.sum(p * np.log(p / q))

def hybrid_score(a, b, cov_inv):
    e = euclidean(a, b)
    c = cosine(a, b)
    m = mahalanobis(a, b, cov_inv)
    k = kl_div(a, b)
    return (e + c + m + k) / 4

# Statistical analysis functions
def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for a list of values."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    return mean, ci

def run_multiple_trials(evaluate_fn, n_trials=5):
    """Run multiple trials and collect statistics."""
    results = {
        'ari_baseline': [],
        'ari_cif': [],
        'nmi_baseline': [],
        'nmi_cif': [],
        'distance_baseline': [],
        'distance_cif': [],
        'ari_improvement': [],
        'nmi_improvement': [],
        'distance_improvement': []
    }
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        trial_results = evaluate_fn()
        
        # Store basic metrics
        results['ari_baseline'].append(trial_results['baseline']['adjusted_rand_index'])
        results['ari_cif'].append(trial_results['cifv3']['adjusted_rand_index'])
        results['nmi_baseline'].append(trial_results['baseline']['normalized_mutual_info'])
        results['nmi_cif'].append(trial_results['cifv3']['normalized_mutual_info'])
        results['distance_baseline'].append(np.mean(trial_results['baseline_distances']))
        results['distance_cif'].append(np.mean(trial_results['cif_distances']))
        
        # Calculate improvements
        ari_imp = ((trial_results['cifv3']['adjusted_rand_index'] - trial_results['baseline']['adjusted_rand_index']) / 
                   max(abs(trial_results['baseline']['adjusted_rand_index']), 1e-10)) * 100
        nmi_imp = ((trial_results['cifv3']['normalized_mutual_info'] - trial_results['baseline']['normalized_mutual_info']) / 
                   max(abs(trial_results['baseline']['normalized_mutual_info']), 1e-10)) * 100
        dist_imp = ((np.mean(trial_results['cif_distances']) - np.mean(trial_results['baseline_distances'])) / 
                    max(abs(np.mean(trial_results['baseline_distances'])), 1e-10)) * 100
        
        results['ari_improvement'].append(ari_imp)
        results['nmi_improvement'].append(nmi_imp)
        results['distance_improvement'].append(dist_imp)
    
    # Calculate statistics
    stats_results = {}
    for metric, values in results.items():
        mean, (ci_low, ci_high) = calculate_confidence_interval(values)
        stats_results[metric] = {
            'mean': mean,
            'std': np.std(values),
            'ci_low': ci_low,
            'ci_high': ci_high
        }
    
    return stats_results

# ========== 6️⃣ Main Evaluation Pipeline ==========
def evaluate_dataset(texts, labels, categories, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
    # Load model
    tokenizer, model = load_model(model_name)
    
    # Get embeddings
    print("Computing embeddings...")
    embeddings = get_embeddings_batch(texts, tokenizer, model, batch_size)
    
    # Initialize CIFv3 model
    print("Initializing CIFv3 model...")
    cifv3 = CIFv3(dim=embeddings.shape[1])
    
    # Process embeddings through CIFv3
    print("Processing embeddings through CIFv3...")
    with torch.no_grad():
        cif_embeddings = torch.zeros_like(embeddings)
        for i in tqdm(range(0, len(embeddings), batch_size), desc="CIFv3 processing"):
            batch = embeddings[i:i+batch_size]
            cif_embeddings[i:i+batch_size] = cifv3(batch)
    
    # Evaluate baseline embeddings
    print("\n===== Baseline Embedding Evaluation =====")
    baseline_results = cluster_and_evaluate(embeddings, labels, title="Baseline Embeddings")
    
    # Evaluate CIFv3 embeddings
    print("\n===== CIFv3 Embedding Evaluation =====")
    cif_results = cluster_and_evaluate(cif_embeddings, labels, title="CIFv3 Embeddings")
    
    # Compute metrics between categories
    print("\nComputing inter-category metrics...")
    X = cif_embeddings.detach().numpy()
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    
    # Calculate separation metrics for both baseline and CIFv3
    unique_categories = list(set(categories))
    baseline_category_centroids = {}
    cif_category_centroids = {}
    
    for cat in unique_categories:
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        baseline_category_centroids[cat] = embeddings[cat_indices].mean(dim=0)
        cif_category_centroids[cat] = cif_embeddings[cat_indices].mean(dim=0)
    
    baseline_distances = []
    cif_distances = []
    
    for i, cat1 in enumerate(unique_categories):
        for j, cat2 in enumerate(unique_categories):
            if j > i:  # Avoid duplicates
                baseline_dist = euclidean(baseline_category_centroids[cat1], baseline_category_centroids[cat2])
                cif_dist = euclidean(cif_category_centroids[cat1], cif_category_centroids[cat2])
                baseline_distances.append(baseline_dist)
                cif_distances.append(cif_dist)
    
    # Visualization of distance distributions
    plt.figure(figsize=(10, 6))
    plt.hist(baseline_distances, bins=20, alpha=0.6, label='Baseline', color='blue')
    plt.hist(cif_distances, bins=20, alpha=0.6, label='CIFv3', color='red')
    plt.title("Inter-Category Distance Distribution")
    plt.xlabel("Euclidean Distance Between Category Centroids")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("inter_category_distances.png")
    plt.close()
    
    # Print results
    print("\n📊 Summary Results:")
    print(f"Baseline Adjusted Rand Index: {baseline_results['adjusted_rand_index']:.4f}")
    print(f"CIFv3 Adjusted Rand Index: {cif_results['adjusted_rand_index']:.4f}")
    print(f"Baseline Normalized Mutual Information: {baseline_results['normalized_mutual_info']:.4f}")
    print(f"CIFv3 Normalized Mutual Information: {cif_results['normalized_mutual_info']:.4f}")
    print(f"Baseline Mean Inter-Category Distance: {np.mean(baseline_distances):.4f}")
    print(f"CIFv3 Mean Inter-Category Distance: {np.mean(cif_distances):.4f}")
    
    # Calculate improvement percentages
    ari_improvement = ((cif_results['adjusted_rand_index'] - baseline_results['adjusted_rand_index']) / 
                       max(abs(baseline_results['adjusted_rand_index']), 1e-10)) * 100
    nmi_improvement = ((cif_results['normalized_mutual_info'] - baseline_results['normalized_mutual_info']) / 
                       max(abs(baseline_results['normalized_mutual_info']), 1e-10)) * 100
    distance_improvement = ((np.mean(cif_distances) - np.mean(baseline_distances)) / 
                           max(abs(np.mean(baseline_distances)), 1e-10)) * 100
    
    print(f"\nCIFv3 Improvements:")
    print(f"Adjusted Rand Index: {ari_improvement:.2f}%")
    print(f"Normalized Mutual Information: {nmi_improvement:.2f}%")
    print(f"Inter-Category Distance: {distance_improvement:.2f}%")
    
    return {
        "baseline": baseline_results,
        "cifv3": cif_results,
        "baseline_distances": baseline_distances,
        "cif_distances": cif_distances
    }

# ========== 7️⃣ CLI Interface ==========
def main():
    parser = argparse.ArgumentParser(description='CIFv3 Clustering on Real-world Datasets')
    parser.add_argument('--dataset', type=str, default='newsgroups', 
                        choices=['newsgroups', 'bbc', 'csv', 'text_files'],
                        help='Dataset to use (newsgroups, bbc, csv, or text_files)')
    parser.add_argument('--csv_path', type=str, default='',
                        help='Path to CSV file for csv dataset')
    parser.add_argument('--text_dir', type=str, default='',
                        help='Directory containing text files for text_files dataset')
    parser.add_argument('--text_col', type=str, default='text',
                        help='Column name for text data in CSV')
    parser.add_argument('--label_col', type=str, default='label',
                        help='Column name for label data in CSV')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Transformer model to use for embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding computation')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to use from dataset')
    parser.add_argument('--n_trials', type=int, default=5,
                        help='Number of trials to run for statistical analysis')
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == 'newsgroups':
        texts, labels, categories = load_newsgroups(n_samples=args.n_samples)
    elif args.dataset == 'bbc':
        texts, labels, categories = load_bbc_news(n_samples=args.n_samples)
    elif args.dataset == 'csv':
        if not args.csv_path:
            print("Error: CSV path must be provided with --csv_path")
            return
        texts, labels, categories = load_csv_dataset(args.csv_path, args.text_col, args.label_col, n_samples=args.n_samples)
    elif args.dataset == 'text_files':
        if not args.text_dir:
            print("Error: Text directory must be provided with --text_dir")
            return
        texts, labels, categories = load_text_files(args.text_dir, n_samples=args.n_samples)
    
    # Check if dataset loaded successfully
    if not texts:
        print("Error: Failed to load dataset")
        return
    
    print(f"Loaded {len(texts)} samples with {len(set(categories))} categories")
    
    # Run evaluation with multiple trials
    stats_results = run_multiple_trials(
        lambda: evaluate_dataset(texts, labels, categories, model_name=args.model, batch_size=args.batch_size),
        n_trials=args.n_trials
    )
    
    # Print statistical results
    print("\n📊 Statistical Results (mean ± 95% CI):")
    print(f"Baseline ARI: {stats_results['ari_baseline']['mean']:.4f} ± {(stats_results['ari_baseline']['ci_high'] - stats_results['ari_baseline']['ci_low'])/2:.4f}")
    print(f"CIFv3 ARI: {stats_results['ari_cif']['mean']:.4f} ± {(stats_results['ari_cif']['ci_high'] - stats_results['ari_cif']['ci_low'])/2:.4f}")
    print(f"Baseline NMI: {stats_results['nmi_baseline']['mean']:.4f} ± {(stats_results['nmi_baseline']['ci_high'] - stats_results['nmi_baseline']['ci_low'])/2:.4f}")
    print(f"CIFv3 NMI: {stats_results['nmi_cif']['mean']:.4f} ± {(stats_results['nmi_cif']['ci_high'] - stats_results['nmi_cif']['ci_low'])/2:.4f}")
    print(f"Baseline Distance: {stats_results['distance_baseline']['mean']:.4f} ± {(stats_results['distance_baseline']['ci_high'] - stats_results['distance_baseline']['ci_low'])/2:.4f}")
    print(f"CIFv3 Distance: {stats_results['distance_cif']['mean']:.4f} ± {(stats_results['distance_cif']['ci_high'] - stats_results['distance_cif']['ci_low'])/2:.4f}")
    
    print("\nImprovements (mean ± 95% CI):")
    print(f"ARI: {stats_results['ari_improvement']['mean']:.2f}% ± {(stats_results['ari_improvement']['ci_high'] - stats_results['ari_improvement']['ci_low'])/2:.2f}%")
    print(f"NMI: {stats_results['nmi_improvement']['mean']:.2f}% ± {(stats_results['nmi_improvement']['ci_high'] - stats_results['nmi_improvement']['ci_low'])/2:.2f}%")
    print(f"Distance: {stats_results['distance_improvement']['mean']:.2f}% ± {(stats_results['distance_improvement']['ci_high'] - stats_results['distance_improvement']['ci_low'])/2:.2f}%")

if __name__ == "__main__":
    main()