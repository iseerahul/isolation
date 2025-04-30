import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import mahalanobis
from scipy.special import rel_entr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==============================
# 1️⃣ Load LLM Model for Sentence Embeddings
# ==============================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    """Converts text into an LLM embedding vector."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze()  # Average token embeddings

# ==============================
# 2️⃣ Context Isolation Framework (CIF)
# ==============================

class ContextIsolationFramework(nn.Module):
    def __init__(self, embedding_dim=384):  # Adjusted for MiniLM
        super(ContextIsolationFramework, self).__init__()
        
        # Enhanced transformation layers
        self.isolation_layer = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),  # Restore to original size
            nn.LayerNorm(embedding_dim)  # Normalize for stable embeddings
        )

    def forward(self, embeddings):
        """Pass embeddings through isolation mechanism."""
        return self.isolation_layer(embeddings)

# ==============================
# 3️⃣ Hybrid Boundary Score Calculation
# ==============================

def cosine_similarity(embedding_a, embedding_b):
    return F.cosine_similarity(embedding_a, embedding_b, dim=0).item()

def euclidean_distance(embedding_a, embedding_b):
    return torch.norm(embedding_a - embedding_b, p=2).item()

def mahalanobis_distance(embedding_a, embedding_b, cov_matrix):
    """Computes Mahalanobis distance with precomputed covariance matrix."""
    a = embedding_a.detach().numpy()
    b = embedding_b.detach().numpy()
    diff = a - b
    return mahalanobis(diff, np.zeros_like(diff), np.linalg.inv(cov_matrix))

def kl_divergence(embedding_a, embedding_b):
    """Computes KL Divergence between two embeddings."""
    a = F.softmax(embedding_a, dim=0).detach().numpy()
    b = F.softmax(embedding_b, dim=0).detach().numpy()
    return sum(rel_entr(a, b))

def calculate_boundary_score(embedding_a, embedding_b, cov_matrix):
    """Computes the hybrid boundary score using multiple metrics."""
    
    cos_sim = cosine_similarity(embedding_a, embedding_b)
    euc_dist = euclidean_distance(embedding_a, embedding_b)
    mah_dist = mahalanobis_distance(embedding_a, embedding_b, cov_matrix)
    kl_div = kl_divergence(embedding_a, embedding_b)
    
    # Weighted Hybrid Score
    boundary_score = euc_dist + (1 - cos_sim) + mah_dist + kl_div
    return boundary_score

# ==============================
# 4️⃣ Experimental Validation & Visualization
# ==============================

def experimental_validation():
    """Runs context isolation test using real LLM embeddings and visualizes the results."""
    
    # Sample contexts
    # contexts = [
    #     "Medical history analysis",
    #     "Financial risk assessment",
    #     "Legal document review",
    #     "AI ethics and regulations",
    #     "Cryptocurrency market trends",
    #     "Machine learning applications in medicine"
    # ]
    
    contexts = [
        "Paris is the capital city of France.",
        "Plants require carbon dioxide for survival.", # outlier
        "Paris is one of the most popular tourist destinations in the world.",
        "indian cuisine is known for its diverse flavors and spices.", # outlier
        "the children are playing in the park.",# outlier
        "France is renowned for its art, cuisine, and history, attracting millions of visitors annually.",
        "The Eiffel Tower, located in Paris, is an iconic symbol of French culture.",
        "The French government operates from the Palais Bourbon and the Élysée Palace.",
        "Paris hosts some of the world's most famous museums, including the Louvre and Musée d'Orsay.",
        "Many major French cities, such as Lyon and Marseille, contribute to the nation's economy.",
        "The Seine River flows through Paris, dividing the city into the Left Bank and Right Bank.",
        "The Treaty of Paris marked the end of several historic conflicts involving France."
    ]

    # Initialize CIF
    isolator = ContextIsolationFramework()
    
    # Convert contexts to embeddings
    baseline_embeddings = [get_embedding(text) for text in contexts]
    
    # Pass through CIF
    cif_embeddings = [isolator(embed) for embed in baseline_embeddings]

    # Compute covariance matrix (for Mahalanobis)
    stacked_embeddings = torch.stack(cif_embeddings).detach().numpy()
    cov_matrix = np.cov(stacked_embeddings, rowvar=False) + np.eye(stacked_embeddings.shape[1]) * 1e-6  # Regularization

    # Store results
    results = {
        'boundary_scores': [],
        'context_distances': []
    }

    # Compute boundary scores
    for i in range(len(cif_embeddings)):
        for j in range(i + 1, len(cif_embeddings)):
            score = calculate_boundary_score(cif_embeddings[i], cif_embeddings[j], cov_matrix)
            results['boundary_scores'].append(score)

    # Compute context distances
    for i in range(len(cif_embeddings)):
        for j in range(i + 1, len(cif_embeddings)):
            distance = euclidean_distance(cif_embeddings[i], cif_embeddings[j])
            results['context_distances'].append(distance)

    # Visualization
    visualize_embeddings(baseline_embeddings, cif_embeddings)

    return results

# ==============================
# 5️⃣ Visualization: Baseline vs. CIF Embeddings
# ==============================

def visualize_embeddings(baseline_embeddings, cif_embeddings):
    """Visualizes embeddings using PCA, t-SNE, and distance histograms."""
    
    # Convert to NumPy
    baseline_np = torch.stack(baseline_embeddings).detach().numpy()
    cif_np = torch.stack(cif_embeddings).detach().numpy()

    # Apply PCA
    pca = PCA(n_components=2)
    baseline_pca = pca.fit_transform(baseline_np)
    cif_pca = pca.transform(cif_np)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=3)  # Perplexity < num_samples
    baseline_tsne = tsne.fit_transform(baseline_np)
    cif_tsne = tsne.fit_transform(cif_np)

    # Plot PCA
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(baseline_pca[:, 0], baseline_pca[:, 1], c='blue', label='Baseline')
    plt.scatter(cif_pca[:, 0], cif_pca[:, 1], c='red', label='CIF')
    plt.title("PCA Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()

    # Plot t-SNE
    plt.subplot(1, 2, 2)
    plt.scatter(baseline_tsne[:, 0], baseline_tsne[:, 1], c='blue', label='Baseline')
    plt.scatter(cif_tsne[:, 0], cif_tsne[:, 1], c='red', label='CIF')
    plt.title("t-SNE Projection")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()

    plt.show()

    # Plot distance distributions
    baseline_distances = [euclidean_distance(torch.tensor(b1), torch.tensor(b2)) 
                          for i, b1 in enumerate(baseline_embeddings) 
                          for j, b2 in enumerate(baseline_embeddings) if i < j]

    cif_distances = [euclidean_distance(torch.tensor(c1), torch.tensor(c2)) 
                     for i, c1 in enumerate(cif_embeddings) 
                     for j, c2 in enumerate(cif_embeddings) if i < j]

    plt.figure(figsize=(8, 5))
    plt.hist(baseline_distances, bins=20, alpha=0.6, color='blue', label='Baseline')
    plt.hist(cif_distances, bins=20, alpha=0.6, color='red', label='CIF')
    plt.title("Context Distance Distribution")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# ==============================
# 6️⃣ Run the Final Experiment
# ==============================

if __name__ == "__main__":
    results = experimental_validation()

    # Display Results
    print("\n🔹 Updated Boundary Scores:", results["boundary_scores"])
    print("🔹 Updated Context Distances:", results["context_distances"])

    # Statistical Analysis
    boundary_variation = np.std(results['boundary_scores'])
    context_separation = np.mean(results['context_distances'])

    print("\n📊 Final Analysis:")
    print(f"✅ Boundary Score Consistency: {boundary_variation:.4f}")
    print(f"✅ Average Context Separation: {context_separation:.4f}")
