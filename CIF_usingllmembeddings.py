import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import mahalanobis
from scipy.special import rel_entr

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
    a = embedding_a.detach().numpy()  # FIX: Added .detach()
    b = embedding_b.detach().numpy()  # FIX: Added .detach()
    diff = a - b
    return mahalanobis(diff, np.zeros_like(diff), np.linalg.inv(cov_matrix))

def kl_divergence(embedding_a, embedding_b):
    """Computes KL Divergence between two embeddings."""
    a = F.softmax(embedding_a, dim=0).detach().numpy()  # FIX: Added .detach()
    b = F.softmax(embedding_b, dim=0).detach().numpy()  # FIX: Added .detach()
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
# 4️⃣ Experimental Validation with Real LLM Embeddings
# ==============================

def experimental_validation():
    """Runs context isolation test using real LLM embeddings."""
    
    # Sample contexts
    contexts = [
        "Medical history analysis",
        "Financial risk assessment",
        "Legal document review"
    ]
    
    # Initialize CIF
    isolator = ContextIsolationFramework()
    
    # Convert contexts to embeddings
    embeddings = [get_embedding(text) for text in contexts]
    
    # Pass through CIF
    isolated_embeddings = [isolator(embed) for embed in embeddings]

    # Compute covariance matrix (for Mahalanobis)
    stacked_embeddings = torch.stack(isolated_embeddings).detach().numpy()  # FIX: Added .detach()
    cov_matrix = np.cov(stacked_embeddings, rowvar=False) + np.eye(stacked_embeddings.shape[1]) * 1e-6  # Regularization

    # Store results
    results = {
        'boundary_scores': [],
        'context_distances': []
    }

    # Compute boundary scores
    for i in range(len(isolated_embeddings)):
        for j in range(i + 1, len(isolated_embeddings)):
            score = calculate_boundary_score(isolated_embeddings[i], isolated_embeddings[j], cov_matrix)
            results['boundary_scores'].append(score)

    # Compute context distances
    for i in range(len(isolated_embeddings)):
        for j in range(i + 1, len(isolated_embeddings)):
            distance = euclidean_distance(isolated_embeddings[i], isolated_embeddings[j])
            results['context_distances'].append(distance)

    return results

# ==============================
# 5️⃣ Run the Final Experiment
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
