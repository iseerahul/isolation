import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================
# 1️⃣ Context Isolation Framework (CIF)
# ==============================
class ContextIsolationFramework(nn.Module):
    def __init__(self, embedding_dim=768):
        super(ContextIsolationFramework, self).__init__()
        
        # Enhanced transformation layers for better isolation
        self.isolation_layer = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),  # Restore to original size
            nn.LayerNorm(embedding_dim)  # Normalize for stable embeddings
        )

    def forward(self, embeddings):
        """Pass embeddings through isolation mechanism."""
        return self.isolation_layer(embeddings)

# ==============================
# 2️⃣ Boundary Score Calculation (Hybrid Approach)
# ==============================
def calculate_boundary_score(embedding_a, embedding_b):
    """Compute combined similarity & distance-based score between two embeddings."""
    
    cosine_similarity = F.cosine_similarity(embedding_a, embedding_b, dim=0)
    euclidean_distance = torch.norm(embedding_a - embedding_b, p=2)
    
    boundary_score = euclidean_distance + (1 - cosine_similarity)
    
    return boundary_score.item()

# ==============================
# 3️⃣ Context Distance Calculation (Ensuring Separation)
# ==============================
def compare_contexts(embedding_a, embedding_b):
    """Measure direct context separation score."""
    return torch.norm(embedding_a - embedding_b, p=2).item()

# ==============================
# 4️⃣ Experimental Validation (Larger Dataset)
# ==============================
def experimental_validation():
    """Run final context isolation experiments on an expanded dataset."""
    
    # Expanded contexts
    contexts = [
        "Physics research paper",
        "Biological genome analysis",
        "AI ethics debate",
        "Creative poetry writing",
        "Casual social conversation",
        "Legal contract review",
        "Financial market analysis",
        "Science fiction storytelling"
    ]
    
    # Initialize CIF
    isolator = ContextIsolationFramework()
    
    # Store results
    results = {
        'boundary_scores': [],
        'context_distances': []
    }

    # Simulated embeddings (Replace with real embeddings in future)
    embeddings = [torch.randn(768) for _ in contexts]

    # Pass through isolation framework
    isolated_embeddings = [isolator(embed) for embed in embeddings]

    # Compute boundary scores
    for i in range(len(isolated_embeddings)):
        for j in range(i + 1, len(isolated_embeddings)):
            score = calculate_boundary_score(isolated_embeddings[i], isolated_embeddings[j])
            results['boundary_scores'].append(score)

    # Compute context distances
    for i in range(len(isolated_embeddings)):
        for j in range(i + 1, len(isolated_embeddings)):
            distance = compare_contexts(isolated_embeddings[i], isolated_embeddings[j])
            results['context_distances'].append(distance)

    return results

# ==============================
# 5️⃣ Running the Experiment
# ==============================
if __name__ == "__main__":
    results = experimental_validation()

    print("🔹 Updated Boundary Scores:", results["boundary_scores"])
    print("🔹 Updated Context Distances:", results["context_distances"])

    boundary_variation = np.std(results['boundary_scores'])
    context_separation = np.mean(results['context_distances'])

    print("\n📊 Final Analysis:")
    print(f"✅ Boundary Score Consistency: {boundary_variation:.4f}")
    print(f"✅ Average Context Separation: {context_separation:.4f}")
