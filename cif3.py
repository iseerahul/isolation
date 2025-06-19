import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ========== 1️ Embedding Loader ==========
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze()

# ========== 2️ Deep CIFv3 Architecture with Residuals ==========
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
    def __init__(self, dim=384, depth=4, num_heads=4):
        super().__init__()
        # Add MCP-specific components
        self.context_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.security_layer = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads) for _ in range(depth)
        ])
        self.stack = nn.Sequential(*[ResidualBlock(dim) for _ in range(depth)])
        
        # Add leakage prevention
        self.leakage_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply context gating
        gate = self.context_gate(x)
        x = x * gate
        
        # Apply security attention layers
        for attn in self.security_layer:
            attn_out, _ = attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = x + attn_out.squeeze(0)
        
        # Apply standard isolation
        x = self.stack(x)
        
        # Check for potential leakage
        leakage_score = self.leakage_detector(x)
        
        return x, leakage_score

# ========== 3️ Metrics ==========
def euclidean(a, b):
    return torch.norm(a - b).item()

def cosine(a, b):
    return 1 - F.cosine_similarity(a, b, dim=0).item()

def mahalanobis(a, b, cov_inv):
    diff = (a - b).detach().numpy()
    return np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))

def kl_div(a, b):
    p = F.softmax(a, dim=0).detach().numpy() + 1e-6
    q = F.softmax(b, dim=0).detach().numpy() + 1e-6
    return np.sum(p * np.log(p / q))

def hybrid_score(a, b, cov_inv):
    e = euclidean(a, b)
    c = cosine(a, b)
    m = mahalanobis(a, b, cov_inv)
    k = kl_div(a, b)
    return (e + c + m + k) / 4

# ========== 4️ Visualization ==========
def visualize(baseline, cif):
    base_np = torch.stack(baseline).detach().numpy()
    cif_np = torch.stack(cif).detach().numpy()

    pca = PCA(n_components=2)
    base_p = pca.fit_transform(base_np)
    cif_p = pca.transform(cif_np)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    base_t = tsne.fit_transform(base_np)
    cif_t = tsne.fit_transform(cif_np)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(base_p[:, 0], base_p[:, 1], c='blue', label='Baseline')
    plt.scatter(cif_p[:, 0], cif_p[:, 1], c='red', label='CIFv3')
    plt.title("PCA Projection")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(base_t[:, 0], base_t[:, 1], c='blue', label='Baseline')
    plt.scatter(cif_t[:, 0], cif_t[:, 1], c='red', label='CIFv3')
    plt.title("t-SNE Projection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== 5️ Evaluate CIFv3 ==========
def evaluate(contexts):
    model = CIFv3()
    embeddings = [get_embedding(c) for c in contexts]
    baseline = [e.clone().detach() for e in embeddings]
    cif_out = [model(e)[0] for e in embeddings]

    X = torch.stack(cif_out).detach().numpy()
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    b_scores, e_dists = [], []
    for i in range(len(cif_out)):
        for j in range(i+1, len(cif_out)):
            score = hybrid_score(cif_out[i], cif_out[j], cov_inv)
            dist = euclidean(cif_out[i], cif_out[j])
            b_scores.append(score)
            e_dists.append(dist)

    print("\n📊 CIFv3 Analysis:")
    print(f"✅ Boundary Score Mean: {np.mean(b_scores):.4f}")
    print(f"✅ Boundary Score Std: {np.std(b_scores):.4f}")
    print(f"✅ Context Separation (Mean Distance): {np.mean(e_dists):.4f}")

    plt.hist([euclidean(a, b) for i, a in enumerate(baseline) for j, b in enumerate(baseline) if j > i],
             bins=20, alpha=0.6, label='Baseline', color='blue')
    plt.hist(e_dists, bins=20, alpha=0.6, label='CIFv3', color='red')
    plt.title("Context Distance Distribution")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    visualize(baseline, cif_out)

# ========== 6️ Run Test ==========
if __name__ == "__main__":
    ctx = [
        "Paris is the capital city of France.",
        "Plants require carbon dioxide for survival.",#outlier
        "Paris is one of the most popular tourist destinations in the world.",
        "Indian cuisine is known for its diverse flavors and spices.",#outlier
        "The children are playing in the park.",  #outlier
        "France is renowned for its art, cuisine, and history.",
        "The Eiffel Tower, located in Paris, is an iconic symbol of French culture.",
        "The French government operates from the Élysée Palace.",
        "My name is Rahul.",#outlier
        "Paris hosts famous museums like the Louvre and Musée d'Orsay.",
        "Major cities like Lyon and Marseille boost France’s economy.",
        "The Seine River flows through Paris.",
        "Donald Trump was the president of the USA.", #outlier
        "The Treaty of Paris marked the end of several historic wars."
        "The groud is the world's largest art museum.",  #outlier
        "india is known for its rich cultural heritage.",   #outlier
        "china is the most populous country in the world.", #outlier
        
    ]
    evaluate(ctx)
