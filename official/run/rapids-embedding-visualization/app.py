import cudf
from cuml.manifold.umap import UMAP
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 10000
n_features = 50
X = np.random.rand(n_samples, n_features)
X_cudf = cudf.DataFrame(X)

embedding = UMAP(n_neighbors=10, min_dist=0.01,  init="random").fit_transform(X_cudf)
embedding_cpu = embedding.to_pandas().values

plt.figure(figsize=(12, 10))
plt.scatter(embedding_cpu[:, 0], embedding_cpu[:, 1], alpha=0.5)
plt.title('UMAP Projection of Random Data')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

plt.savefig('umap_random_data.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'umap_random_data.png'")