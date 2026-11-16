import numpy as np

class KMeansNumPy:
    def __init__(self, n_clusters=4, init="k-means++", max_iters=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None
        self.history_ = []

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        if self.init == "random":
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()
        centroids = [X[rng.integers(0, n_samples)]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / np.sum(dist_sq)
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.centroids = self._init_centroids(X)
        self.history_ = [self.centroids.copy()]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            self.history_.append(new_centroids.copy())

            if np.linalg.norm(new_centroids - self.centroids) <= self.tol:
                self.centroids = new_centroids
                break
            self.centroids = new_centroids

        final_distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        min_dist = np.min(final_distances, axis=1)
        self.inertia_ = float(np.sum(min_dist ** 2))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
