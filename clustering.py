import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def kmeans_scratch(X, k, max_iter=100, tol=1e-4, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices, :]
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            points_in_cluster = X[labels == c]
            if len(points_in_cluster) > 0:
                new_centroids[c] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[c] = X[np.random.choice(n_samples)]

        shift = np.sum(np.linalg.norm(centroids - new_centroids, axis=1))
        centroids = new_centroids

        if shift < tol:
            print(f"Converged at iteration {iteration + 1} with total shift {shift:.4f}.")
            break

    return labels, centroids

# Sample data (10 rows)
data = {
    "culmen_length_mm": [39.1, 39.5, 40.3, 36.7, 39.3, 38.9, 39.2, 41.1, 38.6, 34.6],
    "culmen_depth_mm": [18.7, 17.4, 18.0, 19.3, 20.6, 17.8, 19.6, 17.6, 21.2, 21.1]
}

df = pd.DataFrame(data)

# Impute missing values (if any)
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# Run K-Means clustering
k = 3
labels, centroids_scaled = kmeans_scratch(df_scaled, k, max_iter=100, tol=1e-4, random_state=42)

# Add cluster labels to the original dataframe
df["Cluster"] = labels

# Inverse transform centroids to original scale
centroids_unscaled = scaler.inverse_transform(centroids_scaled)

# Plotting
x_col = "culmen_length_mm"
y_col = "culmen_depth_mm"
x_idx = df_imputed.columns.get_loc(x_col)
y_idx = df_imputed.columns.get_loc(y_col)

plt.figure(figsize=(8, 6))
plt.scatter(
    df[x_col],
    df[y_col],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.7,
    edgecolors='k'
)
plt.scatter(
    centroids_unscaled[:, x_idx],
    centroids_unscaled[:, y_idx],
    s=200,
    c='red',
    marker='X',
    label='Centroids'
)


plt.xlabel("Culmen Length (mm)")
plt.ylabel("Culmen Depth (mm)")
plt.title("K-Means Clustering")
plt.colorbar(label="Cluster")
plt.legend()
plt.show()

# Save the results to a CSV file
df.to_csv("clustered_penguins.csv", index=False)
print("Clustering complete. Results saved to 'clustered_penguins.csv'.")
