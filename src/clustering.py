"""Unsupervised learning layer for RoadSentinel AI.

Includes:
1. Incident motion feature clustering (K-Means + Silhouette score validation)
2. Geospatial accident hotspot clustering (K-Means + PCA + interactive Folium map)
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def find_best_k(
    features: pd.DataFrame,
    k_range=range(2, 7),
    save_plot_path: str = "models/metrics/silhouette_curve.png",
) -> Tuple[int, Dict[int, float]]:
    """Evaluate K-Means across k in k_range, compute silhouette scores, and save curve."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(features)
        scores[k] = float(silhouette_score(features, km.labels_))

    best_k = max(scores, key=scores.get)
    print(f"Silhouette scores: {scores} -> optimal k = {best_k}")

    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.figure(figsize=(7, 4.5))
        plt.plot(list(scores.keys()), list(scores.values()), "bo-", lw=2, markersize=8)
        plt.axvline(x=best_k, color="crimson", linestyle="--", label=f"Optimal k={best_k} (Score={scores[best_k]:.3f})")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("RoadSentinel AI — K-Means Cluster Validation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    return best_k, scores


def train_severity_kmeans(features: pd.DataFrame, k: int = 3) -> KMeans:
    """Cluster vehicle motion features into severity profiles."""
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(features)


def train_spatial_hotspots(
    coords_df: pd.DataFrame,
    k_range=range(2, 8),
    save_pca_plot: str = "models/metrics/spatial_pca_clusters.png",
) -> Tuple[KMeans, StandardScaler, PCA, pd.DataFrame, Dict[int, float]]:
    """Cluster accident geospatial coordinates and generate 2D PCA projection."""
    coords = coords_df[["Start_Lat", "Start_Lng"]].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(coords)

    best_k, scores = find_best_k(
        pd.DataFrame(scaled, columns=["lat", "lng"]),
        k_range=k_range,
        save_plot_path="models/metrics/spatial_silhouette_curve.png",
    )
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(scaled)

    pca = PCA(n_components=2).fit(scaled)
    coords_pca = pca.transform(scaled)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    labeled = coords.copy()
    labeled["hotspot_cluster"] = kmeans.labels_

    if save_pca_plot:
        os.makedirs(os.path.dirname(save_pca_plot), exist_ok=True)
        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=kmeans.labels_, cmap="viridis", alpha=0.5, s=15)
        plt.colorbar(scatter, label="Hotspot Cluster ID")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(f"Geospatial Accident Clusters (PCA 2D Projection, k={best_k})")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_pca_plot, dpi=300, bbox_inches="tight")
        plt.close()

    return kmeans, scaler, pca, labeled, scores


def render_hotspot_map(
    labeled_df: pd.DataFrame,
    out_path: str = "demo/hotspot_map.html",
    sample_size: int = 1500,
) -> str:
    """Render interactive HTML Folium map with clustered accident markers."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_sample = labeled_df.sample(min(len(labeled_df), sample_size), random_state=42)
    center_lat = float(df_sample["Start_Lat"].mean())
    center_lng = float(df_sample["Start_Lng"].mean())

    m = folium.Map(location=[center_lat, center_lng], zoom_start=5, tiles="CartoDB positron")

    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]

    for _, row in df_sample.iterrows():
        cluster_id = int(row.get("hotspot_cluster", 0))
        color = colors[cluster_id % len(colors)]
        folium.CircleMarker(
            location=[row["Start_Lat"], row["Start_Lng"]],
            radius=3.5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Cluster: {cluster_id}<br>Lat: {row['Start_Lat']:.4f}<br>Lng: {row['Start_Lng']:.4f}",
        ).add_to(m)

    m.save(out_path)
    print(f"Generated interactive Folium hotspot map -> {out_path}")
    return out_path
