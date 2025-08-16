#!/usr/bin/env python3
"""
Consolidated Experiment Runner for Unsupervised Learning Assignment
Eliminates code duplication and systematically captures all required metrics

This script runs:
1. Baseline K-Means and EM clustering 
2. PCA, ICA, RP dimensionality reduction with multiple n_components
3. Clustering on reduced spaces (12 combinations)
4. Neural Network analysis on K-Means + DR combinations only
5. Comprehensive metrics collection and reporting

All metrics captured:
- Silhouette, Inertia, CH & DB scores, ARI & NMI
- BIC/AIC for EM
- DR diagnostics (variance, kurtosis, reconstruction error)
- NN speed & error comparisons
"""

import os
import random
import time
import warnings
from typing import Dict, Tuple, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Additional imports for optimal cluster selection
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, mean_squared_error
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
# Import neural network components
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

__all__ = [
    "plot_clusters_2d",
    "plot_clusters_3d",
]


def _maybe_project(X, n_components):
    """Return X projected to *n_components* with PCA when feasible.

    * If X already has <= *n_components* features **or** fewer rows than
      *n_components*, PCA is skipped.
    * If features < *n_components*, the array is zero‑padded so downstream
      code can safely index the missing dimension (useful for centroids when
      k < 3).
    """
    # Cannot fit PCA with fewer samples than components → skip.
    if X.shape[1] >= n_components and X.shape[0] >= n_components:
        return PCA(n_components=n_components, random_state=0).fit_transform(X)

    # Already low‑dimensional: either keep as‑is or pad with zeros.
    if X.shape[1] == n_components:
        return X
    pad_width = n_components - X.shape[1]
    if pad_width <= 0:
        return X[:, :n_components]  # guard (shouldn’t happen)
    return np.hstack([X, np.zeros((X.shape[0], pad_width))])

def plot_clusters_3d(
    X,
    labels,
    centers=None,
    gmm=None,
    *,
    title: str = "",
    filename: str = "",
    y_true=None,
):
    """3‑D scatter of clustering results.

    Notes
    -----
    * If ``X`` has more than 3 features, PCA projects it to 3 PCs for display.
    * For clarity, GMM ellipsoids are *not* drawn in 3‑D (they often clutter);
      instead, means are marked with black "×".  If you need full 3‑D
      ellipsoids, consider plotting only a subset of components.
    """

    X_vis = _maybe_project(X, 3)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Points
    for c in np.unique(labels):
        idx = labels == c
        ax.scatter(
            X_vis[idx, 0], X_vis[idx, 1], X_vis[idx, 2],
            s=10, alpha=0.65, label=f"Cluster {c}"
        )

    # Centroids or GMM means
    marker_kwargs = dict(c="k", s=60, marker="x", linewidths=1.5, depthshade=False)
    if centers is not None:
        centers_vis = _maybe_project(centers, 3)
        ax.scatter(centers_vis[:, 0], centers_vis[:, 1], centers_vis[:, 2], **marker_kwargs, label="Centroid")
    elif gmm is not None:
        means_vis = _maybe_project(gmm.means_, 3)
        ax.scatter(means_vis[:, 0], means_vis[:, 1], means_vis[:, 2], **marker_kwargs, label="Mean")

    # Title with ARI
    if y_true is not None:
        title += f"\n(ARI = {adjusted_rand_score(y_true, labels):.2f})"

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

    # A bit of padding for aesthetics
    for dim in range(3):
        xyz = X_vis[:, dim]
        pad = 0.05 * (xyz.max() - xyz.min())
        getattr(ax, f"set_{'xyz'[dim]}lim")(xyz.min() - pad, xyz.max() + pad)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Saved 3‑D cluster plot → {filename}")
    plt.close()


def plot_clusters_2d(X, labels, centers=None, gmm=None,
                     title: str = "", filename: str = "", y_true=None):
    """
    Visualise clustering results in 2‑D by projecting *X* onto the first two
    principal components when necessary.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster assignments.
    centers : array-like of shape (n_clusters, n_features), optional
        K‑means centroids; plotted as black "×" markers.
    gmm : sklearn.mixture.GaussianMixture, optional
        Fitted GMM; draws 1‑σ covariance ellipses for each component.
    title : str, optional
        Title for the plot.
    filename : str, optional
        If provided, the figure is saved to this path (PNG).
    y_true : array-like of shape (n_samples,), optional
        Ground‑truth labels; if provided, the Adjusted Rand Index is appended
        to the title.
    """
    # 1. Reduce dimensionality for plotting
    if X.shape[1] > 2:
        X_vis = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        X_vis = X

    plt.figure(figsize=(6, 5))

    # 2. Scatter by cluster label
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(X_vis[idx, 0], X_vis[idx, 1], s=12, alpha=0.6,
                    label=f"Cluster {c}")

    # 3. Overlay k‑means centroids if supplied
    if centers is not None:
        if centers.shape[1] > 2:
            centers_vis = PCA(n_components=2, random_state=0).fit_transform(centers)
        else:
            centers_vis = centers
        plt.scatter(centers_vis[:, 0], centers_vis[:, 1], c="k", s=120,
                    marker="x", linewidths=2, label="Centroid")

    # 4. Overlay GMM covariance ellipses if supplied
    if gmm is not None:
        means = gmm.means_
        covs = gmm.covariances_
        for mean, cov in zip(means, covs):
            # Handle full vs. tied/diag covariance shapes
            if cov.ndim == 3:  # full covariance per component
                cov2d = cov[:2, :2]
            else:             # tied or diag
                cov2d = np.diag(cov[:2]) if cov.ndim == 1 else cov[:2, :2]

            eigvals, eigvecs = np.linalg.eigh(cov2d)
            width, height = 2 * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

            ell = Ellipse(mean[:2], width, height, angle=angle,
                          edgecolor="k", facecolor="none", lw=1.5, alpha=0.6)
            plt.gca().add_patch(ell)

    # 5. Append ARI to the title if ground truth provided
    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        title = f"{title}\n(ARI = {ari:.2f})"

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    # 6. Save or show
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Saved cluster plot → {filename}")
    plt.close()


# Helper function to round numerical values to 2 decimal places
def round_numerical_values(value, decimals=2):
    """Round numerical values to specified decimal places, handling None and non-numeric values"""
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if np.isnan(value) or np.isinf(value):
            return value
        return round(float(value), decimals)
    return value


def round_dict_values(data_dict, decimals=2):
    """Recursively round all numerical values in a dictionary to specified decimal places"""
    if isinstance(data_dict, dict):
        return {key: round_dict_values(val, decimals) for key, val in data_dict.items()}
    elif isinstance(data_dict, list):
        return [round_dict_values(item, decimals) for item in data_dict]
    else:
        return round_numerical_values(data_dict, decimals)

# Comprehensive results storage and management
class SimpleResultsManager:
    def __init__(self):
        self.kmeans_results = []
        self.em_results = []
        self.nn_results = []
        self.dr_results = []
        self.all_results = {}

    def add_kmeans_results(self, **kwargs):
        """Store K-Means results"""
        result_entry = {
            'timestamp': time.time(),
            'experiment_type': 'kmeans',
        }
        rounded_kwargs = round_dict_values(kwargs)
        if isinstance(rounded_kwargs, dict):
            result_entry.update(rounded_kwargs)
        self.kmeans_results.append(result_entry)
        
    def add_em_results(self, **kwargs):
        """Store EM results"""
        result_entry = {
            'timestamp': time.time(),
            'experiment_type': 'em',
        }
        rounded_kwargs = round_dict_values(kwargs)
        if isinstance(rounded_kwargs, dict):
            result_entry.update(rounded_kwargs)
        self.em_results.append(result_entry)

    def add_nn_results(self, **kwargs):
        """Store Neural Network results"""
        result_entry = {
            'timestamp': time.time(),
            'experiment_type': 'neural_network',
        }
        rounded_kwargs = round_dict_values(kwargs)
        if isinstance(rounded_kwargs, dict):
            result_entry.update(rounded_kwargs)
        self.nn_results.append(result_entry)

    def add_dr_results(self, **kwargs):
        """Store Dimensionality Reduction results"""
        result_entry = {
            'timestamp': time.time(),
            'experiment_type': 'dimensionality_reduction',
        }
        rounded_kwargs = round_dict_values(kwargs)
        if isinstance(rounded_kwargs, dict):
            result_entry.update(rounded_kwargs)
        self.dr_results.append(result_entry)

    def print_summary_table(self, table_type='comprehensive'):
        """Print summary of results"""
        print(f"\n📊 {table_type.upper()} RESULTS SUMMARY")
        print("=" * 80)

        if table_type == 'kmeans' and self.kmeans_results:
            print(f"K-Means Results: {len(self.kmeans_results)} experiments")
            for result in self.kmeans_results[-3:]:  # Show last 3
                print(f"  - {result.get('dataset', 'Unknown')}: {result.get('dr_method', 'baseline')} method")

        elif table_type == 'em' and self.em_results:
            print(f"EM Results: {len(self.em_results)} experiments")
            for result in self.em_results[-3:]:  # Show last 3
                print(f"  - {result.get('dataset', 'Unknown')}: {result.get('dr_method', 'baseline')} method")

        elif table_type == 'nn' and self.nn_results:
            print(f"Neural Network Results: {len(self.nn_results)} experiments")
            for result in self.nn_results[-3:]:  # Show last 3
                print(
                    f"  - {result.get('dataset_name', 'Unknown')}: {result.get('dr_method', 'unknown')} method, Accuracy: {result.get('accuracy', 0):.2f}")

        elif table_type == 'dr' and self.dr_results:
            print(f"Dimensionality Reduction Results: {len(self.dr_results)} experiments")
            for result in self.dr_results[-3:]:  # Show last 3
                print(
                    f"  - {result.get('dataset_name', 'Unknown')}: {result.get('dr_method', 'unknown')} method, Components: {result.get('n_components', 'N/A')}")

        elif table_type == 'comprehensive':
            total_experiments = len(self.kmeans_results) + len(self.em_results) + len(self.nn_results) + len(
                self.dr_results)
            print(f"Total Experiments: {total_experiments}")
            print(f"  - K-Means: {len(self.kmeans_results)}")
            print(f"  - EM: {len(self.em_results)}")
            print(f"  - Neural Networks: {len(self.nn_results)}")
            print(f"  - Dimensionality Reduction: {len(self.dr_results)}")

    def save_all_tables(self, prefix='consolidated'):
        """Save all results to CSV files"""
        import json

        # Create results directory
        os.makedirs('results', exist_ok=True)

        saved_files = []

        # Save K-Means results (filter for optimal only in consolidated results)
        if self.kmeans_results:
            kmeans_df = pd.DataFrame(self.kmeans_results)

            # For consolidated results, only keep optimal configurations
            if 'final' in prefix or 'consolidated' in prefix:
                optimal_kmeans_df = kmeans_df[kmeans_df['is_optimal'] == 1.0].copy()
                if len(optimal_kmeans_df) > 0:
                    kmeans_file = f'results/{prefix}_kmeans_results.csv'
                    optimal_kmeans_df.to_csv(kmeans_file, index=False)
                    saved_files.append(kmeans_file)
                    print(f"   💾 K-Means results saved: {kmeans_file} ({len(optimal_kmeans_df)} optimal configs)")
                else:
                    print(f"   ⚠️ No optimal K-Means results found for {prefix}")
            else:
                # For individual dataset results, save all
                kmeans_file = f'results/{prefix}_kmeans_results.csv'
                kmeans_df.to_csv(kmeans_file, index=False)
                saved_files.append(kmeans_file)
                print(f"   💾 K-Means results saved: {kmeans_file} (all {len(kmeans_df)} results)")

        # Save EM results (filter for optimal only in consolidated results)
        if self.em_results:
            em_df = pd.DataFrame(self.em_results)

            # For consolidated results, only keep optimal configurations
            if 'final' in prefix or 'consolidated' in prefix:
                optimal_em_df = em_df[em_df['is_optimal'] == 1.0].copy()
                if len(optimal_em_df) > 0:
                    em_file = f'results/{prefix}_em_results.csv'
                    optimal_em_df.to_csv(em_file, index=False)
                    saved_files.append(em_file)
                    print(f"   💾 EM results saved: {em_file} ({len(optimal_em_df)} optimal configs)")
                else:
                    print(f"   ⚠️ No optimal EM results found for {prefix}")
            else:
                # For individual dataset results, save all
                em_file = f'results/{prefix}_em_results.csv'
                em_df.to_csv(em_file, index=False)
                saved_files.append(em_file)
                print(f"   💾 EM results saved: {em_file} (all {len(em_df)} results)")

        # Save NN results (no filtering needed - already optimal only)
        if self.nn_results:
            nn_df = pd.DataFrame(self.nn_results)
            nn_file = f'results/{prefix}_nn_results.csv'
            nn_df.to_csv(nn_file, index=False)
            saved_files.append(nn_file)
            print(f"   💾 Neural Network results saved: {nn_file}")

        # Save DR results (no filtering needed - one per method/dataset combination)
        if self.dr_results:
            dr_df = pd.DataFrame(self.dr_results)
            dr_file = f'results/{prefix}_dr_results.csv'
            dr_df.to_csv(dr_file, index=False)
            saved_files.append(dr_file)
            print(f"   💾 Dimensionality Reduction results saved: {dr_file}")

        # Create comprehensive summary
        if saved_files:
            summary = {
                'total_experiments': len(self.kmeans_results) + len(self.em_results) + len(self.nn_results) + len(
                    self.dr_results),
                'kmeans_count': len(self.kmeans_results),
                'em_count': len(self.em_results),
                'nn_count': len(self.nn_results),
                'dr_count': len(self.dr_results),
                'saved_files': saved_files,
                'timestamp': time.time(),
                'filtering_applied': 'final' in prefix or 'consolidated' in prefix
            }

            summary_file = f'results/{prefix}_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"   📋 Summary saved: {summary_file}")

        return saved_files

    def save_results_json(self, filename='results.json'):
        """Save all results to a single JSON file"""
        import json

        os.makedirs('results', exist_ok=True)

        all_data = {
            'kmeans_results': self.kmeans_results,
            'em_results': self.em_results,
            'nn_results': self.nn_results,
            'dr_results': self.dr_results,
            'metadata': {
                'total_experiments': len(self.kmeans_results) + len(self.em_results) + len(self.nn_results) + len(
                    self.dr_results),
                'export_timestamp': time.time()
            }
        }

        filepath = f'results/{filename}'
        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"   💾 All results saved to JSON: {filepath}")


global_results = SimpleResultsManager()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global settings for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('figures/consolidated', exist_ok=True)
os.makedirs('figures/nndr', exist_ok=True)
os.makedirs('figures/pca', exist_ok=True)
os.makedirs('figures/ica', exist_ok=True)
os.makedirs('figures/rp', exist_ok=True)


class OptimalClusterSelector:
    """
    Class containing methods for automatically finding optimal number of clusters
    """

    @staticmethod
    def elbow_method(X: np.ndarray, max_k: int = 15, random_state: int = RANDOM_STATE) -> Tuple[int, Dict[str, Any]]:
        """
        Find optimal number of clusters using the elbow method
        """
        k_range = range(2, min(max_k + 1, len(X) // 2))
        distortions = []
        inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            kmeans.fit(X)
            distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
            inertias.append(kmeans.inertia_)

        # Calculate rate of change (second derivative)
        if len(inertias) >= 3:
            # Find elbow using rate of change
            rate_changes = []
            for i in range(1, len(inertias) - 1):
                rate_change = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
                rate_changes.append(rate_change)

            # Find the maximum rate of change (elbow point)
            elbow_idx = np.argmax(rate_changes) + 1  # +1 because rate_changes starts from index 1
            optimal_k = list(k_range)[elbow_idx]
        else:
            optimal_k = 2  # Default fallback

        diagnostics = {
            'k_range': list(k_range),
            'inertias': inertias,
            'distortions': distortions,
            'method': 'elbow'
        }

        return optimal_k, diagnostics

    @staticmethod
    def silhouette_method(X: np.ndarray, max_k: int = 15, algorithm: str = 'kmeans',
                          random_state: int = RANDOM_STATE) -> Tuple[int, Dict[str, Any]]:
        """
        Find optimal number of clusters using silhouette analysis
        """
        k_range = range(2, min(max_k + 1, len(X) // 2))
        silhouette_scores = []

        for k in k_range:
            if algorithm.lower() == 'kmeans':
                model = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
                labels = model.fit_predict(X)
            elif algorithm.lower() == 'em':
                # Optimized EM parameters for silhouette analysis
                model = GaussianMixture(
                    n_components=k,
                    random_state=random_state,
                    n_init=1,  # Reduced for speed
                    max_iter=50,  # Reduced for speed
                    tol=1e-3,
                    covariance_type='diag'  # Faster than 'full'
                )
                labels = model.fit_predict(X)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            if len(np.unique(labels)) > 1:
                silhouette_avg = silhouette_score(X, labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(-1)

        optimal_k = list(k_range)[np.argmax(silhouette_scores)]

        diagnostics = {
            'k_range': list(k_range),
            'silhouette_scores': silhouette_scores,
            'method': 'silhouette',
            'algorithm': algorithm
        }

        return optimal_k, diagnostics

    @staticmethod
    def information_criteria_combined(X: np.ndarray, max_k: int = 6,
                                      random_state: int = RANDOM_STATE) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Find optimal number of components for EM using BOTH BIC and AIC in single pass - OPTIMIZED
        """
        k_range = range(1, min(max_k + 1, len(X) // 3))  # More conservative range
        bic_scores = []
        aic_scores = []
        fitted_models = {}  # Cache fitted models

        print(f"      🚀 Fast EM optimization: testing {len(list(k_range))} components with reduced iterations...")

        for k in k_range:
            # Use minimal parameters for speed during optimization
            gmm = GaussianMixture(
                n_components=k,
                random_state=random_state,
                n_init=1,  # Reduced from 3
                max_iter=50,  # Reduced from default 100
                tol=1e-3,  # Slightly relaxed tolerance
                covariance_type='diag'  # Faster than 'full'
            )
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
            fitted_models[k] = gmm  # Cache for potential reuse

        optimal_bic = list(k_range)[np.argmin(bic_scores)]
        optimal_aic = list(k_range)[np.argmin(aic_scores)]

        diagnostics = {
            'k_range': list(k_range),
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'method': 'information_criteria_combined_optimized',
            'fitted_models': fitted_models  # Include cached models
        }

        return {'bic': optimal_bic, 'aic': optimal_aic}, diagnostics

    @staticmethod
    def comprehensive_cluster_selection(X: np.ndarray, algorithm: str = 'kmeans',
                                        max_k: int = 8, random_state: int = RANDOM_STATE) -> Tuple[
        # Further reduced max_k for speed
        int, Dict[str, Any]]:
        """
        Use multiple methods to find optimal number of clusters and return consensus
        """
        methods_results = {}
        optimal_ks = []

        if algorithm.lower() == 'kmeans':
            # Elbow method
            try:
                k_elbow, diag_elbow = OptimalClusterSelector.elbow_method(X, max_k, random_state)
                methods_results['elbow'] = {'k': k_elbow, 'diagnostics': diag_elbow}
                optimal_ks.append(k_elbow)
            except Exception as e:
                print(f"   Warning: Elbow method failed: {e}")

            # Silhouette method
            try:
                k_sil, diag_sil = OptimalClusterSelector.silhouette_method(X, max_k, 'kmeans', random_state)
                methods_results['silhouette'] = {'k': k_sil, 'diagnostics': diag_sil}
                optimal_ks.append(k_sil)
            except Exception as e:
                print(f"   Warning: Silhouette method failed: {e}")

        elif algorithm.lower() == 'em':
            # Re-enabled optimized silhouette method for EM
            try:
                em_max_k = min(max_k, 6)  # Conservative for EM speed
                k_sil, diag_sil = OptimalClusterSelector.silhouette_method(X, em_max_k, 'em', random_state)
                methods_results['silhouette'] = {'k': k_sil, 'diagnostics': diag_sil}
                optimal_ks.append(k_sil)
                print(f"      ✅ Silhouette method completed (EM optimized)")
            except Exception as e:
                print(f"   Warning: Silhouette method failed: {e}")

            # Combined BIC/AIC method (SINGLE PASS) - PRIMARY METHOD
            try:
                # Use smaller max_k for EM optimization
                em_max_k = min(max_k, 6)  # Even more conservative for EM
                optimal_dict, diag_combined = OptimalClusterSelector.information_criteria_combined(X, em_max_k,
                                                                                                   random_state)

                # Add BIC results
                methods_results['bic'] = {'k': optimal_dict['bic'], 'diagnostics': diag_combined}
                optimal_ks.append(optimal_dict['bic'])

                # Add AIC results (same diagnostics, different optimal)
                methods_results['aic'] = {'k': optimal_dict['aic'], 'diagnostics': diag_combined}
                optimal_ks.append(optimal_dict['aic'])
                
            except Exception as e:
                print(f"   Warning: Information criteria method failed: {e}")

        # Consensus: use mode or median of all methods
        if optimal_ks:
            # Use mode (most common) or median if no clear mode
            from collections import Counter
            k_counts = Counter(optimal_ks)
            if len(k_counts) > 0:
                # If there's a clear mode, use it
                most_common_k, count = k_counts.most_common(1)[0]
                if count > 1 or len(optimal_ks) == 1:
                    consensus_k = most_common_k
                else:
                    # No clear consensus, use median
                    consensus_k = int(np.median(optimal_ks))
            else:
                consensus_k = 2  # fallback
        else:
            consensus_k = 2  # fallback if all methods failed

        diagnostics = {
            'all_methods': methods_results,
            'optimal_ks': optimal_ks,
            'consensus_k': consensus_k,
            'algorithm': algorithm,
            'method': 'comprehensive'
        }

        return consensus_k, diagnostics

    @staticmethod
    def plot_cluster_selection_diagnostics(diagnostics: Dict[str, Any], dataset_name: str,
                                           algorithm: str, dr_method: str = "baseline",
                                           n_components: Optional[int] = None):
        """
        Create comprehensive visualization of cluster selection process
        """
        if 'all_methods' not in diagnostics or not diagnostics['all_methods']:
            return

        methods = list(diagnostics['all_methods'].keys())
        n_methods = len(methods)

        if n_methods == 0:
            return

        # Create subplot grid
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))
        if n_methods == 1:
            axes = [axes]

        fig.suptitle(f'{algorithm.upper()} Cluster Selection - {dataset_name}\n'
                     f'DR Method: {dr_method}, Components: {n_components or "Original"}',
                     fontsize=14, fontweight='bold')

        for i, method in enumerate(methods):
            method_data = diagnostics['all_methods'][method]
            method_diagnostics = method_data['diagnostics']
            optimal_k = method_data['k']

            ax = axes[i]

            if method == 'elbow':
                k_range = method_diagnostics['k_range']
                inertias = method_diagnostics['inertias']
                ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
                ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                           label=f'Optimal k={optimal_k}')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Inertia')
                ax.set_title(f'Elbow Method\nOptimal: {optimal_k} clusters')
                ax.grid(True, alpha=0.3)
                ax.legend()

            elif method == 'silhouette':
                k_range = method_diagnostics['k_range']
                silhouette_scores = method_diagnostics['silhouette_scores']
                ax.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
                ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                           label=f'Optimal k={optimal_k}')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Silhouette Score')
                ax.set_title(f'Silhouette Method\nOptimal: {optimal_k} clusters')
                ax.grid(True, alpha=0.3)
                ax.legend()

            elif method == 'gap':
                k_range = method_diagnostics['k_range']
                gaps = method_diagnostics['gaps']
                ax.plot(k_range, gaps, 'mo-', linewidth=2, markersize=8)
                ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                           label=f'Optimal k={optimal_k}')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Gap Statistic')
                ax.set_title(f'Gap Statistic\nOptimal: {optimal_k} clusters')
                ax.grid(True, alpha=0.3)
                ax.legend()

            elif method in ['bic', 'aic']:
                k_range = method_diagnostics['k_range']
                # Use the correct key names for BIC and AIC scores
                scores = method_diagnostics['bic_scores'] if method == 'bic' else method_diagnostics['aic_scores']
                color = 'co' if method == 'bic' else 'yo'
                ax.plot(k_range, scores, color + '-', linewidth=2, markersize=8)
                ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                           label=f'Optimal k={optimal_k}')
                ax.set_xlabel('Number of Components')
                ax.set_ylabel(f'{method.upper()} Score')
                ax.set_title(f'{method.upper()} Method\nOptimal: {optimal_k} components')
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()

        # Save plot
        plot_filename = f"{dataset_name}_{algorithm}_{dr_method}"
        if n_components:
            plot_filename += f"_{n_components}"
        plot_filename += "_cluster_selection.png"

        plt.savefig(f"figures/consolidated/{plot_filename}", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 Cluster selection diagnostics plot saved: {plot_filename}")


class ConsolidatedExperiments:
    """
    Consolidated experiment runner that eliminates duplication and captures all metrics
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.results = {
            'baseline_clustering': [],
            'dr_clustering': [],
            'nn_analysis': [],
            'dr_diagnostics': []
        }

    def load_and_preprocess_data(self, file_path: str, dataset_name: str) -> Tuple[
        np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Single data loading function to eliminate duplication across modules
        """
        print(f"📊 Loading and preprocessing {dataset_name} from {file_path}")

        # Dataset-specific configuration
        label_columns = {
            "CSDataSet": "Cancer_Stage",
            "BankrupcyDS": "Bankrupt?"
        }

        # Non-feature columns to remove (excluding labels)
        id_columns_to_remove = {
            "CSDataSet": ["Patient_ID"],
            "BankrupcyDS": []
        }

        # Load data
        data = pd.read_csv(file_path).dropna()
        data.columns = data.columns.str.strip()

        print(f"   Original data shape: {data.shape}")
        print(f"   Label column: {label_columns[dataset_name]}")
        print(f"   ID columns to remove: {id_columns_to_remove[dataset_name]}")

        # Split into train/test BEFORE removing any columns
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=self.random_state,
            stratify=data[label_columns[dataset_name]] if dataset_name in label_columns else None
        )

        # Convert back to DataFrames
        train_df = pd.DataFrame(train_data, columns=data.columns)
        test_df = pd.DataFrame(test_data, columns=data.columns)

        print(f"   After train/test split: Train {train_df.shape}, Test {test_df.shape}")

        # Extract labels from BOTH train and test sets
        y_train = train_df[label_columns[dataset_name]].copy()
        y_test = test_df[label_columns[dataset_name]].copy()

        print(f"   Labels extracted: Train {len(y_train)}, Test {len(y_test)}")
        print(f"   Label distribution in train: {y_train.value_counts().to_dict()}")
        print(f"   Label distribution in test: {y_test.value_counts().to_dict()}")

        # Remove label column from BOTH train and test sets
        train_df = train_df.drop(columns=[label_columns[dataset_name]])
        test_df = test_df.drop(columns=[label_columns[dataset_name]])

        # Remove ID columns from BOTH train and test sets
        if id_columns_to_remove[dataset_name]:
            train_df = train_df.drop(columns=id_columns_to_remove[dataset_name])
            test_df = test_df.drop(columns=id_columns_to_remove[dataset_name])

        print(f"   After removing labels and IDs: Train {train_df.shape}, Test {test_df.shape}")

        # Identify column types
        cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object']
        num_cols = [c for c in train_df.columns if c not in cat_cols]

        print(f"   Categorical columns: {len(cat_cols)}")
        print(f"   Numerical columns: {len(num_cols)}")

        # Create preprocessing pipeline
        transformers = []
        if num_cols:
            transformers.append(('num', StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols))

        if transformers:
            preprocessor = ColumnTransformer(transformers)
            preprocessor.fit(train_df)
            X_train = preprocessor.transform(train_df)
            X_test = preprocessor.transform(test_df)
        else:
            X_train = train_df.values
            X_test = test_df.values

        # Ensure we have numpy arrays
        if hasattr(X_train, 'toarray') and callable(getattr(X_train, 'toarray', None)):
            X_train = X_train.toarray()
        if hasattr(X_test, 'toarray') and callable(getattr(X_test, 'toarray', None)):
            X_test = X_test.toarray()

        # Convert to numpy arrays if they aren't already
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        print(f"   Processed data shape: Train {X_train.shape}, Test {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def compute_comprehensive_metrics(self, X: np.ndarray, labels: np.ndarray,
                                      y_true: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, float]:
        """
        Compute all clustering metrics in one place to eliminate duplication
        """
        metrics = {}

        # Internal validation metrics
        if len(np.unique(labels)) > 1:
            metrics['silhouette'] = round_numerical_values(silhouette_score(X, labels))
            metrics['calinski_harabasz'] = round_numerical_values(calinski_harabasz_score(X, labels))
            metrics['davies_bouldin'] = round_numerical_values(davies_bouldin_score(X, labels))
        else:
            metrics['silhouette'] = -1.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = np.inf

        # External validation metrics (if ground truth available)
        if y_true is not None:
            # Handle categorical labels
            if hasattr(y_true, 'dtype') and y_true.dtype == 'object':
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
            elif isinstance(y_true, pd.Series):
                if y_true.dtype == 'object':
                    le = LabelEncoder()
                    y_true_encoded = le.fit_transform(y_true.values)
                else:
                    y_true_encoded = y_true.values
            else:
                y_true_encoded = y_true

            metrics['ari'] = round_numerical_values(adjusted_rand_score(y_true_encoded, labels))
            metrics['nmi'] = round_numerical_values(normalized_mutual_info_score(y_true_encoded, labels))
        else:
            metrics['ari'] = None
            metrics['nmi'] = None

        return metrics

    def run_kmeans_experiment(self, X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series,
                              dataset_name: str, dr_method: str = "baseline", n_components: Optional[int] = None) -> \
    Dict[str, Any]:
        """
        Run K-Means clustering with automatic optimal cluster selection
        """
        print(f"   🎯 Finding optimal number of clusters for K-Means using multiple methods...")

        # Use comprehensive cluster selection with reduced max_k for speed
        max_k = min(8, len(X_train) // 20)  # More conservative upper bound for speed
        optimal_k, cluster_diagnostics = OptimalClusterSelector.comprehensive_cluster_selection(
            X_train, algorithm='kmeans', max_k=max_k, random_state=self.random_state
        )

        print(f"   ✅ Optimal K-Means clusters: {optimal_k}")
        print(f"      Methods used: {list(cluster_diagnostics['all_methods'].keys())}")
        if len(cluster_diagnostics['optimal_ks']) > 1:
            print(f"      Individual method results: {cluster_diagnostics['optimal_ks']}")

        # Test the optimal number and a few neighbors for comparison
        test_range = [max(2, optimal_k - 1), optimal_k, optimal_k + 1]
        test_range = [k for k in test_range if k <= max_k]  # Keep within bounds
        test_range = list(set(test_range))  # Remove duplicates
        
        results = []
        for n_clusters in test_range:
            # Fit K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
            labels = kmeans.fit_predict(X_train)

            # Compute all metrics
            metrics = self.compute_comprehensive_metrics(X_train, labels, y_train)

            # Add K-Means specific metrics
            result_dict = {
                'n_clusters': n_clusters,
                'inertia': round_numerical_values(kmeans.inertia_),
                'algorithm': 'K-Means',
                'dataset': dataset_name,
                'dr_method': dr_method,
                'n_components': n_components or X_train.shape[1],
                'is_optimal': n_clusters == optimal_k
            }
            result_dict.update(metrics)
            results.append(result_dict)

        # Store in global results manager
        results_df = pd.DataFrame(results)
        best_result = results_df.loc[results_df['silhouette'].idxmax()]

        # Create diagnostic visualization
        OptimalClusterSelector.plot_cluster_selection_diagnostics(
            cluster_diagnostics, dataset_name, 'kmeans', dr_method, n_components
        )

        # Store each result as a separate row instead of storing the DataFrame as JSON
        for result in results:
            global_results.add_kmeans_results(**result)

        return {
            'results': results,
            'best_config': best_result.to_dict(),
            'optimal_k': optimal_k,
            'cluster_selection_diagnostics': cluster_diagnostics,
            'summary': f"Optimal K-Means: {optimal_k} clusters (consensus), Best performance: {best_result['n_clusters']} clusters, Silhouette: {best_result['silhouette']:.2f}"
        }

    def run_em_experiment(self, X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series,
                          dataset_name: str, dr_method: str = "baseline", n_components: Optional[int] = None) -> Dict[
        str, Any]:
        """
        Run EM/GMM clustering with automatic optimal component selection
        """
        print(f"   🎯 Finding optimal number of components for EM using multiple methods...")

        # Use comprehensive cluster selection with reduced max_k for speed
        max_k = min(6, len(X_train) // 30)  # Even more conservative upper bound for EM
        optimal_k, cluster_diagnostics = OptimalClusterSelector.comprehensive_cluster_selection(
            X_train, algorithm='em', max_k=max_k, random_state=self.random_state
        )

        print(f"   ✅ Optimal EM components: {optimal_k}")
        print(f"      Methods used: {list(cluster_diagnostics['all_methods'].keys())}")
        if len(cluster_diagnostics['optimal_ks']) > 1:
            print(f"      Individual method results: {cluster_diagnostics['optimal_ks']}")

        # Get cached models from cluster selection to avoid redundant fitting
        cached_models = {}
        if 'all_methods' in cluster_diagnostics:
            for method_name, method_data in cluster_diagnostics['all_methods'].items():
                if 'diagnostics' in method_data and 'fitted_models' in method_data['diagnostics']:
                    cached_models = method_data['diagnostics']['fitted_models']
                    break

        # Test only the optimal number (skip neighbors for speed)
        test_range = [optimal_k]
        
        results = []
        for n_comp in test_range:
            # Try to use cached model first
            if n_comp in cached_models:
                print(f"      🚀 Using cached GMM model for {n_comp} components")
                gmm = cached_models[n_comp]
                labels = gmm.predict(X_train)
            else:
                print(f"      🔄 Fitting new GMM model for {n_comp} components")
                # Fit EM/GMM with optimized parameters for final model
                gmm = GaussianMixture(
                    n_components=n_comp,
                    random_state=self.random_state,
                    n_init=2,  # Reduced from 3
                    max_iter=100,
                    tol=1e-4,
                    covariance_type='diag'  # Faster than 'full'
                )
                labels = gmm.fit_predict(X_train)

            # Compute all metrics
            metrics = self.compute_comprehensive_metrics(X_train, labels, y_train)

            # Add EM specific metrics
            result_dict = {
                'n_clusters': n_comp,
                'bic': round_numerical_values(gmm.bic(X_train)),
                'aic': round_numerical_values(gmm.aic(X_train)),
                'lower_bound': round_numerical_values(gmm.lower_bound_),
                'algorithm': 'EM',
                'dataset': dataset_name,
                'dr_method': dr_method,
                'n_components': n_components or X_train.shape[1],
                'is_optimal': n_comp == optimal_k,
                'used_cached_model': n_comp in cached_models
            }
            result_dict.update(metrics)
            results.append(result_dict)

        # Store in global results manager
        results_df = pd.DataFrame(results)
        best_result = results_df.loc[results_df['silhouette'].idxmax()]

        # Create diagnostic visualization
        OptimalClusterSelector.plot_cluster_selection_diagnostics(
            cluster_diagnostics, dataset_name, 'em', dr_method, n_components
        )

        # Store each result as a separate row instead of storing the DataFrame as JSON
        for result in results:
            global_results.add_em_results(**result)

        return {
            'results': results,
            'best_config': best_result.to_dict(),
            'optimal_k': optimal_k,
            'cluster_selection_diagnostics': cluster_diagnostics,
            'summary': f"Optimal EM: {optimal_k} components (consensus), Best performance: {best_result['n_clusters']} components, Silhouette: {best_result['silhouette']:.2f}, BIC: {best_result['bic']:.2f}"
        }

    def optimize_pca_components(self, X_train: np.ndarray, target_variance: float = 0.95) -> Tuple[int, Dict[str, Any]]:
        """
        Hyperparameter search for PCA: Find optimal number of components based on explained variance
        """
        print(f"      🔍 Optimizing PCA components for {target_variance * 100}% explained variance...")

        # Fit PCA with all components to analyze variance
        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(X_train)

        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # Find optimal number of components for target variance
        optimal_components = np.argmax(cumulative_variance >= target_variance) + 1

        # If target variance cannot be reached, use 95% of available components
        if cumulative_variance[-1] < target_variance:
            optimal_components = int(0.95 * len(cumulative_variance))
            achieved_variance = cumulative_variance[optimal_components - 1]
            print(
                f"         ⚠️  Target variance {target_variance * 100}% not achievable. Using {optimal_components} components for {achieved_variance * 100:.2f}% variance")
        else:
            achieved_variance = cumulative_variance[optimal_components - 1]
            print(
                f"         ✅ Optimal components: {optimal_components} (achieves {achieved_variance * 100:.2f}% variance)")

        # Test different variance thresholds for comparison
        thresholds = [0.85, 0.90, 0.95, 0.99]
        threshold_analysis = {}

        for threshold in thresholds:
            if threshold <= cumulative_variance[-1]:
                n_comp = np.argmax(cumulative_variance >= threshold) + 1
                threshold_analysis[threshold] = {
                    'components': n_comp,
                    'achieved_variance': round_numerical_values(cumulative_variance[n_comp - 1])
                }

        diagnostics = {
            'target_variance': target_variance,
            'optimal_components': optimal_components,
            'achieved_variance': round_numerical_values(achieved_variance),
            'total_components': len(cumulative_variance),
            'threshold_analysis': threshold_analysis,
            'all_explained_variance': [round_numerical_values(x) for x in pca_full.explained_variance_ratio_],
            'all_eigenvalues': [round_numerical_values(x) for x in pca_full.explained_variance_],
            'cumulative_variance': [round_numerical_values(x) for x in cumulative_variance]
        }

        return optimal_components, diagnostics

    def optimize_ica_components(self, X_train: np.ndarray, max_components: Optional[int] = None) -> Tuple[
        int, Dict[str, Any]]:
        """
        Find optimal number of ICA components based on kurtosis maximization - OPTIMIZED
        """
        print(f"      🚀 Fast ICA optimization with reduced iterations and deflation algorithm...")

        if max_components is None:
            max_components = min(X_train.shape[1], 15)  # Reduced from 20

        # Use sparse sampling for speed - test fewer components
        max_test = min(max_components, 10)  # Test at most 10 different values
        if max_test <= 5:
            component_range = range(2, max_test + 1)
        else:
            # Sample key points: start, middle, end + a few in between
            component_range = [2, max_test // 4, max_test // 2, 3 * max_test // 4, max_test]
            component_range = list(set(component_range))  # Remove duplicates
            component_range.sort()

        print(f"         Testing {len(component_range)} component configurations: {component_range}")

        kurtosis_scores = []
        convergence_info = []

        for n_comp in component_range:
            try:
                # Optimized ICA parameters for speed
                ica = FastICA(
                    n_components=n_comp,
                    random_state=self.random_state,
                    algorithm='deflation',  # Keep deflation as required
                    max_iter=200,  # Reduced from 1000
                    tol=5e-4,  # Slightly relaxed from 1e-4
                    whiten='unit-variance',  # Can improve convergence speed
                    fun='logcosh'  # Generally faster contrast function
                )
                components = ica.fit_transform(X_train)

                # Calculate average absolute kurtosis
                component_kurtosis = []
                for i in range(n_comp):
                    if components is not None and i < components.shape[1]:
                        k = kurtosis(components[:, i])
                        if k is not None and not np.isnan(k):
                            component_kurtosis.append(abs(k))
                        else:
                            component_kurtosis.append(0.0)
                    else:
                        component_kurtosis.append(0.0)
                avg_kurtosis = np.mean(component_kurtosis) if component_kurtosis else 0.0
                kurtosis_scores.append(avg_kurtosis)

                convergence_info.append({
                    'n_components': n_comp,
                    'converged': ica.n_iter_ < ica.max_iter,
                    'n_iterations': ica.n_iter_,
                    'avg_kurtosis': round_numerical_values(avg_kurtosis)
                })

            except Exception as e:
                print(f"         ⚠️ ICA failed for {n_comp} components: {e}")
                kurtosis_scores.append(0.0)
                convergence_info.append({
                    'n_components': n_comp,
                    'converged': False,
                    'n_iterations': 0,
                    'avg_kurtosis': 0.0
                })

        # Find optimal number of components (highest average kurtosis)
        if kurtosis_scores:
            optimal_idx = np.argmax(kurtosis_scores)
            optimal_components = list(component_range)[optimal_idx]
            optimal_kurtosis = kurtosis_scores[optimal_idx]
        else:
            optimal_components = 5  # Fallback
            optimal_kurtosis = 0.0

        print(f"         ✅ Optimal ICA components: {optimal_components} (avg kurtosis: {optimal_kurtosis:.2f})")

        # Report optimization savings
        original_tests = max_components - 1  # What we would have tested originally
        actual_tests = len(component_range)
        time_saved_pct = (1 - actual_tests / original_tests) * 100 if original_tests > 0 else 0
        print(
            f"         ⚡ Speed optimization: tested {actual_tests}/{original_tests} configurations ({time_saved_pct:.0f}% fewer tests)")

        diagnostics = {
            'component_range': list(component_range),
            'kurtosis_scores': [round_numerical_values(x) for x in kurtosis_scores],
            'optimal_components': optimal_components,
            'optimal_kurtosis': round_numerical_values(optimal_kurtosis),
            'convergence_info': convergence_info,
            'method': 'kurtosis_maximization_optimized',
            'optimization_savings': {
                'tests_run': actual_tests,
                'tests_originally_planned': original_tests,
                'time_saved_pct': round_numerical_values(time_saved_pct)
            }
        }

        return optimal_components, diagnostics

    def optimize_rp_components(self, X_train: np.ndarray, target_distortion: float = 0.1) -> Tuple[int, Dict[str, Any]]:
        """
        Find optimal number of RP components using practical heuristics and bounds with multiple seeds
        """
        print(f"      🔍 Optimizing RP components using practical heuristics with multiple seeds...")

        n_samples, n_features = X_train.shape

        # Use a more practical approach for RP component selection
        # Start with a reasonable range based on dataset characteristics
        min_comp = max(2, int(np.sqrt(n_features)))  # Square root heuristic
        max_comp = min(n_features, int(0.8 * n_features))  # Cap at 80% of original features

        # Also calculate JL bound for reference (but don't use directly)
        eps = target_distortion
        if eps ** 2 / 2 - eps ** 3 / 3 > 0:
            jl_bound = int(4 * np.log(n_samples) / (eps ** 2 / 2 - eps ** 3 / 3))
            jl_bound = min(jl_bound, n_features)  # Cap at original features
        else:
            jl_bound = n_features // 2  # Fallback

        # Use the minimum of practical bound and JL bound
        practical_max = min(max_comp, jl_bound) if jl_bound > 0 else max_comp

        # Test different numbers of components
        component_range = range(min_comp, practical_max + 1, max(1, (practical_max - min_comp) // 10))

        # Test multiple seeds for robustness
        test_seeds = [self.random_state, self.random_state + 42, self.random_state + 99]
        print(f"         Testing components range: {min_comp} to {practical_max} (JL theoretical: {jl_bound})")
        print(f"         Testing {len(test_seeds)} seeds per component count for robustness")

        reconstruction_errors_by_component = []
        seed_stability_by_component = []

        for n_comp in component_range:
            errors_for_this_component = []

            for seed in test_seeds:
                try:
                    rp = GaussianRandomProjection(n_components=int(n_comp), random_state=seed)
                    X_reduced = rp.fit_transform(X_train)

                    # Estimate reconstruction error using pseudo-inverse
                    # This is an approximation since RP is not directly invertible
                    W = rp.components_
                    X_reconstructed = X_reduced @ W
                    recon_error = mean_squared_error(X_train, X_reconstructed)
                    errors_for_this_component.append(recon_error)

                except Exception as e:
                    print(f"         ⚠️ RP failed for {n_comp} components, seed {seed}: {e}")
                    errors_for_this_component.append(float('inf'))

            # Calculate statistics for this component count
            finite_errors = [e for e in errors_for_this_component if np.isfinite(e)]
            if finite_errors:
                mean_error = np.mean(finite_errors)
                std_error = np.std(finite_errors)
                cv_percent = (std_error / mean_error * 100) if mean_error > 0 else 100
            else:
                mean_error = float('inf')
                std_error = 0
                cv_percent = 100

            reconstruction_errors_by_component.append(mean_error)
            seed_stability_by_component.append({
                'n_components': n_comp,
                'mean_error': mean_error,
                'std_error': std_error,
                'cv_percent': cv_percent,
                'all_errors': errors_for_this_component
            })

        # Find optimal number of components (lowest mean reconstruction error)
        if reconstruction_errors_by_component and any(np.isfinite(reconstruction_errors_by_component)):
            # Filter out infinite values
            finite_indices = [i for i, err in enumerate(reconstruction_errors_by_component) if np.isfinite(err)]
            if finite_indices:
                best_idx = min(finite_indices, key=lambda i: reconstruction_errors_by_component[i])
                optimal_components = list(component_range)[best_idx]
                optimal_error = reconstruction_errors_by_component[best_idx]
                optimal_stability = seed_stability_by_component[best_idx]
            else:
                optimal_components = min_comp
                optimal_error = float('inf')
                optimal_stability = None
        else:
            optimal_components = min(practical_max // 2, n_features // 4)  # Conservative fallback
            optimal_error = float('inf')
            optimal_stability = None

        print(
            f"         ✅ Optimal RP components: {optimal_components} (JL bound: {jl_bound})")
        if optimal_stability:
            print(
                f"            Mean error: {optimal_stability['mean_error']:.4f} ± {optimal_stability['std_error']:.4f}")
            print(
                f"            CV: {optimal_stability['cv_percent']:.1f}% ({'stable' if optimal_stability['cv_percent'] < 20 else 'variable'})")

        diagnostics = {
            'component_range': list(component_range),
            'reconstruction_errors_mean': [round_numerical_values(x) for x in reconstruction_errors_by_component],
            'seed_stability_analysis': seed_stability_by_component,
            'optimal_components': optimal_components,
            'optimal_error_mean': round_numerical_values(optimal_error) if np.isfinite(optimal_error) else None,
            'optimal_stability': optimal_stability,
            'jl_theoretical_bound': jl_bound,
            'practical_max': practical_max,
            'target_distortion': target_distortion,
            'seeds_tested': test_seeds,
            'method': 'practical_heuristics_with_jl_reference_multi_seed'
        }

        return optimal_components, diagnostics

    def apply_dimensionality_reduction(self, X_train: np.ndarray, X_test: np.ndarray,
                                       method: str, n_components: int, dataset_name: str) -> Tuple[
        np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply dimensionality reduction with comprehensive diagnostics and plotting
        """
        plt.style.use('default')  # Ensure consistent plotting style
        
        if method.upper() == 'PCA':
            # First, optimize components based on explained variance
            optimal_components, optimization_diagnostics = self.optimize_pca_components(X_train, target_variance=0.95)

            # Use the specified n_components for this experiment, but include optimization info
            transformer = PCA(n_components=n_components, random_state=self.random_state)
            X_train_reduced = transformer.fit_transform(X_train)
            X_test_reduced = transformer.transform(X_test)

            # Create comprehensive PCA plots with optimization results
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'PCA Analysis with Hyperparameter Optimization - {dataset_name}\n'
                         f'Using {n_components} components (Optimal for 95% variance: {optimal_components})',
                         fontsize=14, fontweight='bold')

            # Scree plot
            PC_values = np.arange(transformer.n_components_) + 1
            axes[0, 0].plot(PC_values, transformer.explained_variance_ratio_, 'bo-', linewidth=2)
            axes[0, 0].set_title(f'Scree Plot ({n_components} components)')
            axes[0, 0].set_xlabel('Principal Component')
            axes[0, 0].set_ylabel('Explained Variance Ratio')
            axes[0, 0].grid(True, alpha=0.3)

            # Cumulative variance plot
            cumvar = np.cumsum(transformer.explained_variance_ratio_)
            axes[0, 1].plot(PC_values, cumvar, 'ro-', linewidth=2)
            axes[0, 1].axhline(y=0.95, color='g', linestyle='--', linewidth=2, label='95% Variance Target')
            axes[0, 1].axvline(x=optimal_components, color='purple', linestyle=':', linewidth=2,
                               label=f'Optimal: {optimal_components} components')
            axes[0, 1].set_title(f'Cumulative Explained Variance')
            axes[0, 1].set_xlabel('Principal Component')
            axes[0, 1].set_ylabel('Cumulative Variance Explained')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Variance threshold analysis
            thresholds = list(optimization_diagnostics['threshold_analysis'].keys())
            components_needed = [optimization_diagnostics['threshold_analysis'][t]['components'] for t in thresholds]

            axes[1, 0].bar([f'{t * 100:.0f}%' for t in thresholds], components_needed,
                           alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_title('Components Needed for Variance Thresholds')
            axes[1, 0].set_xlabel('Variance Threshold')
            axes[1, 0].set_ylabel('Number of Components')
            axes[1, 0].grid(True, alpha=0.3)

            # Add value labels on bars
            for i, (threshold, n_comp) in enumerate(zip(thresholds, components_needed)):
                axes[1, 0].text(i, n_comp + 0.5, str(n_comp), ha='center', va='bottom')

            # Component contribution comparison
            top_components = min(10, len(transformer.explained_variance_ratio_))
            axes[1, 1].bar(range(1, top_components + 1),
                           transformer.explained_variance_ratio_[:top_components],
                           alpha=0.7, color='blue', edgecolor='black')
            axes[1, 1].set_title(f'Top {top_components} Components Contribution')
            axes[1, 1].set_xlabel('Principal Component')
            axes[1, 1].set_ylabel('Explained Variance Ratio')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"figures/pca/{dataset_name}_PCA_{n_components}_optimized.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Enhanced PCA diagnostics with optimization results
            diagnostics = {
                'explained_variance_ratio': [round_numerical_values(x) for x in transformer.explained_variance_ratio_],
                'eigenvalues': [round_numerical_values(x) for x in transformer.explained_variance_],
                'cumulative_variance': [round_numerical_values(x) for x in
                                        np.cumsum(transformer.explained_variance_ratio_)],
                'total_variance_explained': round_numerical_values(np.sum(transformer.explained_variance_ratio_)),
                'optimization_results': optimization_diagnostics,
                'components_used': n_components,
                'optimal_components_95pct': optimal_components,
                'variance_achieved': round_numerical_values(cumvar[-1] if len(cumvar) > 0 else 0)
            }

        elif method.upper() == 'ICA':
            print(f"      🚀 Applying optimized ICA transformation (deflation algorithm, {n_components} components)...")
            # Use optimized ICA parameters for final transformation
            transformer = FastICA(
                n_components=n_components,
                random_state=self.random_state,
                algorithm='deflation',  # Keep deflation as required
                max_iter=300,  # Reduced from 1000 but higher than optimization phase
                tol=1e-4,  # Slightly tighter for final model
                whiten='unit-variance',  # Improve convergence
                fun='logcosh'  # Faster contrast function
            )
            X_train_reduced = transformer.fit_transform(X_train)
            X_test_reduced = transformer.transform(X_test)

            # ICA diagnostics - kurtosis
            component_kurtosis = []
            for i in range(n_components):
                if X_train_reduced is not None and i < X_train_reduced.shape[1]:
                    k = kurtosis(X_train_reduced[:, i])
                    if k is not None and not np.isnan(k):
                        component_kurtosis.append(round_numerical_values(abs(k)))
                    else:
                        component_kurtosis.append(0.0)
                else:
                    component_kurtosis.append(0.0)

            # Create comprehensive ICA plots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Kurtosis plot
            axes[0].plot(list(range(n_components)), component_kurtosis, 'go-')
            axes[0].set_title(f'ICA Component Kurtosis - {dataset_name}\n({n_components} components)')
            axes[0].set_xlabel('ICA Component')
            axes[0].set_ylabel('Absolute Kurtosis')
            axes[0].grid(True, alpha=0.3)

            # Kurtosis distribution
            axes[1].hist(component_kurtosis, bins=min(10, n_components), alpha=0.7, color='green', edgecolor='black')
            kurtosis_array = np.array([k for k in component_kurtosis if k is not None and not np.isnan(k)])
            mean_kurtosis = round_numerical_values(np.mean(kurtosis_array)) if len(kurtosis_array) > 0 else 0.0
            axes[1].axvline(mean_kurtosis, color='red', linestyle='--',
                            label=f'Mean: {mean_kurtosis:.2f}')
            axes[1].set_title(f'Kurtosis Distribution - {dataset_name}')
            axes[1].set_xlabel('Absolute Kurtosis')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"figures/ica/{dataset_name}_ICA_{n_components}.png", dpi=300, bbox_inches='tight')
            plt.close()

            diagnostics = {
                'component_kurtosis': component_kurtosis,
                'avg_kurtosis': mean_kurtosis,
                'max_kurtosis': round_numerical_values(np.max(kurtosis_array)) if len(kurtosis_array) > 0 else 0.0
            }

        elif method.upper() == 'RP':
            # Test multiple random seeds for robustness (RP is stochastic)
            test_seeds = [self.random_state, self.random_state + 42, self.random_state + 99]
            seed_results = []

            print(f"      🎲 Testing RP with {len(test_seeds)} different random seeds for robustness...")

            for seed in test_seeds:
                transformer = GaussianRandomProjection(n_components=int(n_components), random_state=seed)
                X_train_reduced_seed = transformer.fit_transform(X_train)
                X_test_reduced_seed = transformer.transform(X_test)

                # RP diagnostics - reconstruction error for this seed
                W = transformer.components_
                X_reconstructed = X_train_reduced_seed @ W
                reconstruction_error = round_numerical_values(mean_squared_error(X_train, X_reconstructed))

                seed_results.append({
                    'seed': seed,
                    'reconstruction_error': reconstruction_error,
                    'component_variances': [round_numerical_values(x) for x in np.var(X_train_reduced_seed, axis=0)]
                })

            # Use the first seed's transformation for downstream analysis
            transformer = GaussianRandomProjection(n_components=int(n_components), random_state=test_seeds[0])
            X_train_reduced = transformer.fit_transform(X_train)
            X_test_reduced = transformer.transform(X_test)

            # Calculate statistics across seeds
            reconstruction_errors = [r['reconstruction_error'] for r in seed_results]
            mean_recon_error = round_numerical_values(np.mean(reconstruction_errors))
            std_recon_error = round_numerical_values(np.std(reconstruction_errors))

            print(f"         📊 RP Reconstruction Error across seeds: {mean_recon_error:.2f} ± {std_recon_error:.2f}")

            # Component variance analysis (using first seed)
            component_variances = seed_results[0]['component_variances']

            # Create comprehensive RP plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f'Random Projection Analysis - {dataset_name}\n({n_components} components, {len(test_seeds)} seeds tested)',
                fontsize=14, fontweight='bold')

            # Component variance plot (first seed)
            axes[0, 0].plot(range(n_components), component_variances, 'mo-')
            axes[0, 0].set_title(f'RP Component Variances\n(Seed: {test_seeds[0]})')
            axes[0, 0].set_xlabel('RP Component')
            axes[0, 0].set_ylabel('Variance')
            axes[0, 0].grid(True, alpha=0.3)

            # Reconstruction error across seeds
            axes[0, 1].bar(range(len(test_seeds)), reconstruction_errors, alpha=0.7, color='purple', edgecolor='black')
            axes[0, 1].axhline(y=mean_recon_error, color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {mean_recon_error:.2f}')
            axes[0, 1].set_xticks(range(len(test_seeds)))
            axes[0, 1].set_xticklabels([f'Seed {s}' for s in test_seeds], rotation=45)
            axes[0, 1].set_title(f'Reconstruction Error by Seed\n(σ = {std_recon_error:.2f})')
            axes[0, 1].set_ylabel('Mean Squared Error')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Reconstruction error distribution (using first seed for detailed analysis)
            W = transformer.components_
            X_reconstructed = X_train_reduced @ W
            error_per_feature = [round_numerical_values(x) for x in np.mean((X_train - X_reconstructed) ** 2, axis=0)]
            axes[1, 0].hist(error_per_feature, bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 0].axvline(reconstruction_errors[0], color='red', linestyle='--',
                               label=f'Overall MSE: {reconstruction_errors[0]:.2f}')
            axes[1, 0].set_title(f'Reconstruction Error Distribution\n(Seed: {test_seeds[0]})')
            axes[1, 0].set_xlabel('Mean Squared Error per Feature')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Stability analysis
            stability_metrics = ['Mean Error', 'Std Error', 'CV (%)', 'Seeds Tested']
            cv_percent = (std_recon_error / mean_recon_error * 100) if mean_recon_error > 0 else 0
            stability_values = [mean_recon_error, std_recon_error, cv_percent, len(test_seeds)]

            bars = axes[1, 1].bar(stability_metrics, stability_values, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_title('RP Stability Analysis')
            axes[1, 1].set_ylabel('Value')

            # Add value labels on bars
            for bar, value in zip(bars, stability_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{value:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(f"figures/rp/{dataset_name}_RP_{n_components}.png", dpi=300, bbox_inches='tight')
            plt.close()

            diagnostics = {
                'reconstruction_error_mean': mean_recon_error,
                'reconstruction_error_std': std_recon_error,
                'reconstruction_error_cv_percent': round_numerical_values(cv_percent),
                'seeds_tested': test_seeds,
                'seed_results': seed_results,
                'projection_matrix_shape': W.shape,
                'component_variances': component_variances,
                'avg_component_variance': round_numerical_values(np.mean(
                    np.array([v for v in component_variances if v is not None]))) if component_variances else 0.0,
                'stability_assessment': 'stable' if cv_percent < 10 else 'variable'
            }

        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        return X_train_reduced, X_test_reduced, diagnostics

    def run_nn_analysis(self, X_train_reduced: np.ndarray, X_test_reduced: np.ndarray,
                        y_train: pd.Series, y_test: pd.Series, dataset_name: str,
                        dr_method: str, n_components: int) -> Dict[str, Any]:
        """
        Run neural network analysis with hyperparameter tuning (no k-fold CV)
        """
        print(f"   🧠 Running NN analysis with hyperparameter tuning on {dr_method} reduced data...")

        start_time = time.time()

        try:
            # Handle categorical labels for cancer data
            if dataset_name == "CSDataSet":
                # Map cancer stages to binary: early (0) vs advanced (1)
                def map_cancer_stage(x):
                    if x in ['Stage 0', 'Stage I', 'Stage II']:
                        return 0
                    else:
                        return 1

                y_train_processed = y_train.apply(map_cancer_stage).values
                y_test_processed = y_test.apply(map_cancer_stage).values
            else:
                y_train_processed = y_train.values
                y_test_processed = y_test.values

            # Split training data into train/validation for proper hyperparameter selection
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_reduced, y_train_processed, test_size=0.2, random_state=self.random_state
            )

            # Define hyperparameter grid for tuning
            param_grid = {
                'hidden_layer_sizes': [(64, 32)],
                'learning_rate_init': [0.01, 0.1],
                'activation': ['relu']
            }

            print(f"      🔍 Testing {len(param_grid['learning_rate_init'])} learning rate configurations...")

            # Manual hyperparameter search using validation set
            best_score = -1
            best_params = {
                'hidden_layer_sizes': param_grid['hidden_layer_sizes'][0],
                'learning_rate_init': param_grid['learning_rate_init'][0],
                'activation': param_grid['activation'][0]
            }
            best_model = None

            for lr in param_grid['learning_rate_init']:
                print(f"         Testing learning rate: {lr}")

                # Train model with current hyperparameters
                model = MLPClassifier(
                    hidden_layer_sizes=param_grid['hidden_layer_sizes'][0],
                    learning_rate_init=lr,
                    activation=param_grid['activation'][0],
                    max_iter=100,
                    random_state=self.random_state,
                    verbose=False,
                    early_stopping=True,
                    validation_fraction=0.1
                )

                model.fit(X_train_split, y_train_split)

                # Evaluate on validation set (not test set!)
                val_score = model.score(X_val_split, y_val_split)

                if val_score > best_score:
                    best_score = val_score
                    best_params = {
                        'hidden_layer_sizes': param_grid['hidden_layer_sizes'][0],
                        'learning_rate_init': lr,
                        'activation': param_grid['activation'][0]
                    }
                    best_model = model

            # If no model was found, create a default one
            if best_model is None:
                print(f"      ⚠️ No model found during hyperparameter search, using default parameters")
                best_model = MLPClassifier(
                    hidden_layer_sizes=best_params['hidden_layer_sizes'],
                    learning_rate_init=best_params['learning_rate_init'],
                    activation=best_params['activation'],
                    max_iter=100,
                    random_state=self.random_state,
                    verbose=False
                )
                best_model.fit(X_train_reduced, y_train_processed)
                best_score = best_model.score(X_val_split, y_val_split)

            # Retrain the best model on the full training set for final evaluation
            print(f"      🔄 Retraining best model on full training set...")
            final_model = MLPClassifier(
                **best_params,
                max_iter=100,
                random_state=self.random_state,
                verbose=False
            )
            final_model.fit(X_train_reduced, y_train_processed)

            print(f"      ✅ Best hyperparameters found:")
            for param, value in best_params.items():
                print(f"         {param}: {value}")
            print(f"      📊 Best Validation Score: {best_score:.4f}")

            train_sizes = [0.2, 0.6, 1.0]
            train_scores = []
            val_scores = []
            train_losses = []

            for train_size in train_sizes:
                # Create subset
                subset_size = int(train_size * len(X_train_reduced))
                indices = np.random.choice(len(X_train_reduced), subset_size, replace=False)
                X_subset = X_train_reduced[indices]
                y_subset = y_train_processed[indices]

                # Train model with best hyperparameters
                model_temp = MLPClassifier(**best_params, max_iter=100, random_state=self.random_state,
                                           verbose=False)
                model_temp.fit(X_subset, y_subset)

                # Evaluate
                train_scores.append(round_numerical_values(model_temp.score(X_subset, y_subset)))
                val_scores.append(round_numerical_values(model_temp.score(X_test_reduced, y_test_processed)))
                if hasattr(model_temp, 'loss_'):
                    train_losses.append(round_numerical_values(model_temp.loss_))
                else:
                    train_losses.append(0)

            # Evaluate final model (trained on full training set)
            train_pred = final_model.predict(X_train_reduced)
            test_pred = final_model.predict(X_test_reduced)

            train_accuracy = round_numerical_values(accuracy_score(y_train_processed, train_pred))
            test_accuracy = round_numerical_values(accuracy_score(y_test_processed, test_pred))

            nn_time = round_numerical_values(time.time() - start_time)

            # Create comprehensive NN visualization with hyperparameter results
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Neural Network Analysis - {dr_method} ({n_components} components)\n{dataset_name}',
                         fontsize=14, fontweight='bold')

            # Learning curves
            axes[0, 0].plot(train_sizes, train_scores, 'b-o', label='Training Accuracy', linewidth=2)
            axes[0, 0].plot(train_sizes, val_scores, 'r-o', label='Validation Accuracy', linewidth=2)
            axes[0, 0].set_title('Learning Curves (Best Model)')
            axes[0, 0].set_xlabel('Training Set Size Fraction')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Hyperparameter comparison (learning rates tested)
            lr_values = param_grid['learning_rate_init']
            lr_labels = [f'LR: {lr}' for lr in lr_values]
            test_scores_by_lr = []
            for lr in lr_values:
                temp_model = MLPClassifier(
                    hidden_layer_sizes=param_grid['hidden_layer_sizes'][0],
                    learning_rate_init=lr,
                    activation=param_grid['activation'][0],
                    max_iter=100,
                    random_state=self.random_state,
                    verbose=False
                )
                temp_model.fit(X_train_reduced, y_train_processed)
                test_scores_by_lr.append(temp_model.score(X_test_reduced, y_test_processed))

            axes[0, 1].bar(range(len(lr_values)), test_scores_by_lr, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_xticks(range(len(lr_values)))
            axes[0, 1].set_xticklabels(lr_labels)
            axes[0, 1].set_title('Hyperparameter Comparison (No CV)')
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Test Accuracy')
            axes[0, 1].grid(True, alpha=0.3)

            # Training loss over iterations (if available)
            if hasattr(final_model, 'loss_curve_') and final_model.loss_curve_ is not None:
                axes[1, 0].plot(final_model.loss_curve_, 'g-', linewidth=2)
                axes[1, 0].set_title('Training Loss Curve (Final Model)')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                # Plot training losses from learning curve
                axes[1, 0].plot(train_sizes, train_losses, 'm-o', linewidth=2)
                axes[1, 0].set_title('Training Loss vs Data Size')
                axes[1, 0].set_xlabel('Training Set Size Fraction')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)

            # Performance comparison
            metrics = ['Train Acc', 'Test Acc', 'Best LR Score', 'Time (s)']
            values = [train_accuracy, test_accuracy, best_score, nn_time]
            colors = ['blue', 'red', 'green', 'orange']

            bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Performance Summary (No CV)')
            axes[1, 1].set_ylabel('Value')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{value:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(f"figures/nndr/{dataset_name}_{dr_method}_{n_components}_NN_tuned.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Store NN results with flattened hyperparameter information
            global_results.add_nn_results(
                dataset_name=dataset_name,
                dr_method=dr_method,
                n_components=n_components,
                nn_time=nn_time,
                accuracy=test_accuracy,
                train_accuracy=train_accuracy,
                best_test_score=best_score,
                # Flatten best_params into individual columns
                best_hidden_layer_sizes=str(best_params.get('hidden_layer_sizes', 'N/A')),
                best_learning_rate_init=best_params.get('learning_rate_init', 'N/A'),
                best_activation=best_params.get('activation', 'N/A')
            )

            return {
                'success': True,
                'time': nn_time,
                'accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'best_test_score': round_numerical_values(best_score),
                'best_params': best_params,
                'iterations': 100,
                'message': f"NN hyperparameter tuning completed in {nn_time:.2f}s (Test Accuracy: {test_accuracy:.2f}, Train Accuracy: {train_accuracy:.2f}, 100 iterations)"
            }

        except Exception as e:
            print(f"   ❌ NN analysis failed: {e}")
            return {
                'success': False,
                'time': 0.0,
                'message': f"NN analysis failed: {str(e)}"
            }

    def run_baseline_experiments(self, X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: pd.Series, y_test: pd.Series, dataset_name: str) -> Dict[str, Any]:
        """
        Step 1: Run baseline K-Means and EM clustering on original data
        """
        print(f"\n{'=' * 80}")
        print(f"STEP 1: BASELINE CLUSTERING - {dataset_name}")
        print(f"{'=' * 80}")

        results = {}

        # Baseline K-Means
        print("🔵 Running baseline K-Means...")
        kmeans_results = self.run_kmeans_experiment(X_train, X_test, y_train, dataset_name, "baseline")
        results['kmeans'] = kmeans_results
        print(f"   ✅ {kmeans_results['summary']}")

        # Baseline EM
        print("🔴 Running baseline EM...")
        em_results = self.run_em_experiment(X_train, X_test, y_train, dataset_name, "baseline")
        results['em'] = em_results
        print(f"   ✅ {em_results['summary']}")

        # --- Cluster Visualization ---
        # K-Means: Use optimal_k and cluster centers
        from sklearn.cluster import KMeans
        optimal_k = kmeans_results['optimal_k']
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init='auto')
        kmeans_labels = kmeans.fit_predict(X_train)
        plot_clusters_2d(
            X_train, kmeans_labels, centers=kmeans.cluster_centers_, gmm=None,
            title=f"{dataset_name} K-Means Baseline Clusters (k={optimal_k})",
            filename=f"figures/consolidated/{dataset_name}_kmeans_baseline_2d.png",
            y_true=y_train.values
        )
        plot_clusters_3d(
            X_train, kmeans_labels, centers=kmeans.cluster_centers_, gmm=None,
            title=f"{dataset_name} K-Means Baseline Clusters (k={optimal_k})",
            filename=f"figures/consolidated/{dataset_name}_kmeans_baseline_3d.png",
            y_true=y_train.values
        )
        # EM/GMM: Use optimal_k and fitted GMM
        from sklearn.mixture import GaussianMixture
        optimal_em_k = em_results['optimal_k']
        gmm = GaussianMixture(n_components=optimal_em_k, random_state=self.random_state, n_init=2, max_iter=100, tol=1e-4, covariance_type='diag')
        gmm_labels = gmm.fit_predict(X_train)
        plot_clusters_2d(
            X_train, gmm_labels, centers=None, gmm=gmm,
            title=f"{dataset_name} EM (GMM) Baseline Clusters (k={optimal_em_k})",
            filename=f"figures/consolidated/{dataset_name}_em_baseline_2d.png",
            y_true=y_train.values
        )
        plot_clusters_3d(
            X_train, gmm_labels, centers=None, gmm=gmm,
            title=f"{dataset_name} EM (GMM) Baseline Clusters (k={optimal_em_k})",
            filename=f"figures/consolidated/{dataset_name}_em_baseline_3d.png",
            y_true=y_train.values
        )
        return results

    def run_dr_experiments(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: pd.Series, y_test: pd.Series, dataset_name: str) -> Dict[str, Any]:
        """
        Steps 2-4: Run dimensionality reduction + clustering + NN analysis with automatic component optimization
        """
        print(f"\n{'=' * 80}")
        print(f"STEPS 2-4: DIMENSIONALITY REDUCTION + CLUSTERING + NN - {dataset_name}")
        print(f"{'=' * 80}")

        dr_methods = ['PCA', 'ICA', 'RP']
        all_results = {}

        for dr_method in dr_methods:
            print(f"\n{'-' * 60}")
            print(f"DIMENSIONALITY REDUCTION: {dr_method} (Automatic Optimization)")
            print(f"{'-' * 60}")

            try:
                # Find optimal number of components for each method
                if dr_method == 'PCA':
                    optimal_components, opt_diagnostics = self.optimize_pca_components(X_train, target_variance=0.95)
                elif dr_method == 'ICA':
                    optimal_components, opt_diagnostics = self.optimize_ica_components(X_train, max_components=min(
                        X_train.shape[1], 20))
                elif dr_method == 'RP':
                    optimal_components, opt_diagnostics = self.optimize_rp_components(X_train, target_distortion=0.1)
                else:
                    continue

                print(f"\n   📐 {dr_method} with optimal {optimal_components} components")

                # Apply dimensionality reduction with optimal components
                X_train_reduced, X_test_reduced, diagnostics = self.apply_dimensionality_reduction(
                    X_train, X_test, dr_method, optimal_components, dataset_name
                )

                print(f"      Shape: {X_train.shape} → {X_train_reduced.shape}")

                # Add optimization diagnostics to the DR diagnostics
                diagnostics['optimization'] = opt_diagnostics

                # Store DR diagnostics
                global_results.add_dr_results(
                    dataset_name=dataset_name,
                    dr_method=dr_method.lower(),
                    n_components=optimal_components,
                    original_features=X_train.shape[1],
                    reduced_features=X_train_reduced.shape[1],
                    explained_variance=diagnostics.get('total_variance_explained'),
                    # Handle both old and new RP reconstruction error formats
                    reconstruction_error=diagnostics.get('reconstruction_error_mean') or diagnostics.get(
                        'reconstruction_error'),
                    reconstruction_error_std=diagnostics.get('reconstruction_error_std'),
                    reconstruction_error_cv=diagnostics.get('reconstruction_error_cv_percent'),
                    stability_assessment=diagnostics.get('stability_assessment'),
                    optimization_method=opt_diagnostics.get('method'),
                    optimization_score=opt_diagnostics.get('optimal_kurtosis') or opt_diagnostics.get(
                        'achieved_variance') or opt_diagnostics.get('optimal_error_mean') or opt_diagnostics.get(
                        'optimal_error')
                )

                # Run K-Means on reduced data
                print(f"      🔵 K-Means on {dr_method} reduced data...")
                kmeans_results = self.run_kmeans_experiment(
                    X_train_reduced, X_test_reduced, y_train, dataset_name, dr_method.lower(), optimal_components
                )

                # Run EM on reduced data
                print(f"      🔴 EM on {dr_method} reduced data...")
                em_results = self.run_em_experiment(
                    X_train_reduced, X_test_reduced, y_train, dataset_name, dr_method.lower(), optimal_components
                )

                # Run NN analysis (only for Cancer Data)
                if dataset_name == "CSDataSet":
                    nn_results = self.run_nn_analysis(
                        X_train_reduced, X_test_reduced, y_train, y_test,
                        dataset_name, dr_method.lower(), optimal_components
                    )
                else:
                    nn_results = {
                        'success': False,
                        'time': 0.0,
                        'message': f"NN analysis skipped for {dataset_name} (only running on Cancer Data)"
                    }

                all_results[dr_method] = {
                    'optimal_components': optimal_components,
                    'optimization_diagnostics': opt_diagnostics,
                    'diagnostics': diagnostics,
                    'kmeans': kmeans_results,
                    'em': em_results,
                    'nn': nn_results
                }

                print(f"      ✅ {kmeans_results['summary']}")
                print(f"      ✅ {em_results['summary']}")
                print(f"      ✅ {nn_results['message']}")

            except Exception as e:
                print(f"      ❌ Error with {dr_method} optimization: {e}")
                continue

        return all_results

    def generate_comprehensive_report(self, dataset_name: str):
        """
        Generate comprehensive report with all metrics
        """
        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE REPORT - {dataset_name}")
        print(f"{'=' * 80}")

        # Generate and print all consolidated tables
        global_results.print_summary_table('comprehensive')
        global_results.print_summary_table('kmeans')
        global_results.print_summary_table('em')
        global_results.print_summary_table('nn')
        global_results.print_summary_table('dr')

        # Save all tables
        global_results.save_all_tables(f'{dataset_name}_consolidated')
        global_results.save_results_json(f'{dataset_name}_results.json')

    def run_single_dataset_experiment(self, dataset_name: str, file_path: str):
        """
        Run complete experiment pipeline for a single dataset
        """
        print(f"\n🚀 Starting comprehensive experiment for {dataset_name}")
        start_time = time.time()

        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data(file_path, dataset_name)

        # Step 1: Baseline clustering
        baseline_results = self.run_baseline_experiments(X_train, X_test, y_train, y_test, dataset_name)

        # Steps 2-4: DR + Clustering + NN
        dr_results = self.run_dr_experiments(X_train, X_test, y_train, y_test, dataset_name)

        # Generate comprehensive report
        self.generate_comprehensive_report(dataset_name)

        total_time = time.time() - start_time
        print(f"\n✅ {dataset_name} experiment completed in {total_time:.2f} seconds")

        return {
            'baseline': baseline_results,
            'dimensionality_reduction': dr_results,
            'total_time': total_time
        }


def main():
    """
    Main execution function - runs all experiments
    """
    print("🎓 CS 7641 - Unsupervised Learning Assignment")
    print("Consolidated Experiment Runner - Eliminates Code Duplication")
    print("=" * 80)

    # Dataset configuration
    datasets = {
        "BankrupcyDS": "data/company_bankruptcy_data.csv",
        "CSDataSet": "data/global_cancer_patients_2015_2024.csv"
    }

    # Initialize experiment runner
    experiments = ConsolidatedExperiments()
    all_results = {}

    total_start_time = time.time()

    # Run experiments for each dataset
    for dataset_name, file_path in datasets.items():
        if not os.path.exists(file_path):
            print(f"❌ Data file not found: {file_path}")
            continue

        results = experiments.run_single_dataset_experiment(dataset_name, file_path)
        all_results[dataset_name] = results

    # Generate final combined report
    print(f"\n{'=' * 120}")
    print("FINAL COMBINED ANALYSIS - ALL DATASETS")
    print(f"{'=' * 120}")

    # Print final consolidated tables
    global_results.print_summary_table('comprehensive')

    # Save final results
    global_results.save_all_tables('final_consolidated')
    global_results.save_results_json('final_all_results.json')

    total_time = time.time() - total_start_time

    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"📁 Results saved in: figures/consolidated_results/")
    print(f"📁 NN plots saved in: figures/nndr/")
    print(f"📋 Check final_consolidated_comprehensive_results.csv for 12-combo table")


if __name__ == "__main__":
    main()
