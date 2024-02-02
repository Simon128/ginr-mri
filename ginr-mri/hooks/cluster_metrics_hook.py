from typing import TYPE_CHECKING
from enum import StrEnum
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import torch.distributed as dist
import umap

from .hook import Hook
from ..models import ModelOutput
from ..metrics import cluster_metrics as cm

if TYPE_CHECKING:
    from ..engine import Engine

class ClusterMetricsMode(StrEnum):
    before_latent_transformation = "before_latent_transformation"
    after_latent_transformation = "after_latent_transformation"
    after_weight_modulation = "after_weight_modulation"

class ClusterMetricsHook(Hook):
    def __init__(
            self, 
            priority: int = 0, 
            frequency: int = 100,
            mode: ClusterMetricsMode = ClusterMetricsMode.before_latent_transformation,
            stages: list[str] = ["train", "val"],
            embedding_positions: list[int] = [],
            min_clusters: int = 2,
            max_clusters: int = 10
        ) -> None:
        super().__init__(priority)
        for s in stages:
            assert s in ["train", "val"]
        self.mode = mode
        self.frequency = frequency
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.stages = stages
        self.device = dist.get_rank() if dist.is_initialized else "cuda"
        self.embedding_positions = embedding_positions

    def pre_fit(
        self, 
        engine: 'Engine',
        model: nn.Module,
        train_dataset: Dataset, 
        val_dataset: Dataset,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        **kwargs
    ) -> dict | None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model

    def get_plot(self, title: str, y_label: str, x_label: str, xticks, data, yticks = np.arange(0, 1.1, 0.1)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xticks, data)
        ax.set_ylabel(y_label)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        return fig

    def process_batch(self, batch):
        if batch[2] == "channelwise":
            batch_size = batch[0].shape[0]
            return batch[0].flatten(start_dim=0, end_dim=1), np.array([0, 1, 2, 3] * batch_size)
        else:
            batch_size = batch[0].shape[0]
            return batch[0], np.array([0] * batch_size)

    def get_latent_embeddings(self, mode: str, emb_pos: int, max_elements: int = 1000):
        if mode == "valid":
            loader = self.val_dataloader
        elif mode == "train":
            loader = self.train_dataloader

        labels = []
        embeddings = []
        total = min(max_elements, len(loader.dataset))
        with torch.inference_mode():
            pbar = tqdm(total=total, desc=f"Inferring latent embeddings from pos {emb_pos}")
            for batch in loader:
                xs, label = self.process_batch(batch)
                labels.append(label)
                xs = xs.to(self.device)
                if self.mode == ClusterMetricsMode.before_latent_transformation:
                    emb = self.model.backbone(xs)
                elif self.mode == ClusterMetricsMode.after_latent_transformation:
                    emb = self.model.backbone(xs)
                    if self.model.latent_transform is not None:
                        emb = self.model.latent_transform(emb)

                embeddings.append(emb[emb_pos].cpu().numpy())
                n_elements = sum([xs.shape[0] for _ in embeddings])
                pbar.update(min(xs.shape[0], total - (n_elements - xs.shape[0])))
                if n_elements >= total:
                    break
            pbar.close()

        embeddings = np.concatenate(embeddings, axis=0)
        if len(labels) > 0:
            labels = np.concatenate(labels, axis=0)

        return embeddings, labels

    def get_umap(self, embeddings, labels, dims=2):
        umap_embeddings = umap.UMAP(n_components=dims).fit_transform(embeddings)
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111)
        ax.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            cmap=colormaps["tab10"],
            c=labels if len(labels) > 0 else None,
            s=10
        )
        ax.set_aspect('equal', 'datalim')
        ax.set_title(f"UMAP - Embeddings {dims}D")
        return fig

    def get_cluster_eval(self, embeddings, labels):
        # is start and end reasonable?
        cluster_nums = [k for k in range(self.min_clusters, self.max_clusters + 1)]
        sil_scores = []
        purity_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        rand_scores = []
        jaccard_scores = []
        for k in tqdm(cluster_nums, desc=f"Computing latent cluster metrics"):
            cluster_predictions = KMeans(k, n_init="auto").fit_predict(embeddings)
            sil = silhouette_score(embeddings, cluster_predictions)
            tptnfpfn = cm.tptnfpfn(cluster_predictions, labels, k)
            tp, tn, fp, fn = tptnfpfn
            purity = cm.compute_purity(cluster_predictions, labels, k)
            rand = cm.rand_coefficient(cluster_predictions, labels, k, tptnfpfn)
            jaccard = cm.jaccard_coefficient(cluster_predictions, labels, k, tpfpfn=(tp, fp, tn))
            f_1, precision, recall = cm.f_beta_precision_recall(cluster_predictions, labels, k, tpfpfn=(tp, fp, tn))
            sil_scores.append(float(sil))
            purity_scores.append(float(purity))
            f1_scores.append(float(f_1))
            precision_scores.append(float(precision))
            recall_scores.append(float(recall))
            rand_scores.append(float(rand))
            jaccard_scores.append(float(jaccard))

        return dict(
            num_clusters=cluster_nums,
            silhoutte_scores=sil_scores,
            purity=purity_scores,
            precision=precision_scores,
            recall=recall_scores,
            rand_index=rand_scores,
            jaccard_index=jaccard_scores,
            f1=f1_scores,
            plt_silhoutte_scores=self.get_plot("Latent Clustering: Silhouette Scores", "silhoutte score", "#clusters", cluster_nums, sil_scores, np.arange(-1, 1.1, 0.2)),
            plt_purity=self.get_plot("Latent Clustering: Purity Scores", "purity", "#clusters", cluster_nums, purity_scores),
            plt_precision=self.get_plot("Latent Clustering: Precision Scores", "precision", "#clusters", cluster_nums, precision_scores),
            plt_recall=self.get_plot("Latent Clustering: Recall Scores", "recall", "#clusters", cluster_nums, recall_scores),
            plt_rand_index=self.get_plot("Latent Clustering: Rand Scores", "rand score", "#clusters", cluster_nums, rand_scores),
            plt_jaccard_index=self.get_plot("Latent Clustering: Jaccard Scores", "jaccard score", "#clusters", cluster_nums, jaccard_scores),
            plt_f1=self.get_plot("Latent Clustering: F1 Scores", "f1 score", "#clusters", cluster_nums, f1_scores)
        )

    def post_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        if "val" in self.stages and epoch % self.frequency == 0:
            output = {
            }
            for emb_pos in self.embedding_positions:
                embeddings, labels = self.get_latent_embeddings("val", emb_pos)
                metrics = self.get_cluster_eval(embeddings, labels)
                umap_2d = self.get_umap(embeddings, labels, 2)
                umap_3d = self.get_umap(embeddings, labels, 3)
                output[emb_pos] = {
                    "umap_2d": umap_2d,
                    "umap_3d": umap_3d
                }
                for k, val in metrics.items():
                    output[emb_pos][k] = val

            return {
                "cluster_metrics": output
            }

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        if "train" in self.stages and epoch % self.frequency == 0:
            output = {
                "mode": str(self.mode)
            }
            for emb_pos in self.embedding_positions:
                embeddings, labels = self.get_latent_embeddings("train", emb_pos)
                metrics = self.get_cluster_eval(embeddings, labels)
                umap_2d = self.get_umap(embeddings, labels, 2)
                umap_3d = self.get_umap(embeddings, labels, 3)
                output[emb_pos] = {
                    "umap_2d": umap_2d,
                    "umap_3d": umap_3d
                }
                for k, val in metrics.items():
                    output[emb_pos][k] = val

            return {
                "cluster_metrics": output
            }
