#!/usr/bin/env python
"""
foampilot.gnn — Module GNN Surrogate pour simulations OpenFOAM (2D/3D)
======================================================================

Gestion unifiée des cas 2D et 3D avec détection automatique de la dimension.
Support des maillages structurés/non-structurés, échantillonnage adaptatif,
et visualisation adaptée à la dimension.

Usage :
    from foampilot.gnn import Experiment, GNNConfig

    cfg = GNNConfig.for_nozzle()  # ou .for_airfoil(), .for_urban()
    cfg.model.n_layers = 8
    exp = Experiment(cfg)
    exp.extract_graphs()          # OpenFOAM → .pt
    exp.fit()                     # entraînement
    exp.evaluate()                 # métriques physiques
    exp.compare("simulations/sim_0099")
"""

from __future__ import annotations

import json
import random
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour 3D

from foampilot import postprocess


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class SamplerConfig:
    """Zones d'échantillonnage : liste de (dist_max, keep_rate)."""
    preset: Optional[str] = None
    zones: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.05, 1.00), (0.50, 0.33), (1e9, 0.10)
    ])
    
    def __post_init__(self):
        """Validation des zones."""
        for i, (d_max, rate) in enumerate(self.zones):
            if rate <= 0 or rate > 1:
                raise ValueError(f"Le taux d'échantillonnage doit être dans (0,1], reçu {rate}")
            if i > 0 and d_max <= self.zones[i-1][0]:
                raise ValueError("Les distances maximales doivent être croissantes")


@dataclass
class ModelConfig:
    name: str = "MeshGraphNet"
    hidden_dim: int = 128
    n_layers: int = 6
    edge_dim: int = 3  # delta/|delta| + distance
    use_dynamic_edges: bool = False  # True pour mettre à jour edge_attr


@dataclass
class LossConfig:
    data: float = 1.0
    inlet: float = 2.0
    no_slip: float = 5.0
    divergence: float = 0.0  # pénalité de divergence (0 = désactivé)
    continuity: float = 0.0  # pénalité de continuité massique


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-3
    lr_max: float = 5e-3
    weight_decay: float = 1e-5
    patience: int = 20
    noise_std: float = 0.002
    val_split: float = 0.2
    num_workers: int = 2
    gradient_clip: float = 1.0  # clipping pour stabilité
    use_amp: bool = True  # mixed precision


@dataclass
class GNNConfig:
    """Configuration complète d'une expérience GNN (support 2D/3D)."""
    
    name: str = "exp_001"
    case_type: str = "generic"
    fields_out: List[str] = field(default_factory=lambda: ["p", "Ux", "Uy", "Uz"])
    physical_params_keys: List[str] = field(default_factory=list)
    
    sim_dir: Path = Path("simulations")
    graph_dir: Path = Path("graphs")
    model_dir: Path = Path("models")
    
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    seed: int = 42
    dimension: int = 2  # 2 ou 3, détection auto si 0
    
    def __post_init__(self):
        """Validation et détection automatique de la dimension."""
        if self.dimension not in [0, 2, 3]:
            raise ValueError(f"dimension doit être 0 (auto), 2 ou 3, reçu {self.dimension}")
        
        # Validation des champs de sortie
        velocity_fields = [f for f in self.fields_out if f.upper().startswith("U")]
        if velocity_fields:
            expected_dims = {
                "U": self.dimension if self.dimension > 0 else 3,
                "Ux": 1, "Uy": 1, "Uz": 1,
                "UxUy": 2, "UxUyUz": 3
            }
            for f in velocity_fields:
                if f not in expected_dims:
                    warnings.warn(f"Champ de vitesse non standard: {f}")
    
    @property
    def output_dim(self) -> int:
        return len(self.fields_out)
    
    @property
    def u_slice(self) -> slice:
        """Tranche pour les composantes de vitesse."""
        u_indices = [i for i, f in enumerate(self.fields_out) if f.upper().startswith("U")]
        if not u_indices:
            return slice(0, 0)
        return slice(u_indices[0], u_indices[-1] + 1)
    
    @property
    def n_physical_params(self) -> int:
        return len(self.physical_params_keys)
    
    @property
    def effective_dimension(self) -> int:
        """Dimension effective (détectée ou configurée)."""
        return self.dimension
    
    def node_input_dim(self, n_node_types: int = 6) -> int:
        """Dimension d'entrée des nœuds (position + type + distance + params)."""
        pos_dim = self.effective_dimension  # coordonnées spatiales
        return pos_dim + n_node_types + 1 + self.n_physical_params
    
    def experiment_dir(self) -> Path:
        return Path(self.model_dir) / self.name
    
    def to_dict(self) -> dict:
        """Sérialisation pour JSON."""
        d = asdict(self)
        # Conversion des Path en str
        d["sim_dir"] = str(d["sim_dir"])
        d["graph_dir"] = str(d["graph_dir"])
        d["model_dir"] = str(d["model_dir"])
        return d
    
    @classmethod
    def for_nozzle(cls, dimension: int = 2) -> GNNConfig:
        """Configuration pour nozzle axisymétrique."""
        cfg = cls(
            name="nozzle_exp",
            case_type="nozzle",
            dimension=dimension,
            fields_out=["p", "T", "Ux", "Uy"] if dimension == 2 else ["p", "T", "Ux", "Uy", "Uz"],
            physical_params_keys=["R_throat", "R_exit", "p_total", "p_outlet", "AR_exit"],
        )
        cfg.sampler = SamplerConfig(
            preset="nozzle",
            zones=[(5e-3, 1.0), (0.05, 0.50), (1e9, 0.25)]
        )
        cfg.loss = LossConfig(data=1.0, inlet=2.0, no_slip=8.0)
        return cfg
    
    @classmethod
    def for_airfoil(cls, dimension: int = 2) -> GNNConfig:
        """Configuration pour profil d'aile (2D par extrusion)."""
        cfg = cls(
            name="airfoil_exp",
            case_type="airfoil",
            dimension=dimension,
            fields_out=["p", "Ux", "Uy"],
            physical_params_keys=["angle_of_attack", "Re", "Ma"],
        )
        cfg.sampler = SamplerConfig(
            preset="airfoil",
            zones=[(0.05, 1.0), (0.50, 0.33), (1e9, 0.10)]
        )
        cfg.loss = LossConfig(data=1.0, inlet=2.0, no_slip=5.0)
        return cfg
    
    @classmethod
    def for_urban(cls, dimension: int = 3) -> GNNConfig:
        """Configuration pour écoulement urbain (3D)."""
        cfg = cls(
            name="urban_exp",
            case_type="urban",
            dimension=dimension,
            fields_out=["p", "Ux", "Uy", "Uz"],
            physical_params_keys=["wind_angle", "wind_speed", "turb_intensity"],
        )
        cfg.sampler = SamplerConfig(
            preset="urban",
            zones=[(1.0, 1.0), (10.0, 0.50), (1e9, 0.10)]
        )
        return cfg


# =============================================================================
# 2. TYPES DE NŒUDS
# =============================================================================

NODE_FLUID = 0
NODE_INLET = 1
NODE_WALL = 2
NODE_OUTLET = 3
NODE_SYMMETRY = 4
NODE_WEDGE = 5
N_NODE_TYPES = 6

PATCH_MAP = {
    "inlet": NODE_INLET, "inflow": NODE_INLET,
    "outlet": NODE_OUTLET, "outflow": NODE_OUTLET,
    "wall": NODE_WALL, "nozzle": NODE_WALL, "airfoil": NODE_WALL, "building": NODE_WALL,
    "symmetry": NODE_SYMMETRY, "axis": NODE_SYMMETRY,
    "front": NODE_WEDGE, "back": NODE_WEDGE,
}

SAMPLER_PRESETS = {
    "airfoil": [(0.05, 1.0), (0.50, 0.33), (1e9, 0.10)],
    "nozzle": [(5e-3, 1.0), (0.05, 0.50), (1e9, 0.25)],
    "urban": [(1.0, 1.0), (10.0, 0.50), (1e9, 0.10)],
    "full": [(1e9, 1.0)],
}


# =============================================================================
# 3. GRAPH EXTRACTION (2D/3D)
# =============================================================================

class SmartSampler:
    """Échantillonnage adaptatif par zones de distance à la paroi."""
    
    def __init__(self, zones: List[Tuple[float, float]]):
        self.zones = sorted(zones, key=lambda z: z[0])
        self._validate_zones()
    
    def _validate_zones(self):
        """Validation des zones."""
        for i, (d_max, rate) in enumerate(self.zones):
            if rate <= 0 or rate > 1:
                raise ValueError(f"Taux d'échantillonnage invalide: {rate}")
            if i > 0 and d_max <= self.zones[i-1][0]:
                raise ValueError(f"Zones non croissantes: {self.zones[i-1][0]} >= {d_max}")
    
    @classmethod
    def from_preset(cls, preset: str, override: Optional[List[Tuple]] = None) -> SmartSampler:
        """Crée un sampler à partir d'un preset."""
        if override:
            return cls(override)
        if preset not in SAMPLER_PRESETS:
            raise ValueError(f"Preset inconnu: '{preset}'. Disponibles: {list(SAMPLER_PRESETS)}")
        return cls(SAMPLER_PRESETS[preset])
    
    def sample(self, dist_to_wall: np.ndarray, node_types: Optional[np.ndarray] = None) -> np.ndarray:
        """Échantillonne les indices à conserver."""
        n = len(dist_to_wall)
        keep = np.zeros(n, dtype=bool)
        
        # Toujours conserver les nœuds de bordure
        if node_types is not None:
            keep[node_types != NODE_FLUID] = True
        
        # Échantillonnage par zones
        for i, (d_max, rate) in enumerate(self.zones):
            d_min = self.zones[i-1][0] if i > 0 else 0.0
            in_zone = (dist_to_wall >= d_min) & (dist_to_wall < d_max)
            zone_idx = np.where(in_zone)[0]
            
            if len(zone_idx) == 0:
                continue
            
            n_keep = max(1, int(len(zone_idx) * rate))
            # Mélange pour éviter le biais de grille
            zone_list = zone_idx.tolist()
            random.shuffle(zone_list)
            keep[zone_list[:n_keep]] = True
        
        return np.where(keep)[0]


def foam_to_graph(
    case_path: Path,
    case_type: str = "generic",
    fields_out: Optional[List[str]] = None,
    physical_params: Optional[Dict[str, float]] = None,
    sampler: Optional[SmartSampler] = None,
    connectivity: str = "mesh",
    k_neighbors: int = 6,
    time_step: str = "latest",
    dimension: int = 0,  # 0 = auto-détection
) -> Data:
    """Convertit un cas OpenFOAM en graphe PyTorch Geometric (2D/3D)."""
    
    if fields_out is None:
        fields_out = ["p", "Ux", "Uy"]
    if physical_params is None:
        physical_params = {}
    
    case_path = Path(case_path)
    if not case_path.exists():
        raise FileNotFoundError(f"Cas introuvable: {case_path}")
    
    # Chargement des données OpenFOAM
    foam_post = postprocess.FoamPostProcessing(case_path=case_path)
    available = foam_post.get_all_time_steps()
    if not available:
        raise ValueError(f"Aucun time step trouvé dans {case_path}")
    
    ts = available[-1] if time_step == "latest" else time_step
    structure = foam_post.load_time_step(ts)
    cell_mesh = structure["cell"]
    boundaries = structure["boundaries"]
    
    n_pts = cell_mesh.n_points
    
    # Détection automatique de la dimension
    points = cell_mesh.points
    if dimension == 0:
        # Détection: si tous les points ont z ≈ 0, c'est 2D
        if points.shape[1] >= 3 and np.allclose(points[:, 2], 0.0, atol=1e-6):
            dimension = 2
        else:
            dimension = points.shape[1]  # 2 ou 3
    
    # Classification des nœuds
    node_types = torch.zeros(n_pts, dtype=torch.long)
    kd_mesh = KDTree(points[:, :dimension])
    
    for patch_name, patch_mesh in boundaries.items():
        if patch_mesh.n_points == 0:
            continue
        
        ntype = NODE_FLUID
        for key, val in PATCH_MAP.items():
            if key in patch_name.lower():
                ntype = val
                break
        
        # Projection des points de la frontière sur le maillage
        patch_points = patch_mesh.points[:, :dimension]
        if len(patch_points) > 0:
            _, ids = kd_mesh.query(patch_points)
            node_types[ids] = ntype
    
    # Calcul de la distance à la paroi
    wall_mask = (node_types.numpy() == NODE_WALL)
    wall_pts = points[wall_mask][:, :dimension]
    
    if len(wall_pts) > 0:
        kd_wall = KDTree(wall_pts)
        dist_wall, _ = kd_wall.query(points[:, :dimension])
    else:
        dist_wall = np.ones(n_pts) * 1e3
        warnings.warn("Aucun nœud de type WALL trouvé")
    
    # Échantillonnage
    if sampler is not None:
        keep = sampler.sample(dist_wall, node_types.numpy())
    else:
        keep = np.arange(n_pts)
    
    n_kept = len(keep)
    
    # Construction des features des nœuds
    pos_full = torch.tensor(points[:, :dimension], dtype=torch.float)
    pos = pos_full[keep]
    
    # One-hot encoding du type de nœud
    type_oh = F.one_hot(
        node_types[keep].clamp(0, N_NODE_TYPES - 1),
        num_classes=N_NODE_TYPES
    ).float()
    
    # Distance à la paroi
    dist_t = torch.tensor(dist_wall[keep], dtype=torch.float).unsqueeze(1)
    
    # Paramètres physiques (broadcastés)
    if physical_params:
        # Trier pour reproductibilité
        param_values = [float(physical_params[k]) for k in sorted(physical_params.keys())]
        param_t = torch.tensor([param_values] * n_kept, dtype=torch.float)
    else:
        param_t = torch.zeros(n_kept, 0)
    
    # Assemblage final des features
    x = torch.cat([pos, type_oh, dist_t, param_t], dim=1)
    
    # Cibles (champs de sortie)
    y_parts = []
    for field in fields_out:
        if field in cell_mesh.point_data:
            arr = cell_mesh.point_data[field][keep]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            y_parts.append(torch.tensor(arr, dtype=torch.float))
        else:
            warnings.warn(f"Champ {field} non trouvé, rempli avec des zéros")
            y_parts.append(torch.zeros(n_kept, 1))
    
    y = torch.cat(y_parts, dim=1)
    
    # Construction de la connectivité
    keep_set = set(keep.tolist())
    idx_map = {old: new for new, old in enumerate(keep)}
    
    if connectivity == "mesh":
        try:
            # Extraction des arêtes du maillage
            surf = cell_mesh.extract_all_edges()
            lines = surf.lines.reshape(-1, 3)
            src_a, dst_a = lines[:, 1], lines[:, 2]
            
            # Filtrage vectorisé
            keep_np = np.zeros(n_pts, dtype=bool)
            keep_np[keep] = True
            mask = keep_np[src_a] & keep_np[dst_a]
            
            src_f = np.array([idx_map[s] for s in src_a[mask]])
            dst_f = np.array([idx_map[d] for d in dst_a[mask]])
            
            # Graphe non dirigé
            edge_index = torch.tensor(
                np.stack([np.concatenate([src_f, dst_f]),
                         np.concatenate([dst_f, src_f])]),
                dtype=torch.long
            )
            
        except Exception as e:
            warnings.warn(f"Extraction du maillage échouée ({e}), fallback sur KNN")
            connectivity = "knn"
    
    if connectivity == "knn" or edge_index is None:
        # KNN comme fallback
        pos_np = pos.numpy()
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="ball_tree").fit(pos_np)
        _, idx = nbrs.kneighbors(pos_np)
        
        src = np.repeat(np.arange(n_kept), k_neighbors)
        dst = idx[:, 1:].flatten()  # exclure le point lui-même
        
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    
    # Features des arêtes
    s, d = edge_index
    delta = pos[s] - pos[d]
    dist_e = torch.norm(delta, dim=1, keepdim=True).clamp(min=1e-8)
    edge_attr = torch.cat([delta / dist_e, dist_e], dim=1)
    
    # Création du graphe
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_type=node_types[keep],
        pos=pos
    )
    
    # Métadonnées
    data.case_path = str(case_path)
    data.fields_out = fields_out
    data.physical_params = physical_params
    data.case_type = case_type
    data.time_step = ts
    data.dimension = dimension
    
    return data


# =============================================================================
# 4. NORMALIZER
# =============================================================================

class FieldNormalizer:
    """Z-score normalizer avec support multi-dimensionnel."""
    
    def __init__(self):
        self._stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._fitted = False
    
    def fit_from_dataset(self, dataset, device: str = "cpu") -> FieldNormalizer:
        """Calcule les statistiques sur le dataset."""
        all_x = []
        all_y = []
        all_e = []
        
        for data in dataset:
            all_x.append(data.x)
            all_y.append(data.y)
            all_e.append(data.edge_attr)
        
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_e = torch.cat(all_e, dim=0)
        
        for key, tensor in [("x", all_x), ("y", all_y), ("edge", all_e)]:
            mean = tensor.mean(dim=0).to(device)
            std = tensor.std(dim=0).to(device)
            std[std < 1e-8] = 1.0  # Éviter division par zéro
            
            self._stats[key] = {
                "mean": mean,
                "std": std,
                "dim": tensor.shape[1]
            }
        
        self._fitted = True
        print(f"[FieldNormalizer] fit: x={all_x.shape}, y={all_y.shape}, edge={all_e.shape}")
        return self
    
    def _encode(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Normalisation."""
        if not self._fitted:
            raise RuntimeError("Normalizer non entraîné. Appeler fit_from_dataset d'abord.")
        
        stats = self._stats[key]
        return (tensor.to(stats["mean"].device) - stats["mean"]) / stats["std"]
    
    def _decode(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Dénormalisation."""
        if not self._fitted:
            raise RuntimeError("Normalizer non entraîné.")
        
        stats = self._stats[key]
        return tensor.to(stats["mean"].device) * stats["std"] + stats["mean"]
    
    def encode_x(self, x): return self._encode("x", x)
    def decode_x(self, x): return self._decode("x", x)
    def encode_y(self, y): return self._encode("y", y)
    def decode_y(self, y): return self._decode("y", y)
    def encode_edge(self, e): return self._encode("edge", e)
    
    def save(self, path: Path):
        """Sauvegarde les statistiques."""
        torch.save(self._stats, path)
        print(f"[FieldNormalizer] → {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> FieldNormalizer:
        """Charge les statistiques."""
        normalizer = cls()
        stats = torch.load(path, map_location=device, weights_only=True)
        
        # Conversion explicite pour éviter les problèmes de device
        normalizer._stats = {
            k: {kk: vv.to(device) for kk, vv in v.items()}
            for k, v in stats.items()
        }
        normalizer._fitted = True
        
        print(f"[FieldNormalizer] ← {path}")
        return normalizer


# =============================================================================
# 5. MODELS (2D/3D générique)
# =============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron avec LayerNorm."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Pas de SiLU sur la dernière couche
                layers.append(nn.SiLU())
        
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        return self.norm(self.net(x))


class InteractionBlock(MessagePassing):
    """Bloc d'interaction avec mise à jour optionnelle des arêtes."""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, update_edges: bool = False):
        super().__init__(aggr="sum")
        
        self.update_edges = update_edges
        
        # MLP pour les messages (edge features)
        self.edge_mlp = MLP(node_dim * 2 + edge_dim, hidden_dim, edge_dim)
        
        # MLP pour la mise à jour des nœuds
        self.node_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim)
        
        # MLP optionnel pour la mise à jour des arêtes
        if update_edges:
            self.edge_update_mlp = MLP(edge_dim, hidden_dim, edge_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """Forward pass."""
        # Agrégation des messages
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Mise à jour des nœuds (résiduelle)
        x_new = x + self.node_mlp(torch.cat([x, agg], dim=-1))
        
        # Mise à jour optionnelle des arêtes
        if self.update_edges:
            edge_attr = edge_attr + self.edge_update_mlp(edge_attr)
        
        return x_new, edge_attr
    
    def message(self, x_i, x_j, edge_attr):
        """Calcul des messages."""
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class MeshGraphNet(nn.Module):
    """MeshGraphNet avec support 2D/3D."""
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 3,
        output_dim: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 6,
        use_dynamic_edges: bool = False,
    ):
        super().__init__()
        
        # Encodeurs
        self.node_enc = MLP(node_input_dim, hidden_dim, hidden_dim)
        self.edge_enc = MLP(edge_input_dim, hidden_dim, hidden_dim)
        
        # Blocs d'interaction
        self.blocks = nn.ModuleList([
            InteractionBlock(hidden_dim, hidden_dim, hidden_dim, use_dynamic_edges)
            for _ in range(n_layers)
        ])
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        """Forward pass."""
        h_node = self.node_enc(data.x)
        h_edge = self.edge_enc(data.edge_attr)
        
        for block in self.blocks:
            h_node, h_edge = block(h_node, data.edge_index, h_edge)
        
        return self.decoder(h_node)
    
    def n_params(self) -> int:
        """Nombre de paramètres."""
        return sum(p.numel() for p in self.parameters())


# Registry des modèles
MODELS = {
    "MeshGraphNet": MeshGraphNet,
    # Ajouter d'autres modèles ici
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Fabrique de modèles."""
    if name not in MODELS:
        raise ValueError(
            f"Modèle inconnu: '{name}'. "
            f"Disponibles: {list(MODELS)}"
        )
    return MODELS[name](**kwargs)


# =============================================================================
# 6. LOSS FUNCTIONS
# =============================================================================

class HybridLoss(nn.Module):
    """Loss hybride avec contraintes physiques."""
    
    def __init__(self, cfg: LossConfig, u_slice: slice, dimension: int = 2):
        super().__init__()
        self.cfg = cfg
        self.u_slice = u_slice
        self.dim = dimension
    
    def forward(self, pred: torch.Tensor, batch) -> Tuple[torch.Tensor, Dict]:
        """Calcul de la loss."""
        losses = {}
        
        # Loss data (tous les nœuds sauf parois)
        data_mask = (batch.node_type != NODE_WALL)
        if data_mask.any():
            losses["data"] = F.mse_loss(pred[data_mask], batch.y[data_mask])
        else:
            losses["data"] = torch.tensor(0.0, device=pred.device)
        
        # Loss inlet
        inlet_mask = (batch.node_type == NODE_INLET)
        if inlet_mask.any():
            losses["inlet"] = F.mse_loss(pred[inlet_mask], batch.y[inlet_mask])
        else:
            losses["inlet"] = torch.tensor(0.0, device=pred.device)
        
        # Loss no-slip (U=0 sur les parois)
        wall_mask = (batch.node_type == NODE_WALL)
        if wall_mask.any() and self.u_slice.stop > self.u_slice.start:
            u_pred = pred[wall_mask][:, self.u_slice]
            losses["no_slip"] = F.mse_loss(u_pred, torch.zeros_like(u_pred))
        else:
            losses["no_slip"] = torch.tensor(0.0, device=pred.device)
        
        # Pénalité de divergence (optionnelle)
        if self.cfg.divergence > 0 and self.dim >= 2:
            losses["divergence"] = self._compute_divergence(pred, batch)
        else:
            losses["divergence"] = torch.tensor(0.0, device=pred.device)
        
        # Loss totale pondérée
        total = (
            self.cfg.data * losses["data"] +
            self.cfg.inlet * losses["inlet"] +
            self.cfg.no_slip * losses["no_slip"] +
            self.cfg.divergence * losses["divergence"]
        )
        
        losses["total"] = total
        return total, losses
    
    def _compute_divergence(self, pred: torch.Tensor, batch) -> torch.Tensor:
        """Calcule la divergence de la vitesse (approximation par arêtes)."""
        if not hasattr(batch, "edge_index") or batch.edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Récupérer les indices des composantes de vitesse
        u_start, u_end = self.u_slice.start, self.u_slice.stop
        if u_start >= u_end:
            return torch.tensor(0.0, device=pred.device)
        
        # Vitesses aux nœuds
        u = pred[:, u_start:u_end]
        
        # Positions
        pos = batch.pos
        
        # Pour chaque arête, estimer la divergence
        s, d = batch.edge_index
        delta = pos[s] - pos[d]
        dist = torch.norm(delta, dim=1, keepdim=True).clamp(min=1e-8)
        direction = delta / dist
        
        # Projection de la différence de vitesse sur la direction
        u_diff = u[s] - u[d]
        div_approx = torch.abs((u_diff * direction).sum(dim=1)) / dist.squeeze()
        
        return div_approx.mean()


# =============================================================================
# 7. DATASET
# =============================================================================

class FoamGraphDataset(Dataset):
    """Dataset pour les graphes OpenFOAM."""
    
    def __init__(self, root: Path, transform=None, pre_filter=None):
        self.root = Path(root)
        self._files = sorted(self.root.glob("*.pt"))
        
        if not self._files:
            raise FileNotFoundError(f"Aucun fichier .pt trouvé dans {root}")
        
        print(f"[FoamGraphDataset] {len(self._files)} graphes chargés depuis {root}")
        
        super().__init__(root, transform, pre_filter)
    
    def len(self):
        return len(self._files)
    
    def get(self, idx: int):
        """Charge un graphe."""
        data = torch.load(self._files[idx], weights_only=False)
        
        # Vérifier la compatibilité
        if not hasattr(data, "dimension"):
            data.dimension = 2  # compatibilité ascendante
        
        return data
    
    def shuffle(self) -> FoamGraphDataset:
        """Mélange le dataset."""
        random.shuffle(self._files)
        return self
    
    def __getitem__(self, idx):
        """Support du slicing."""
        if isinstance(idx, slice):
            subset = object.__new__(FoamGraphDataset)
            subset.root = self.root
            subset._files = self._files[idx]
            subset._transform = getattr(self, "_transform", None)
            subset._pre_filter = getattr(self, "_pre_filter", None)
            return subset
        
        return self.get(idx)
    
    def __iter__(self):
        """Itérateur."""
        for i in range(len(self._files)):
            yield self.get(i)


# =============================================================================
# 8. METRICS PHYSIQUES (2D/3D)
# =============================================================================

def compute_physics_metrics(
    pred_raw: torch.Tensor,
    true_raw: torch.Tensor,
    fields: List[str],
    node_type: torch.Tensor,
    case_type: str = "generic",
    dimension: int = 2,
    rho: float = 1.225,
    U_inf: Optional[float] = None,
) -> Dict[str, float]:
    """Calcule des métriques physiques spécifiques au cas."""
    
    # Passage en numpy (sur CPU)
    mask = ((node_type == NODE_FLUID) | (node_type == NODE_OUTLET)).cpu().numpy()
    pred_np = pred_raw.cpu().numpy()[mask]
    true_np = true_raw.cpu().numpy()[mask]
    
    if len(pred_np) == 0:
        return {}
    
    f_idx = {f: i for i, f in enumerate(fields)}
    metrics = {}
    
    # Erreur relative par champ
    for f in fields:
        if f not in f_idx:
            continue
        i = f_idx[f]
        denom = np.abs(true_np[:, i]).mean() + 1e-8
        metrics[f"rel_err_{f}"] = float(
            np.abs(pred_np[:, i] - true_np[:, i]).mean() / denom
        )
    
    # Erreur sur la magnitude de la vitesse
    u_fields = [f for f in fields if f.upper().startswith("U")]
    if len(u_fields) >= 2:
        u_indices = [f_idx[f] for f in u_fields]
        
        U_pred = np.linalg.norm(pred_np[:, u_indices], axis=1)
        U_true = np.linalg.norm(true_np[:, u_indices], axis=1)
        
        metrics["rel_err_Umag"] = float(
            np.abs(U_pred - U_true).mean() / (U_true.mean() + 1e-8)
        )
        metrics["max_err_Umag"] = float(np.abs(U_pred - U_true).max())
    
    # Métriques spécifiques airfoil
    if case_type == "airfoil" and "p" in f_idx and U_inf:
        q = 0.5 * rho * U_inf ** 2
        Cp_pred = pred_np[:, f_idx["p"]] / q
        Cp_true = true_np[:, f_idx["p"]] / q
        
        metrics["rel_err_Cp"] = float(
            np.abs(Cp_pred - Cp_true).mean() / (np.abs(Cp_true).mean() + 1e-8)
        )
    
    # Métriques spécifiques nozzle
    if case_type == "nozzle" and "T" in f_idx and len(u_fields) >= 1:
        gamma, R = 1.4, 287.0
        
        T_pred = np.clip(pred_np[:, f_idx["T"]], 1.0, None)
        T_true = np.clip(true_np[:, f_idx["T"]], 1.0, None)
        
        u_indices = [f_idx[f] for f in u_fields]
        U_pred = np.linalg.norm(pred_np[:, u_indices], axis=1)
        U_true = np.linalg.norm(true_np[:, u_indices], axis=1)
        
        Ma_pred = U_pred / np.sqrt(gamma * R * T_pred)
        Ma_true = U_true / np.sqrt(gamma * R * T_true)
        
        metrics["rel_err_Mach"] = float(
            np.abs(Ma_pred - Ma_true).mean() / (Ma_true.mean() + 1e-8)
        )
        metrics["max_Ma_pred"] = float(Ma_pred.max())
    
    return metrics


# =============================================================================
# 9. PREDICTION ET VISUALISATION (2D/3D)
# =============================================================================

@torch.no_grad()
def predict_case(
    case_path: Union[str, Path],
    model: nn.Module,
    normalizer: FieldNormalizer,
    cfg: GNNConfig,
    physical_params: Optional[Dict] = None,
) -> Dict:
    """Prédiction sur un cas complet."""
    
    sampler = SmartSampler.from_preset("full")  # Garder tous les points
    data = foam_to_graph(
        Path(case_path),
        cfg.case_type,
        cfg.fields_out,
        physical_params,
        sampler=sampler,
        dimension=cfg.dimension
    )
    
    device = next(model.parameters()).device
    data = data.to(device)
    
    # Sauvegarde des données originales
    y_true = data.y.cpu().clone()
    pos = data.pos.cpu().clone()
    node_type = data.node_type.cpu().clone()
    
    # Normalisation
    data.x = normalizer.encode_x(data.x)
    data.edge_attr = normalizer.encode_edge(data.edge_attr)
    
    # Prédiction
    pred_enc = model(data)
    pred = normalizer.decode_y(pred_enc).cpu()
    
    # Assemblage des résultats
    result = {
        "pos": pos.numpy(),
        "node_type": node_type.numpy(),
        "dimension": cfg.effective_dimension,
    }
    
    for i, f in enumerate(cfg.fields_out):
        result[f"pred_{f}"] = pred[:, i].numpy()
        result[f"true_{f}"] = y_true[:, i].numpy()
    
    result["metrics"] = compute_physics_metrics(
        pred, y_true, cfg.fields_out, node_type,
        cfg.case_type, cfg.effective_dimension
    )
    
    return result


def compare_with_foam(
    case_path: Union[str, Path],
    model: nn.Module,
    normalizer: FieldNormalizer,
    cfg: GNNConfig,
    physical_params: Optional[Dict] = None,
    output_path: Optional[Path] = None,
) -> Dict:
    """Compare GNN et OpenFOAM avec visualisation."""
    
    result = predict_case(case_path, model, normalizer, cfg, physical_params)
    
    if output_path is None:
        output_path = Path.cwd() / f"compare_{Path(case_path).stem}.png"
    
    pos = result["pos"]
    dim = result["dimension"]
    fields = cfg.fields_out
    
    if dim == 2:
        _visualize_2d(result, fields, output_path)
    else:
        _visualize_3d(result, fields, output_path)
    
    print(f"[compare_with_foam] → {output_path}")
    return result


def _visualize_2d(result: Dict, fields: List[str], output_path: Path):
    """Visualisation 2D."""
    pos = result["pos"]
    
    fig, axes = plt.subplots(len(fields), 3, figsize=(15, 4 * len(fields)))
    if len(fields) == 1:
        axes = axes.reshape(1, -1)
    
    for row, f in enumerate(fields):
        true_vals = result[f"true_{f}"]
        pred_vals = result[f"pred_{f}"]
        err_vals = np.abs(pred_vals - true_vals)
        
        vmin, vmax = true_vals.min(), true_vals.max()
        
        # OpenFOAM
        sc0 = axes[row, 0].scatter(
            pos[:, 0], pos[:, 1], c=true_vals,
            s=1, cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[row, 0].set_title(f"{f} — OpenFOAM")
        axes[row, 0].set_aspect("equal")
        plt.colorbar(sc0, ax=axes[row, 0])
        
        # GNN
        sc1 = axes[row, 1].scatter(
            pos[:, 0], pos[:, 1], c=pred_vals,
            s=1, cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[row, 1].set_title(f"{f} — GNN")
        axes[row, 1].set_aspect("equal")
        plt.colorbar(sc1, ax=axes[row, 1])
        
        # Erreur
        sc2 = axes[row, 2].scatter(
            pos[:, 0], pos[:, 1], c=err_vals,
            s=1, cmap="hot", vmin=0, vmax=err_vals.max()
        )
        axes[row, 2].set_title(f"|err| max={err_vals.max():.2e}")
        axes[row, 2].set_aspect("equal")
        plt.colorbar(sc2, ax=axes[row, 2])
    
    plt.suptitle(f"GNN vs OpenFOAM — {Path(output_path).stem}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _visualize_3d(result: Dict, fields: List[str], output_path: Path):
    """Visualisation 3D (projection 2D par défaut)."""
    pos = result["pos"]
    
    # Pour la 3D, on visualise soit :
    # - Des coupes à z constant
    # - Une projection
    # - Une figure 3D interactive (mais pas en statique)
    
    # Version simple : scatter 3D avec couleur
    fig = plt.figure(figsize=(18, 6))
    
    for i, f in enumerate(fields[:3]):  # Limiter à 3 champs max
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        true_vals = result[f"true_{f}"]
        sc = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=true_vals, cmap="viridis", s=1
        )
        ax.set_title(f"{f} — OpenFOAM")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.colorbar(sc, ax=ax, shrink=0.5)
    
    plt.suptitle(f"GNN vs OpenFOAM (3D) — {Path(output_path).stem}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Avertissement pour visualisation avancée
    warnings.warn(
        "Visualisation 3D basique. Pour une analyse détaillée, "
        "utilisez ParaView ou des coupes 2D."
    )


# =============================================================================
# 10. EXPERIMENT
# =============================================================================

class Experiment:
    """Orchestre une expérience GNN complète (2D/3D)."""
    
    def __init__(self, cfg: GNNConfig):
        self.cfg = cfg
        self.exp_dir = cfg.experiment_dir()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde de la configuration
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        
        # Fixer les seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Experiment '{cfg.name}'] device={self.device}, dimension={cfg.effective_dimension}")
        
        # Initialisation
        self.sampler = SmartSampler.from_preset(
            cfg.sampler.preset or "full",
            cfg.sampler.zones
        )
        self.model = None
        self.normalizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.history = {}
    
    def extract_graphs(self, force: bool = False) -> Experiment:
        """Extrait les graphes des simulations OpenFOAM."""
        cfg = self.cfg
        graph_dir = Path(cfg.graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        sim_dirs = [p for p in sorted(Path(cfg.sim_dir).iterdir()) if p.is_dir()]
        print(f"\n[extract_graphs] {len(sim_dirs)} cas → {graph_dir}")
        
        n_ok = n_skip = n_fail = 0
        
        for case_path in sim_dirs:
            out_path = graph_dir / f"{case_path.name}.pt"
            
            if out_path.exists() and not force:
                n_skip += 1
                continue
            
            params = self._read_physical_params(case_path)
            if params is None:
                print(f"  ⚠  {case_path.name} : config.json introuvable, ignoré")
                n_fail += 1
                continue
            
            try:
                data = foam_to_graph(
                    case_path,
                    cfg.case_type,
                    cfg.fields_out,
                    params,
                    self.sampler,
                    "mesh",
                    dimension=cfg.dimension
                )
                torch.save(data, out_path)
                n_ok += 1
                print(f"  ✓  {case_path.name}  ({data.x.shape[0]} nœuds, "
                      f"{data.edge_index.shape[1]} arêtes, {data.dimension}D)")
                
            except Exception as e:
                print(f"  ✗  {case_path.name} : {e}")
                n_fail += 1
        
        print(f"[extract_graphs] ✓ {n_ok}  skip {n_skip}  ✗ {n_fail}")
        return self
    
    def _read_physical_params(self, case_path: Path) -> Optional[Dict]:
        """Lit les paramètres physiques depuis les fichiers de config."""
        # Chercher le fichier de configuration
        config_files = [
            case_path / "nozzle_config.json",
            case_path / "airfoil_config.json",
            case_path / "buildings_config.json",
            case_path / "config.json",
        ]
        
        config_path = None
        for f in config_files:
            if f.exists():
                config_path = f
                break
        
        if config_path is None:
            return None
        
        with open(config_path) as f:
            raw = json.load(f)
        
        ct = self.cfg.case_type
        
        try:
            if ct == "nozzle":
                nz = raw.get("nozzle", {})
                op = raw.get("operating", {})
                
                R_throat = nz.get("R_throat", 0.08)
                R_exit = nz.get("R_exit", 0.18)
                
                all_params = {
                    "R_throat": R_throat,
                    "R_exit": R_exit,
                    "p_total": op.get("p_total_inlet", 3e5),
                    "p_outlet": op.get("p_static_outlet", 1e4),
                    "AR_exit": (R_exit / max(R_throat, 1e-8)) ** 2,
                }
                
            elif ct == "airfoil":
                sim = raw.get("simulation", {})
                all_params = {
                    "angle_of_attack": sim.get("angle_of_attack", 0.0),
                    "Re": sim.get("Re", 1e6),
                    "Ma": sim.get("Ma", 0.1),
                }
                
            elif ct == "urban":
                sim = raw.get("simulation", {})
                dom = raw.get("domaine_fluide", {})
                all_params = {
                    "wind_angle": dom.get("rotation_angle", 0.0),
                    "wind_speed": sim.get("inlet_velocity", 5.0),
                    "turb_intensity": sim.get("turbulence_intensity", 0.05),
                }
                
            else:
                # Cas générique : prendre toutes les valeurs numériques
                all_params = {
                    k: v for k, v in raw.items()
                    if isinstance(v, (int, float))
                }
            
            # Filtrer selon les clés demandées
            return {
                k: all_params[k]
                for k in self.cfg.physical_params_keys
                if k in all_params
            }
            
        except Exception as e:
            warnings.warn(f"Erreur lecture params {case_path}: {e}")
            return None
    
    def load_dataset(self) -> Experiment:
        """Charge les datasets d'entraînement et validation."""
        cfg = self.cfg
        
        full_dataset = FoamGraphDataset(root=cfg.graph_dir).shuffle()
        
        # Split train/val
        n_train = int(len(full_dataset) * (1 - cfg.train.val_split))
        self.train_dataset = full_dataset[:n_train]
        self.val_dataset = full_dataset[n_train:]
        
        # Vérifier la dimension
        sample = self.train_dataset[0]
        if hasattr(sample, "dimension"):
            detected_dim = sample.dimension
            if self.cfg.dimension == 0:
                self.cfg.dimension = detected_dim
                print(f"  Dimension auto-détectée: {detected_dim}D")
            elif detected_dim != self.cfg.dimension:
                warnings.warn(
                    f"Dimension détectée ({detected_dim}D) "
                    f"différente de la config ({self.cfg.dimension}D)"
                )
        
        print(f"\n[load_dataset] {len(self.train_dataset)} train / {len(self.val_dataset)} val")
        print(f"  input_dim={self.train_dataset[0].x.shape[1]}, "
              f"output_dim={self.train_dataset[0].y.shape[1]}")
        
        return self
    
    def build_model(self) -> Experiment:
        """Construit le modèle GNN."""
        if self.train_dataset is None:
            self.load_dataset()
        
        cfg = self.cfg
        
        # Déterminer les dimensions d'entrée/sortie
        input_dim = self.train_dataset[0].x.shape[1]
        output_dim = self.train_dataset[0].y.shape[1]
        
        # Vérification
        expected_input = cfg.node_input_dim()
        if input_dim != expected_input:
            print(f"  ⚠  input_dim={input_dim} ≠ attendu {expected_input}")
        
        # Création du modèle
        self.model = get_model(
            cfg.model.name,
            node_input_dim=input_dim,
            edge_input_dim=cfg.model.edge_dim,
            output_dim=output_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
            use_dynamic_edges=cfg.model.use_dynamic_edges,
        ).to(self.device)
        
        print(f"\n[build_model] {cfg.model.name} — {self.model.n_params():,} paramètres")
        print(f"  {cfg.effective_dimension}D, hidden={cfg.model.hidden_dim}, "
              f"layers={cfg.model.n_layers}")
        
        return self
    
    def fit(self) -> Experiment:
        """Entraîne le modèle."""
        if self.model is None:
            self.build_model()
        
        cfg = self.cfg
        tc = cfg.train
        
        # Normalisation
        self.normalizer = FieldNormalizer()
        self.normalizer.fit_from_dataset(self.train_dataset, device=str(self.device))
        self.normalizer.save(self.exp_dir / "normalizer_stats.pt")
        
        # Loss
        loss_fn = HybridLoss(
            cfg.loss,
            u_slice=cfg.u_slice,
            dimension=cfg.effective_dimension
        ).to(self.device)
        
        # DataLoaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=tc.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=tc.batch_size,
            shuffle=False,
            num_workers=tc.num_workers,
            pin_memory=True
        )
        
        # Optimiseur
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=tc.lr_max,
            steps_per_epoch=len(train_loader),
            epochs=tc.epochs,
            pct_start=0.3
        )
        
        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if tc.use_amp and self.device.type == "cuda" else None
        
        # Historique
        self.history = {
            "train_total": [], "train_data": [], "train_inlet": [],
            "train_no_slip": [], "val_total": [], "val_data": [],
            "lr": []
        }
        
        best_val_loss = float("inf")
        no_improve = 0
        
        print(f"\n[fit] {tc.epochs} epochs max — patience={tc.patience}")
        
        for epoch in range(1, tc.epochs + 1):
            t0 = time.time()
            
            # Entraînement
            self.model.train()
            train_losses = {"total": 0.0, "data": 0.0, "inlet": 0.0, "no_slip": 0.0}
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Normalisation
                batch.x = self.normalizer.encode_x(batch.x)
                batch.edge_attr = self.normalizer.encode_edge(batch.edge_attr)
                batch.y = self.normalizer.encode_y(batch.y)
                
                # Bruit pour régularisation
                if tc.noise_std > 0:
                    batch.x = batch.x + torch.randn_like(batch.x) * tc.noise_std
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = self.model(batch)
                        loss, losses = loss_fn(pred, batch)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = self.model(batch)
                    loss, losses = loss_fn(pred, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                    optimizer.step()
                
                scheduler.step()
                
                for k in train_losses:
                    train_losses[k] += losses.get(k, torch.tensor(0.0)).item()
            
            for k in train_losses:
                train_losses[k] /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_losses = {"total": 0.0, "data": 0.0}
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    
                    batch.x = self.normalizer.encode_x(batch.x)
                    batch.edge_attr = self.normalizer.encode_edge(batch.edge_attr)
                    batch.y = self.normalizer.encode_y(batch.y)
                    
                    pred = self.model(batch)
                    _, losses = loss_fn(pred, batch)
                    
                    val_losses["total"] += losses["total"].item()
                    val_losses["data"] += losses.get("data", torch.tensor(0.0)).item()
            
            for k in val_losses:
                val_losses[k] /= max(len(val_loader), 1)
            
            # Mise à jour de l'historique
            self.history["train_total"].append(train_losses["total"])
            self.history["train_data"].append(train_losses["data"])
            self.history["train_inlet"].append(train_losses["inlet"])
            self.history["train_no_slip"].append(train_losses["no_slip"])
            self.history["val_total"].append(val_losses["total"])
            self.history["val_data"].append(val_losses["data"])
            self.history["lr"].append(scheduler.get_last_lr()[0])
            
            # Logging
            print(
                f"Ep {epoch:04d} | train {train_losses['total']:.5f} "
                f"(data={train_losses['data']:.5f}, noslip={train_losses['no_slip']:.5f}) | "
                f"val {val_losses['total']:.5f} | lr={self.history['lr'][-1]:.1e} | "
                f"{time.time()-t0:.1f}s"
            )
            
            # Early stopping
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                no_improve = 0
                torch.save(self.model.state_dict(), self.exp_dir / "best.pt")
                print(f"    ↳ best.pt (val={best_val_loss:.6f})")
            else:
                no_improve += 1
            
            # Checkpoints périodiques
            if epoch % 25 == 0:
                torch.save(
                    self.model.state_dict(),
                    self.exp_dir / f"ckpt_e{epoch:04d}.pt"
                )
            
            if no_improve >= tc.patience:
                print(f"  Early stopping à l'epoch {epoch}")
                break
        
        # Sauvegarde finale
        torch.save(self.model.state_dict(), self.exp_dir / "final.pt")
        self._plot_convergence()
        
        print(f"\n[fit] terminé — best val = {best_val_loss:.6f}")
        return self
    
    def _plot_convergence(self):
        """Trace la convergence."""
        if not self.history:
            return
        
        epochs = range(1, len(self.history["train_total"]) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.semilogy(epochs, self.history["train_total"], label="train total", color="steelblue")
        ax1.semilogy(epochs, self.history["val_total"], label="val total", color="tomato")
        ax1.semilogy(epochs, self.history["train_data"], label="train data", 
                     color="steelblue", ls="--", alpha=0.5)
        ax1.semilogy(epochs, self.history["train_no_slip"], label="no-slip",
                     color="orange", ls=":", alpha=0.7)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (log)")
        ax1.set_title(f"Convergence — {self.cfg.name} ({self.cfg.effective_dimension}D)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, self.history["lr"], color="seagreen")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("OneCycleLR")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "convergence.png", dpi=150)
        plt.close()
    
    def evaluate(self, dataset=None, load_best: bool = True) -> Dict:
        """Évalue le modèle sur un dataset."""
        if load_best:
            best_path = self.exp_dir / "best.pt"
            if best_path.exists():
                self.model.load_state_dict(
                    torch.load(best_path, map_location=self.device)
                )
                print(f"  Modèle chargé: {best_path}")
        
        self.model.eval()
        
        dataset = dataset or self.val_dataset
        if dataset is None:
            raise ValueError("Aucun dataset fourni pour l'évaluation")
        
        all_metrics = []
        
        with torch.no_grad():
            for data in dataset:
                data = data.to(self.device)
                
                # Normalisation
                x_norm = self.normalizer.encode_x(data.x)
                edge_norm = self.normalizer.encode_edge(data.edge_attr)
                
                # Prédiction
                pred_enc = self.model(data)
                pred = self.normalizer.decode_y(pred_enc).cpu()
                
                # Métriques
                metrics = compute_physics_metrics(
                    pred,
                    data.y.cpu(),
                    self.cfg.fields_out,
                    data.node_type.cpu(),
                    self.cfg.case_type,
                    self.cfg.effective_dimension
                )
                all_metrics.append(metrics)
        
        # Agrégation
        aggregated = {}
        if all_metrics:
            keys = all_metrics[0].keys()
            for key in keys:
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    aggregated[f"mean_{key}"] = float(np.mean(values))
                    aggregated[f"std_{key}"] = float(np.std(values))
        
        print(f"\n[evaluate] {len(dataset)} cas :")
        for k, v in aggregated.items():
            print(f"  {k:35s} = {v:.4f}")
        
        # Sauvegarde
        with open(self.exp_dir / "eval_metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        
        return aggregated
    
    def compare(self, case_path: Union[str, Path], output_name: Optional[str] = None) -> Dict:
        """Compare la prédiction GNN avec OpenFOAM."""
        case_path = Path(case_path)
        
        if output_name is None:
            output_name = f"compare_{case_path.name}.png"
        
        params = self._read_physical_params(case_path)
        
        return compare_with_foam(
            case_path,
            self.model,
            self.normalizer,
            self.cfg,
            params,
            self.exp_dir / output_name
        )
    
    @classmethod
    def load(cls, exp_dir: Path) -> Experiment:
        """Charge une expérience existante."""
        exp_dir = Path(exp_dir)
        
        # Chargement de la config
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration introuvable: {config_path}")
        
        with open(config_path) as f:
            raw = json.load(f)
        
        # Reconstruction de la config
        cfg = GNNConfig(
            name=raw["name"],
            case_type=raw["case_type"],
            fields_out=raw["fields_out"],
            physical_params_keys=raw["physical_params_keys"],
            sim_dir=Path(raw["sim_dir"]),
            graph_dir=Path(raw["graph_dir"]),
            model_dir=Path(raw["model_dir"]),
            dimension=raw.get("dimension", 2)
        )
        
        # Sous-configs
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        
        if "train" in raw:
            for k, v in raw["train"].items():
                if hasattr(cfg.train, k):
                    setattr(cfg.train, k, v)
        
        # Création de l'expérience
        exp = cls(cfg)
        
        # Chargement du dataset
        exp.load_dataset()
        
        # Construction du modèle
        exp.build_model()
        
        # Chargement des poids
        best_path = exp_dir / "best.pt"
        if best_path.exists():
            exp.model.load_state_dict(
                torch.load(best_path, map_location=exp.device)
            )
            print(f"  Modèle chargé: {best_path}")
        
        # Chargement du normalizer
        norm_path = exp_dir / "normalizer_stats.pt"
        if norm_path.exists():
            exp.normalizer = FieldNormalizer.load(norm_path, str(exp.device))
        
        return exp


# =============================================================================
# 11. API PUBLIQUE
# =============================================================================

__all__ = [
    # Configuration
    "GNNConfig",
    "SamplerConfig",
    "ModelConfig",
    "LossConfig",
    "TrainConfig",
    
    # Classes principales
    "Experiment",
    "SmartSampler",
    "FieldNormalizer",
    "FoamGraphDataset",
    "HybridLoss",
    
    # Modèles
    "get_model",
    "MeshGraphNet",
    
    # Utilitaires
    "foam_to_graph",
    "compute_physics_metrics",
    "predict_case",
    "compare_with_foam",
    
    # Constantes
    "NODE_FLUID",
    "NODE_INLET",
    "NODE_WALL",
    "NODE_OUTLET",
    "N_NODE_TYPES",
]