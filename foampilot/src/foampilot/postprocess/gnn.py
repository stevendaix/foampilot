#!/usr/bin/env python
"""
foampilot.gnn — Module GNN Surrogate GÉNÉRIQUE pour OpenFOAM
==============================================================

MODULE 100% GÉNÉRIQUE — aucun cas spécifique hardcodé.
Les configurations par cas vivent dans foampilot.cases.* ou dans vos scripts.

Usage :
    from foampilot.gnn import Experiment, GNNConfig
    
    # Configuration explicite (pas de .for_nozzle() hardcodé)
    cfg = GNNConfig(
        name="mon_cas",
        fields_out=["p", "T", "Ux", "Uy"],
        physical_params_keys=["param1", "param2"],
        sampler_zones=[(0.05, 1.0), (0.5, 0.5), (1e9, 0.1)],
    )
    
    exp = Experiment(cfg)
    exp.extract_graphs()
    exp.fit()
"""

from __future__ import annotations

import json
import random
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

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

from foampilot import postprocess


# =============================================================================
# 1. CONFIGURATION GÉNÉRIQUE
# =============================================================================

@dataclass
class GNNConfig:
    """
    Configuration 100% générique d'une expérience GNN.
    Aucun cas spécifique hardcodé — tout est paramétrable.
    """
    
    # ── Identité ──────────────────────────────────────────────────────────────
    name: str = "exp_001"
    
    # ── Champs de sortie (ce que le GNN prédit) ──────────────────────────────
    fields_out: List[str] = field(default_factory=lambda: ["p", "Ux", "Uy"])
    
    # ── Paramètres physiques d'entrée (features scalaires) ───────────────────
    # Ce sont les paramètres qui varient d'un cas à l'autre
    # Ex nozzle : ["R_throat", "R_exit", "p_total"]
    # Ex airfoil: ["angle_of_attack", "Re", "Ma"]
    physical_params_keys: List[str] = field(default_factory=list)
    
    # ── Paths ─────────────────────────────────────────────────────────────────
    sim_dir: Path = Path("simulations")
    graph_dir: Path = Path("graphs")
    model_dir: Path = Path("models")
    
    # ── Échantillonnage ───────────────────────────────────────────────────────
    # Liste de (dist_max_m, keep_rate) — aucun preset hardcodé
    sampler_zones: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.05, 1.00),  # couche limite
        (0.50, 0.33),  # champ proche
        (1e9,  0.10),  # champ lointain
    ])
    
    # ── Modèle ────────────────────────────────────────────────────────────────
    model_name: str = "MeshGraphNet"
    hidden_dim: int = 128
    n_layers: int = 6
    use_dynamic_edges: bool = False
    
    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_data: float = 1.0
    loss_inlet: float = 2.0
    loss_no_slip: float = 5.0
    
    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-3
    lr_max: float = 5e-3
    weight_decay: float = 1e-5
    patience: int = 20
    noise_std: float = 0.002
    val_split: float = 0.2
    num_workers: int = 2
    gradient_clip: float = 1.0
    use_amp: bool = False  # désactivé par défaut (sécurité)
    
    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    dimension: int = 0  # 0 = auto-détection, 2 ou 3 = forcé
    
    # ── Callbacks optionnels (injectés de l'extérieur) ───────────────────────
    # Permet d'injecter des fonctions custom sans modifier le module
    param_reader: Optional[Callable[[Path], Optional[Dict]]] = None
    metrics_computer: Optional[Callable] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Propriétés calculées
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def output_dim(self) -> int:
        return len(self.fields_out)
    
    @property
    def u_slice(self) -> slice:
        """Indices des composantes de vitesse dans fields_out."""
        u_idx = [i for i, f in enumerate(self.fields_out) if f.upper().startswith("U")]
        return slice(u_idx[0], u_idx[-1] + 1) if u_idx else slice(0, 0)
    
    @property
    def n_physical_params(self) -> int:
        return len(self.physical_params_keys)
    
    def node_input_dim(self, n_node_types: int = 6) -> int:
        """Dimension d'entrée : pos + type_onehot + dist_wall + params."""
        pos_dim = self.dimension if self.dimension > 0 else 2
        return pos_dim + n_node_types + 1 + self.n_physical_params
    
    def experiment_dir(self) -> Path:
        return Path(self.model_dir) / self.name
    
    def to_dict(self) -> dict:
        """Sérialisation JSON."""
        d = asdict(self)
        # Conversion Path → str
        for k in ["sim_dir", "graph_dir", "model_dir"]:
            d[k] = str(d[k])
        # Supprimer les callables (non sérialisables)
        d.pop("param_reader", None)
        d.pop("metrics_computer", None)
        return d


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


# =============================================================================
# 3. GRAPH EXTRACTION GÉNÉRIQUE
# =============================================================================

class SmartSampler:
    """Échantillonnage adaptatif générique."""
    
    def __init__(self, zones: List[Tuple[float, float]]):
        self.zones = sorted(zones, key=lambda z: z[0])
        self._validate()
    
    def _validate(self):
        for i, (d_max, rate) in enumerate(self.zones):
            if rate <= 0 or rate > 1:
                raise ValueError(f"Taux invalide: {rate}")
            if i > 0 and d_max <= self.zones[i-1][0]:
                raise ValueError("Zones non croissantes")
    
    def sample(self, dist_to_wall: np.ndarray, node_types: Optional[np.ndarray] = None) -> np.ndarray:
        """Échantillonne les indices à conserver."""
        n = len(dist_to_wall)
        keep = np.zeros(n, dtype=bool)
        
        # Toujours garder les frontières
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
            # Shuffle pour éviter biais de grille
            zone_list = zone_idx.tolist()
            random.shuffle(zone_list)
            keep[zone_list[:n_keep]] = True
        
        return np.where(keep)[0]


def foam_to_graph(
    case_path: Path,
    fields_out: List[str],
    physical_params: Dict[str, float],
    sampler: Optional[SmartSampler] = None,
    connectivity: str = "mesh",
    k_neighbors: int = 6,
    time_step: str = "latest",
    dimension: int = 0,
) -> Data:
    """
    Convertit un cas OpenFOAM en graphe PyG.
    100% générique — aucune logique métier spécifique.
    """
    
    case_path = Path(case_path)
    if not case_path.exists():
        raise FileNotFoundError(f"Cas introuvable: {case_path}")
    
    # Chargement OpenFOAM
    foam_post = postprocess.FoamPostProcessing(case_path=case_path)
    available = foam_post.get_all_time_steps()
    if not available:
        raise ValueError(f"Aucun time step: {case_path}")
    
    ts = available[-1] if time_step == "latest" else time_step
    structure = foam_post.load_time_step(ts)
    cell_mesh = structure["cell"]
    boundaries = structure["boundaries"]
    
    n_pts = cell_mesh.n_points
    points = cell_mesh.points
    
    # Auto-détection dimension
    if dimension == 0:
        if points.shape[1] >= 3 and np.allclose(points[:, 2], 0.0, atol=1e-6):
            dimension = 2
        else:
            dimension = points.shape[1]
    
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
        patch_pts = patch_mesh.points[:, :dimension]
        if len(patch_pts) > 0:
            _, ids = kd_mesh.query(patch_pts)
            node_types[ids] = ntype
    
    # Distance paroi
    wall_mask = (node_types.numpy() == NODE_WALL)
    wall_pts = points[wall_mask][:, :dimension]
    if len(wall_pts) > 0:
        dist_wall, _ = KDTree(wall_pts).query(points[:, :dimension])
    else:
        dist_wall = np.ones(n_pts) * 1e3
        warnings.warn("Aucun nœud WALL trouvé")
    
    # Échantillonnage
    keep = sampler.sample(dist_wall, node_types.numpy()) if sampler else np.arange(n_pts)
    n_kept = len(keep)
    
    # Features noeuds
    pos_full = torch.tensor(points[:, :dimension], dtype=torch.float)
    pos = pos_full[keep]
    type_oh = F.one_hot(node_types[keep].clamp(0, N_NODE_TYPES-1), N_NODE_TYPES).float()
    dist_t = torch.tensor(dist_wall[keep], dtype=torch.float).unsqueeze(1)
    
    # Paramètres physiques (triés pour reproductibilité)
    if physical_params:
        param_vals = [float(physical_params[k]) for k in sorted(physical_params.keys())]
        param_t = torch.tensor([param_vals] * n_kept, dtype=torch.float)
    else:
        param_t = torch.zeros(n_kept, 0)
    
    x = torch.cat([pos, type_oh, dist_t, param_t], dim=1)
    
    # Cibles
    y_parts = []
    for field in fields_out:
        if field in cell_mesh.point_data:
            arr = cell_mesh.point_data[field][keep]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            y_parts.append(torch.tensor(arr, dtype=torch.float))
        else:
            warnings.warn(f"Champ {field} absent, rempli avec zéros")
            y_parts.append(torch.zeros(n_kept, 1))
    y = torch.cat(y_parts, dim=1)
    
    # Connectivité
    keep_set = set(keep.tolist())
    idx_map = {old: new for new, old in enumerate(keep)}
    edge_index = None
    
    if connectivity == "mesh":
        try:
            surf = cell_mesh.extract_all_edges()
            lines = surf.lines.reshape(-1, 3)
            src_a, dst_a = lines[:, 1], lines[:, 2]
            # Filtrage vectorisé
            keep_np = np.zeros(n_pts, dtype=bool)
            keep_np[keep] = True
            mask = keep_np[src_a] & keep_np[dst_a]
            src_f = np.array([idx_map[s] for s in src_a[mask]])
            dst_f = np.array([idx_map[d] for d in dst_a[mask]])
            edge_index = torch.tensor(
                np.stack([np.concatenate([src_f, dst_f]),
                         np.concatenate([dst_f, src_f])]),
                dtype=torch.long
            )
        except Exception as e:
            warnings.warn(f"Extraction maillage échouée ({e}), fallback KNN")
            connectivity = "knn"
    
    if connectivity == "knn" or edge_index is None:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm="ball_tree").fit(pos.numpy())
        _, idx = nbrs.kneighbors(pos.numpy())
        src = np.repeat(np.arange(n_kept), k_neighbors)
        dst = idx[:, 1:].flatten()
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    
    # Features arêtes
    s, d = edge_index
    delta = pos[s] - pos[d]
    dist_e = torch.norm(delta, dim=1, keepdim=True).clamp(min=1e-8)
    edge_attr = torch.cat([delta / dist_e, dist_e], dim=1)
    
    # Graphe
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                node_type=node_types[keep], pos=pos)
    data.case_path = str(case_path)
    data.fields_out = fields_out
    data.physical_params = physical_params
    data.time_step = ts
    data.dimension = dimension
    return data


# =============================================================================
# 4. NORMALIZER
# =============================================================================

class FieldNormalizer:
    """Z-score générique."""
    
    def __init__(self):
        self._stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._fitted = False
    
    def fit_from_dataset(self, dataset, device: str = "cpu") -> FieldNormalizer:
        all_x = torch.cat([d.x for d in dataset], dim=0)
        all_y = torch.cat([d.y for d in dataset], dim=0)
        all_e = torch.cat([d.edge_attr for d in dataset], dim=0)
        
        for key, t in [("x", all_x), ("y", all_y), ("edge", all_e)]:
            mean = t.mean(dim=0).to(device)
            std = t.std(dim=0).to(device)
            std[std < 1e-8] = 1.0
            self._stats[key] = {"mean": mean, "std": std}
        
        self._fitted = True
        print(f"[FieldNormalizer] fit: x={all_x.shape}, y={all_y.shape}, edge={all_e.shape}")
        return self
    
    def _enc(self, k, t):
        if not self._fitted:
            raise RuntimeError("Normalizer non entraîné")
        s = self._stats[k]
        return (t.to(s["mean"].device) - s["mean"]) / s["std"]
    
    def _dec(self, k, t):
        if not self._fitted:
            raise RuntimeError("Normalizer non entraîné")
        s = self._stats[k]
        return t.to(s["mean"].device) * s["std"] + s["mean"]
    
    def encode_x(self, x): return self._enc("x", x)
    def decode_x(self, x): return self._dec("x", x)
    def encode_y(self, y): return self._enc("y", y)
    def decode_y(self, y): return self._dec("y", y)
    def encode_edge(self, e): return self._enc("edge", e)
    
    def save(self, path: Path):
        torch.save(self._stats, path)
        print(f"[FieldNormalizer] → {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> FieldNormalizer:
        n = cls()
        n._stats = {k: {kk: vv.to(device) for kk, vv in v.items()}
                    for k, v in torch.load(path, map_location=device, weights_only=True).items()}
        n._fitted = True
        print(f"[FieldNormalizer] ← {path}")
        return n


# =============================================================================
# 5. MODELS (code inchangé — déjà générique)
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_d: int, h_d: int, out_d: int, n: int = 2):
        super().__init__()
        dims = [in_d] + [h_d] * (n - 1) + [out_d]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_d)
    
    def forward(self, x):
        return self.norm(self.net(x))


class InteractionBlock(MessagePassing):
    def __init__(self, node_d: int, edge_d: int, h_d: int, update_edges: bool = False):
        super().__init__(aggr="sum")
        self.update_edges = update_edges
        self.edge_mlp = MLP(node_d * 2 + edge_d, h_d, edge_d)
        self.node_mlp = MLP(node_d + edge_d, h_d, node_d)
        if update_edges:
            self.edge_update_mlp = MLP(edge_d, h_d, edge_d)
    
    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_new = x + self.node_mlp(torch.cat([x, agg], dim=-1))
        if self.update_edges:
            edge_attr = edge_attr + self.edge_update_mlp(edge_attr)
        return x_new, edge_attr
    
    def message(self, x_i, x_j, edge_attr):
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class MeshGraphNet(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int = 3,
                 output_dim: int = 3, hidden_dim: int = 128, n_layers: int = 6,
                 use_dynamic_edges: bool = False):
        super().__init__()
        h = hidden_dim
        self.node_enc = MLP(node_input_dim, h, h)
        self.edge_enc = MLP(edge_input_dim, h, h)
        self.blocks = nn.ModuleList([
            InteractionBlock(h, h, h, use_dynamic_edges) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, output_dim))
    
    def forward(self, data):
        h = self.node_enc(data.x)
        e = self.edge_enc(data.edge_attr)
        for block in self.blocks:
            h, e = block(h, data.edge_index, e)
        return self.decoder(h)
    
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


MODELS = {"MeshGraphNet": MeshGraphNet}


def get_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Modèle inconnu: '{name}'. Disponibles: {list(MODELS)}")
    return MODELS[name](**kwargs)


# =============================================================================
# 6. LOSS GÉNÉRIQUE
# =============================================================================

class HybridLoss(nn.Module):
    """Loss générique — pas de calcul physique spécifique."""
    
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, pred: torch.Tensor, batch) -> Tuple[torch.Tensor, Dict]:
        losses = {}
        
        # Data loss (tous noeuds sauf paroi)
        data_mask = (batch.node_type != NODE_WALL)
        losses["data"] = F.mse_loss(pred[data_mask], batch.y[data_mask]) if data_mask.any() else torch.tensor(0.0, device=pred.device)
        
        # Inlet loss
        inlet_mask = (batch.node_type == NODE_INLET)
        losses["inlet"] = F.mse_loss(pred[inlet_mask], batch.y[inlet_mask]) if inlet_mask.any() else torch.tensor(0.0, device=pred.device)
        
        # No-slip loss (U=0 sur paroi)
        wall_mask = (batch.node_type == NODE_WALL)
        u_slice = self.cfg.u_slice
        if wall_mask.any() and u_slice.stop > u_slice.start:
            u_pred = pred[wall_mask][:, u_slice]
            losses["no_slip"] = F.mse_loss(u_pred, torch.zeros_like(u_pred))
        else:
            losses["no_slip"] = torch.tensor(0.0, device=pred.device)
        
        # Loss totale
        total = (
            self.cfg.loss_data * losses["data"] +
            self.cfg.loss_inlet * losses["inlet"] +
            self.cfg.loss_no_slip * losses["no_slip"]
        )
        losses["total"] = total
        return total, losses


# =============================================================================
# 7. DATASET
# =============================================================================

class FoamGraphDataset(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)
        self._files = sorted(self.root.glob("*.pt"))
        if not self._files:
            raise FileNotFoundError(f"Aucun .pt dans {root}")
        print(f"[FoamGraphDataset] {len(self._files)} graphes dans {root}")
    
    def len(self):
        return len(self._files)
    
    def get(self, idx: int):
        return torch.load(self._files[idx], weights_only=False)
    
    def shuffle(self) -> FoamGraphDataset:
        random.shuffle(self._files)
        return self
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            sub = object.__new__(FoamGraphDataset)
            sub.root = self.root
            sub._files = self._files[idx]
            return sub
        return self.get(idx)


# =============================================================================
# 8. MÉTRIQUES GÉNÉRIQUES
# =============================================================================

def compute_generic_metrics(
    pred_raw: torch.Tensor,
    true_raw: torch.Tensor,
    fields: List[str],
    node_type: torch.Tensor,
) -> Dict[str, float]:
    """
    Métriques génériques — pas de calcul physique spécifique.
    Les métriques métier (Cp, Mach) sont dans les callbacks utilisateur.
    """
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
        metrics[f"rel_err_{f}"] = float(np.abs(pred_np[:, i] - true_np[:, i]).mean() / denom)
        metrics[f"max_err_{f}"] = float(np.abs(pred_np[:, i] - true_np[:, i]).max())
    
    # Magnitude vitesse (si plusieurs composantes U)
    u_fields = [f for f in fields if f.upper().startswith("U")]
    if len(u_fields) >= 2:
        u_idx = [f_idx[f] for f in u_fields]
        U_pred = np.linalg.norm(pred_np[:, u_idx], axis=1)
        U_true = np.linalg.norm(true_np[:, u_idx], axis=1)
        metrics["rel_err_Umag"] = float(np.abs(U_pred - U_true).mean() / (U_true.mean() + 1e-8))
        metrics["max_err_Umag"] = float(np.abs(U_pred - U_true).max())
    
    return metrics


# =============================================================================
# 9. EXPERIMENT GÉNÉRIQUE
# =============================================================================

class Experiment:
    """Expérience GNN 100% générique."""
    
    def __init__(self, cfg: GNNConfig):
        self.cfg = cfg
        self.exp_dir = cfg.experiment_dir()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Experiment '{cfg.name}'] device={self.device}")
        
        self.sampler = SmartSampler(cfg.sampler_zones)
        self.model = None
        self.normalizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.history = {}
    
    def extract_graphs(self, force: bool = False) -> Experiment:
        """Extraction générique des graphes."""
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
            
            # Lecture des paramètres physiques via callback utilisateur
            if cfg.param_reader is None:
                warnings.warn("Aucun param_reader fourni — utilisation de paramètres vides")
                params = {}
            else:
                params = cfg.param_reader(case_path)
                if params is None:
                    print(f"  ⚠  {case_path.name} : param_reader a retourné None, ignoré")
                    n_fail += 1
                    continue
            
            try:
                data = foam_to_graph(
                    case_path, cfg.fields_out, params, self.sampler,
                    "mesh", dimension=cfg.dimension
                )
                torch.save(data, out_path)
                n_ok += 1
                print(f"  ✓  {case_path.name}  ({data.x.shape[0]} nœuds, {data.edge_index.shape[1]} arêtes)")
            except Exception as e:
                print(f"  ✗  {case_path.name} : {e}")
                n_fail += 1
        
        print(f"[extract_graphs] ✓ {n_ok}  skip {n_skip}  ✗ {n_fail}")
        return self
    
    def load_dataset(self) -> Experiment:
        cfg = self.cfg
        full = FoamGraphDataset(root=cfg.graph_dir).shuffle()
        n_tr = int(len(full) * (1 - cfg.val_split))
        self.train_dataset, self.val_dataset = full[:n_tr], full[n_tr:]
        print(f"\n[load_dataset] {len(self.train_dataset)} train / {len(self.val_dataset)} val")
        return self
    
    def build_model(self) -> Experiment:
        if self.train_dataset is None:
            self.load_dataset()
        cfg = self.cfg
        input_dim = self.train_dataset[0].x.shape[1]
        output_dim = self.train_dataset[0].y.shape[1]
        self.model = get_model(
            cfg.model_name, node_input_dim=input_dim, edge_input_dim=3,
            output_dim=output_dim, hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers, use_dynamic_edges=cfg.use_dynamic_edges
        ).to(self.device)
        print(f"\n[build_model] {cfg.model_name} — {self.model.n_params():,} paramètres")
        return self
    
    def fit(self) -> Experiment:
        if self.model is None:
            self.build_model()
        cfg = self.cfg
        
        # Normalizer
        self.normalizer = FieldNormalizer().fit_from_dataset(self.train_dataset, str(self.device))
        self.normalizer.save(self.exp_dir / "normalizer_stats.pt")
        
        # Loss + optimiseur
        loss_fn = HybridLoss(cfg).to(self.device)
        train_loader = DataLoader(self.train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr_max, steps_per_epoch=len(train_loader), epochs=cfg.epochs, pct_start=0.3
        )
        scaler = torch.cuda.amp.GradScaler() if cfg.use_amp and self.device.type == "cuda" else None
        
        self.history = {k: [] for k in ["train_total", "train_data", "train_no_slip", "val_total", "lr"]}
        best_val, no_improve = float("inf"), 0
        
        print(f"\n[fit] {cfg.epochs} epochs max — patience={cfg.patience}")
        
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            self.model.train()
            tl = {"total": 0.0, "data": 0.0, "no_slip": 0.0}
            
            for batch in train_loader:
                batch = batch.to(self.device)
                batch.x = self.normalizer.encode_x(batch.x)
                batch.edge_attr = self.normalizer.encode_edge(batch.edge_attr)
                batch.y = self.normalizer.encode_y(batch.y)
                if cfg.noise_std > 0:
                    batch.x += torch.randn_like(batch.x) * cfg.noise_std
                
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = self.model(batch)
                        loss, ld = loss_fn(pred, batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = self.model(batch)
                    loss, ld = loss_fn(pred, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                    optimizer.step()
                scheduler.step()
                for k in tl:
                    tl[k] += ld.get(k, torch.tensor(0.0)).item()
            for k in tl:
                tl[k] /= len(train_loader)
            
            self.model.eval()
            vl = {"total": 0.0}
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    batch.x = self.normalizer.encode_x(batch.x)
                    batch.edge_attr = self.normalizer.encode_edge(batch.edge_attr)
                    batch.y = self.normalizer.encode_y(batch.y)
                    pred = self.model(batch)
                    _, ld = loss_fn(pred, batch)
                    vl["total"] += ld["total"].item()
            vl["total"] /= max(len(val_loader), 1)
            
            for k in ["train_total", "train_data", "train_no_slip"]:
                self.history[k].append(tl[k[6:]])
            self.history["val_total"].append(vl["total"])
            self.history["lr"].append(scheduler.get_last_lr()[0])
            
            print(f"Ep {epoch:04d} | train {tl['total']:.5f} | val {vl['total']:.5f} | {time.time()-t0:.1f}s")
            
            if vl["total"] < best_val:
                best_val, no_improve = vl["total"], 0
                torch.save(self.model.state_dict(), self.exp_dir / "best.pt")
            else:
                no_improve += 1
            if epoch % 25 == 0:
                torch.save(self.model.state_dict(), self.exp_dir / f"ckpt_e{epoch:04d}.pt")
            if no_improve >= cfg.patience:
                print(f"  Early stopping à l'epoch {epoch}")
                break
        
        torch.save(self.model.state_dict(), self.exp_dir / "final.pt")
        print(f"\n[fit] terminé — best val = {best_val:.6f}")
        return self
    
    def evaluate(self, load_best: bool = True) -> Dict:
        if load_best:
            self.model.load_state_dict(torch.load(self.exp_dir / "best.pt", map_location=self.device))
        self.model.eval()
        
        all_metrics = []
        with torch.no_grad():
            for data in self.val_dataset:
                data = data.to(self.device)
                pred = self.normalizer.decode_y(self.model(data)).cpu()
                
                # Métriques génériques
                m = compute_generic_metrics(pred, data.y.cpu(), self.cfg.fields_out, data.node_type.cpu())
                
                # Métriques custom via callback
                if self.cfg.metrics_computer:
                    m_custom = self.cfg.metrics_computer(pred, data.y.cpu(), self.cfg.fields_out, data.node_type.cpu())
                    m.update(m_custom)
                
                all_metrics.append(m)
        
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0]:
                vals = [mm[key] for mm in all_metrics if key in mm]
                aggregated[f"mean_{key}"] = float(np.mean(vals))
                aggregated[f"std_{key}"] = float(np.std(vals))
        
        print(f"\n[evaluate] {len(self.val_dataset)} cas :")
        for k, v in aggregated.items():
            print(f"  {k:35s} = {v:.4f}")
        
        with open(self.exp_dir / "eval_metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        return aggregated


# =============================================================================
# 10. API
# =============================================================================

__all__ = [
    "GNNConfig", "Experiment", "SmartSampler", "FieldNormalizer",
    "foam_to_graph", "get_model", "HybridLoss", "FoamGraphDataset",
    "compute_generic_metrics",
    "NODE_FLUID", "NODE_INLET", "NODE_WALL", "NODE_OUTLET", "N_NODE_TYPES",
]
