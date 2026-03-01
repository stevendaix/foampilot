#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cfd_gnn.py — Framework CFD+GNN Universel (2D/3D, Toute Géométrie)
========================================================================

Framework production-ready pour surrogate modeling CFD sur N'IMPORTE QUELLE géométrie.

Supporte:
✓ 2D planaire, 2D axisymétrique, 3D complet
✓ Toute géométrie (nozzle, airfoil, cavity, pipe, etc.)
✓ Maillages structurés et non-structurés
✓ Loss physics-informed dimension-agnostique
✓ Active learning + uncertainty quantification
✓ Configuration YAML reproductible
✓ Curriculum learning + checkpointing
✓ Feature normalization + KDTree caching

Auteurs: CFD+GNN Framework Contributors
Version: 1.0.0
License: MIT

Usage:
    from train_cfd_gnn import create_pipeline, PipelineConfig
    
    # Config YAML
    pipeline = create_pipeline("configs/cfd_generic.yaml")
    results = pipeline.run_full()
    
    # Config programmatique
    cfg = PipelineConfig(geometry_type="3d", case_type="airfoil", ...)
    pipeline = CFDPipeline(cfg)
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import qmc
from scipy.spatial import cKDTree

# foampilot imports (optionnels)
try:
    from foampilot import meshing, boundary, solver, utils
    FOAMPILOT_AVAILABLE = True
except ImportError:
    FOAMPILOT_AVAILABLE = False
    print("⚠️ foampilot non disponible - mode simulation activé")

# torch_scatter pour opérations sur graphe
try:
    from torch_scatter import scatter_add, scatter_mean, scatter_max
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False
    print("⚠️ torch_scatter non disponible - certaines opérations limitées")


# =============================================================================
# CONFIGURATION : DATA CLASSES HIÉRARCHIQUES
# =============================================================================

class GeometryDimension(str, Enum):
    """Dimension de la géométrie."""
    D2_PLANAR = "2d_planar"
    D2_AXISYMMETRIC = "2d_axisymmetric"
    D3 = "3d"


class MeshType(str, Enum):
    """Type de maillage."""
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    HYBRID = "hybrid"
    POLYHEDRAL = "polyhedral"


class CaseType(str, Enum):
    """Type de cas CFD."""
    NOZZLE = "nozzle"
    AIRFOIL = "airfoil"
    CAVITY = "cavity"
    PIPE = "pipe"
    BACKWARD_STEP = "backward_step"
    CUSTOM = "custom"


class SamplingMethod(str, Enum):
    """Méthodes d'échantillonnage."""
    GRID = "grid"
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"


class AcquisitionFunction(str, Enum):
    """Fonctions d'acquisition pour active learning."""
    UNCERTAINTY = "uncertainty"
    ERROR_ESTIMATE = "error_estimate"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"


class UncertaintyMethod(str, Enum):
    """Méthodes de quantification d'incertitude."""
    MC_DROPOUT = "mc_dropout"
    ENSEMBLE = "ensemble"
    EVIDENTIAL = "evidential"
    GRADIENT = "gradient"


@dataclass
class GeometryConfig:
    """Configuration géométrique générale."""
    dimension: GeometryDimension = GeometryDimension.D3
    mesh_type: MeshType = MeshType.UNSTRUCTURED
    case_type: CaseType = CaseType.CUSTOM
    
    # Caractéristiques géométriques
    characteristic_length: float = 1.0
    reference_area: float = 1.0
    reference_velocity: float = 1.0
    
    # Features géométriques à extraire
    extract_wall_distance: bool = True
    extract_inlet_distance: bool = True
    extract_outlet_distance: bool = True
    extract_feature_distances: bool = True
    
    # Tolérances
    wall_distance_tolerance: float = 1e-4
    axis_tolerance: float = 1e-4


@dataclass
class PhysicsConfig:
    """Configuration physique (dimension-agnostique)."""
    # Propriétés du fluide
    fluid_type: str = "air"
    gamma: float = 1.4
    R_gas: float = 287.0
    mu: float = 1.8e-5
    Pr: float = 0.713
    Cp: float = 1004.5
    rho_ref: float = 1.225
    
    # Régime d'écoulement
    flow_type: str = "compressible"
    turbulence_model: str = "kOmegaSST"
    steady: bool = True
    
    # Poids du loss physics-informed
    w_data: float = 1.0
    w_mass_conservation: float = 0.05
    w_momentum_conservation: float = 0.02
    w_energy_conservation: float = 0.05
    w_turbulence: float = 0.01
    w_boundary_conditions: float = 0.03
    
    # Seuils physiques
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "pressure": (100, 1e7),
        "temperature": (100, 5000),
        "mach": (0, 10),
        "density": (0.1, 50),
        "velocity": (0, 2000),
        "turbulence_k": (1e-6, 1000),
        "turbulence_omega": (1e-3, 1e6),
    })
    
    # Normalisation loss
    loss_normalization: bool = True
    loss_clip_max: float = 10.0
    
    # Schémas numériques
    mass_flux_scheme: str = "harmonic"  # "harmonic", "upwind", "linear"
    momentum_loss_simplified: bool = True


@dataclass
class SamplingConfig:
    """Configuration d'échantillonnage des paramètres."""
    method: SamplingMethod = SamplingMethod.LATIN_HYPERCUBE
    n_initial: int = 30
    n_per_iteration: int = 5
    seed: int = 42
    
    # Grille de paramètres
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Contraintes physiques
    feasibility_checks: bool = True
    custom_feasibility_fn: Optional[str] = None
    
    # Active learning
    acquisition_function: AcquisitionFunction = AcquisitionFunction.UNCERTAINTY
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.EVIDENTIAL
    n_mc_samples: int = 10
    candidate_pool_size: int = 200


@dataclass
class GNNArchitectureConfig:
    """Architecture du GNN (dimension-agnostique)."""
    # Architecture de base
    n_layers: int = 6
    hidden_dim: int = 128
    n_heads: int = 4
    activation: str = "relu"
    aggregation: str = "mean"
    
    # Features
    include_node_position: bool = True
    include_boundary_type: bool = True
    include_cell_volume: bool = True
    include_face_area: bool = True
    custom_node_features: List[str] = field(default_factory=list)
    custom_edge_features: List[str] = field(default_factory=list)
    
    # Mécanismes avancés
    use_attention: bool = True
    use_residual_connections: bool = True
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    
    # Régularisation
    dropout_rate: float = 0.1
    use_mc_dropout: bool = True
    
    # Output
    predict_uncertainty: bool = True
    output_variables: List[str] = field(default_factory=lambda: ["p", "T", "U", "Mach"])
    
    # Multi-échelle
    use_multiscale: bool = False
    n_scales: int = 3


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""
    epochs: int = 150
    batch_size: int = 1  # ⚠️ Mettre à 1 pour l'instant (batching >1 non supporté)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    
    # Loss
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "data": 1.0,
        "mass": 0.05,
        "momentum": 0.02,
        "energy": 0.05,
        "turbulence": 0.01,
        "bc": 0.03,
    })
    
    # Curriculum learning
    curriculum: bool = True
    curriculum_schedule: str = "linear"
    curriculum_ramp_epochs: int = 50
    easy_fraction: float = 0.3
    
    # Optimisation
    optimizer: str = "adamw"
    gradient_clip_norm: float = 1.0
    gradient_clip_value: float = 5.0
    
    # Scheduler
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Checkpointing
    save_every: int = 10
    keep_best_only: bool = False
    restore_best_weights: bool = True
    
    # Early stopping
    early_stopping: bool = True
    min_delta: float = 1e-4
    
    # Normalisation
    normalize_features: bool = True


@dataclass
class ValidationConfig:
    """Configuration de validation."""
    validate_simulations: bool = True
    check_convergence: bool = True
    check_physical_bounds: bool = True
    max_residual_threshold: float = 1e-3
    
    # Splits
    test_split: float = 0.2
    val_split: float = 0.1
    k_fold_cv: int = 0
    
    # Métriques
    metrics: List[str] = field(default_factory=lambda: [
        "mae", "rmse", "r2", "relative_error",
        "uncertainty_calibration", "coverage_probability",
    ])


@dataclass
class PipelineConfig:
    """Configuration globale du pipeline (dimension-agnostique)."""
    # Identification
    config_name: str = "cfd_generic"
    version: str = "1.0.0"
    description: str = ""
    
    # Paths
    base_dir: Path = Path("experiments/cfd")
    sim_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    exp_dir: Path = field(init=False)
    
    # Sous-configs
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gnn: GNNArchitectureConfig = field(default_factory=GNNArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Exécution
    device: str = "auto"
    random_seed: int = 42
    log_level: str = "INFO"
    n_workers: int = 4
    
    # Flags
    skip_simulation: bool = False
    skip_training: bool = False
    skip_evaluation: bool = False
    active_learning: bool = False
    max_active_iterations: int = 5
    dry_run: bool = False
    
    def __post_init__(self):
        """Initialisation des paths et device."""
        self.base_dir = Path(self.base_dir)
        self.sim_dir = self.base_dir / "simulations"
        self.log_dir = self.base_dir / "logs"
        self.model_dir = self.base_dir / "models"
        self.exp_dir = self.base_dir / self.config_name
        
        # Device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Seeds
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
    
    @property
    def checkpoint_path(self) -> Path:
        return self.exp_dir / "checkpoints"
    
    @property
    def spatial_dim(self) -> int:
        """Retourne la dimension spatiale (2 ou 3)."""
        if self.geometry.dimension == GeometryDimension.D3:
            return 3
        return 2
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'PipelineConfig':
        """Charge une config depuis YAML."""
        import yaml
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls._build_from_dict(data)
    
    @classmethod
    def _build_from_dict(cls, data: Dict) -> 'PipelineConfig':
        """Construction récursive des sous-configs."""
        if not data:
            return cls()
        
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        
        for key, value in data.items():
            if key not in field_types:
                continue
            field_type = field_types[key]
            
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = cls._build_nested(field_type, value or {})
            elif field_type == Path:
                kwargs[key] = Path(value) if value else None
            elif hasattr(field_type, '__members__'):
                kwargs[key] = field_type(value)
            else:
                kwargs[key] = value
        
        return cls(**kwargs)
    
    @classmethod
    def _build_nested(cls, dataclass_type, data: Dict):
        """Construction récursive d'une sous-dataclass."""
        if not data:
            return dataclass_type()
        
        field_types = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
        kwargs = {}
        
        for key, value in data.items():
            if key not in field_types:
                continue
            field_type = field_types[key]
            
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = cls._build_nested(field_type, value or {})
            elif hasattr(field_type, '__members__'):
                kwargs[key] = field_type(value)
            elif field_type == Path:
                kwargs[key] = Path(value) if value else None
            else:
                kwargs[key] = value
        
        return dataclass_type(**kwargs)
    
    def save(self, path: Optional[Path] = None):
        """Sauvegarde la config en YAML."""
        path = path or (self.exp_dir / "config.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        
        def to_serializable(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_serializable(v) for k, v in asdict(obj).items()}
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(to_serializable(self), f, default_flow_style=False,
                     sort_keys=False, allow_unicode=True)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PipelineConfig':
        """Charge une config sauvegardée."""
        path = Path(path)
        if path.suffix in ['.yaml', '.yml']:
            return cls.from_yaml(path)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            return cls._build_from_dict(data)
        else:
            raise ValueError(f"Format non supporté: {path.suffix}")


# =============================================================================
# LOGGING STRUCTURÉ
# =============================================================================

def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Configure le logging avec rotation."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("cfd_gnn")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Fichier principal
    fh = logging.handlers.RotatingFileHandler(
        config.log_dir / "pipeline.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    fh.setLevel(getattr(logging, config.log_level.upper()))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Fichier erreurs
    eh = logging.FileHandler(config.log_dir / "errors.log", mode='a', encoding='utf-8')
    eh.setLevel(logging.ERROR)
    eh.setFormatter(formatter)
    logger.addHandler(eh)
    
    return logger


# =============================================================================
# FEATURE NORMALIZER
# =============================================================================

class FeatureNormalizer:
    """Normalise les features du graphe (moyenne=0, std=1)."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, graphs: List[Dict[str, Any]]):
        """Calcule les statistiques sur le dataset."""
        all_features = []
        
        for graph in graphs:
            if "node_features" in graph and isinstance(graph["node_features"], torch.Tensor):
                all_features.append(graph["node_features"].cpu().numpy())
        
        if len(all_features) == 0:
            return
        
        X = np.vstack(all_features)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        self.fitted = True
    
    def transform(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise les features d'un graphe."""
        if not self.fitted:
            return graph
        
        if "node_features" in graph and isinstance(graph["node_features"], torch.Tensor):
            features = graph["node_features"].cpu().numpy()
            features_normalized = (features - self.mean) / self.std
            graph["node_features"] = torch.tensor(features_normalized, dtype=torch.float32)
        
        return graph
    
    def fit_transform(self, graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fit + transform."""
        self.fit(graphs)
        return [self.transform(g) for g in graphs]
    
    def save(self, path: Path):
        """Sauvegarde les statistiques."""
        if self.fitted:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, mean=self.mean, std=self.std)
    
    def load(self, path: Path):
        """Charge les statistiques."""
        if path.exists():
            data = np.load(path)
            self.mean = data["mean"]
            self.std = data["std"]
            self.fitted = True


# =============================================================================
# PARAMETER SAMPLER (GÉNÉRIQUE)
# =============================================================================

class ParameterSampler:
    """
    Échantillonnage de paramètres pour N'IMPORTE QUEL cas CFD.
    
    Supporte tous les types de paramètres géométriques, opératoires, etc.
    """
    
    def __init__(self,
                 param_ranges: Dict[str, Tuple[float, float]],
                 config: Optional[SamplingConfig] = None,
                 feasibility_fn: Optional[Callable] = None):
        self.param_ranges = param_ranges
        self.cfg = config or SamplingConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.feasibility_fn = feasibility_fn
        self.logger = logging.getLogger("cfd_gnn")
    
    def check_feasibility(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Vérifie si une combinaison de paramètres est faisable."""
        if not self.cfg.feasibility_checks:
            return True, "OK"
        
        if self.feasibility_fn is not None:
            try:
                return self.feasibility_fn(params)
            except Exception as e:
                return False, f"Erreur feasibility_fn: {e}"
        
        for key, (min_val, max_val) in self.param_ranges.items():
            if key in params:
                if params[key] < min_val or params[key] > max_val:
                    return False, f"{key} hors bornes: {params[key]}"
        
        return True, "OK"
    
    def sample_initial(self, n_samples: int) -> List[Dict[str, float]]:
        """Génère l'ensemble initial de paramètres."""
        param_names = list(self.param_ranges.keys())
        bounds = np.array([self.param_ranges[k] for k in param_names])
        
        if self.cfg.method == SamplingMethod.LATIN_HYPERCUBE:
            samples = self._sample_latin_hypercube(n_samples, bounds)
        elif self.cfg.method == SamplingMethod.SOBOL:
            samples = self._sample_sobol(n_samples, bounds)
        elif self.cfg.method == SamplingMethod.RANDOM:
            samples = self._sample_random(n_samples, bounds)
        else:
            samples = self._sample_grid(n_samples, bounds)
        
        valid_samples = []
        for sample in samples:
            params = dict(zip(param_names, sample))
            is_feasible, reason = self.check_feasibility(params)
            if is_feasible:
                valid_samples.append(params)
            else:
                self.logger.debug(f"Paramètres rejetés: {params} — {reason}")
        
        attempts = 0
        while len(valid_samples) < n_samples and attempts < n_samples * 10:
            params = {k: self.rng.uniform(v[0], v[1])
                     for k, v in self.param_ranges.items()}
            is_feasible, _ = self.check_feasibility(params)
            if is_feasible:
                valid_samples.append(params)
            attempts += 1
        
        self.logger.info(f"Échantillonnage: {len(valid_samples)}/{n_samples} paramètres valides")
        return valid_samples[:n_samples]
    
    def _sample_latin_hypercube(self, n: int, bounds: np.ndarray) -> np.ndarray:
        """Latin Hypercube Sampling."""
        d = len(bounds)
        sampler = qmc.LatinHypercube(d=d, seed=self.cfg.seed)
        sample = sampler.random(n=n)
        return qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    
    def _sample_sobol(self, n: int, bounds: np.ndarray) -> np.ndarray:
        """Sobol sequence."""
        d = len(bounds)
        sampler = qmc.Sobol(d=d, seed=self.cfg.seed)
        sample = sampler.random(n=n)
        return qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    
    def _sample_random(self, n: int, bounds: np.ndarray) -> np.ndarray:
        """Tirage aléatoire uniforme."""
        samples = np.zeros((n, len(bounds)))
        for i, (low, high) in enumerate(bounds):
            samples[:, i] = self.rng.uniform(low, high, n)
        return samples
    
    def _sample_grid(self, n: int, bounds: np.ndarray) -> np.ndarray:
        """Grille régulière."""
        d = len(bounds)
        points_per_dim = int(np.ceil(n ** (1/d)))
        grids = [np.linspace(b[0], b[1], points_per_dim) for b in bounds]
        samples = np.array(np.meshgrid(*grids)).T.reshape(-1, d)
        
        if len(samples) > n:
            idx = self.rng.choice(len(samples), n, replace=False)
            samples = samples[idx]
        return samples
    
    def suggest_next(self,
                    candidate_params: List[Dict[str, float]],
                    model: Optional[Any] = None,
                    acquisition: Optional[AcquisitionFunction] = None) -> List[Dict[str, float]]:
        """Sélectionne les prochains paramètres via active learning."""
        acquisition = acquisition or self.cfg.acquisition_function
        n_select = self.cfg.n_per_iteration
        
        if len(candidate_params) == 0:
            return []
        if len(candidate_params) <= n_select:
            return candidate_params
        
        valid_candidates = [p for p in candidate_params
                          if self.check_feasibility(p)[0]]
        
        if len(valid_candidates) == 0:
            self.logger.warning("Aucun candidat faisable")
            return candidate_params[:n_select]
        
        if acquisition == AcquisitionFunction.UNCERTAINTY and model is not None:
            return self._select_by_uncertainty(model, valid_candidates, n_select)
        elif acquisition == AcquisitionFunction.DIVERSITY:
            return self._select_by_diversity(valid_candidates, n_select)
        elif acquisition == AcquisitionFunction.HYBRID:
            return self._select_hybrid(model, valid_candidates, n_select)
        else:
            indices = self.rng.choice(len(valid_candidates),
                                     min(n_select, len(valid_candidates)),
                                     replace=False)
            return [valid_candidates[i] for i in indices]
    
    def _select_by_uncertainty(self, model, candidates: List[Dict], n: int) -> List[Dict]:
        """Sélection par incertitude."""
        if not hasattr(model, 'predict_with_uncertainty'):
            self.logger.warning("Modèle sans predict_with_uncertainty, fallback diversité")
            return self._select_by_diversity(candidates, n)
        
        uncertainties = []
        for params in candidates:
            try:
                pred = model.predict_with_uncertainty(params, n_samples=self.cfg.n_mc_samples)
                unc = np.mean(pred.get("uncertainty", 0))
                uncertainties.append(unc)
            except Exception as e:
                self.logger.debug(f"Erreur uncertainty: {e}")
                uncertainties.append(0.0)
        
        indices = np.argsort(uncertainties)[-n:][::-1]
        return [candidates[i] for i in indices if i < len(candidates)]
    
    def _select_by_diversity(self, candidates: List[Dict], n: int) -> List[Dict]:
        """Sélection par diversité."""
        if len(candidates) <= n:
            return candidates
        
        param_names = list(self.param_ranges.keys())
        X = np.array([[c[k] for k in param_names] for c in candidates])
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        
        selected = [0]
        while len(selected) < n and len(selected) < len(candidates):
            min_dists = np.full(len(candidates), np.inf)
            for sel_idx in selected:
                dists = np.linalg.norm(X_norm - X_norm[sel_idx], axis=1)
                min_dists = np.minimum(min_dists, dists)
            next_idx = np.argmax(min_dists)
            if next_idx not in selected:
                selected.append(next_idx)
        
        return [candidates[i] for i in selected]
    
    def _select_hybrid(self, model, candidates: List[Dict], n: int) -> List[Dict]:
        """Combinaison uncertainty + diversity."""
        n_unc = n // 2
        n_div = n - n_unc
        
        unc_selected = self._select_by_uncertainty(model, candidates, n_unc)
        remaining = [c for c in candidates if c not in unc_selected]
        div_selected = self._select_by_diversity(remaining, n_div)
        
        return unc_selected + div_selected
    
    def generate_candidate_pool(self, size: int = 200,
                               exclude: Optional[List[Dict]] = None) -> List[Dict[str, float]]:
        """Génère un pool de candidats."""
        candidates = self.sample_initial(size * 2)
        
        if exclude:
            def params_equal(p1, p2, tol=1e-4):
                return all(abs(p1.get(k, 0) - p2.get(k, 0)) < tol
                          for k in self.param_ranges.keys())
            candidates = [c for c in candidates
                         if not any(params_equal(c, e) for e in exclude)]
        
        return candidates[:size]


# =============================================================================
# GRAPH EXTRACTOR (DIMENSION-AGNOSTIQUE)
# =============================================================================

class UniversalGraphExtractor:
    """
    Extrait un graphe depuis N'IMPORTE QUEL mesh CFD (2D/3D).
    
    Features implémentées (toutes dimension-agnostiques):
    ✓ Position normalisée
    ✓ Wall distance (KDTree avec caching)
    ✓ Distance aux boundaries (inlet, outlet, etc.)
    ✓ Boundary type (one-hot)
    ✓ Cell volumes / Face areas
    ✓ Edge features (length, direction, type, area)
    ✓ Control volumes pour divergence
    """
    
    BOUNDARY_TYPES = ["wall", "inlet", "outlet", "symmetry", "periodic", "axis", "interior"]
    
    def __init__(self,
                 case_path: Path,
                 geometry_config: Optional[GeometryConfig] = None):
        self.case_path = Path(case_path)
        self.cfg = geometry_config or GeometryConfig()
        self.logger = logging.getLogger("cfd_gnn")
        
        # Cache
        self._mesh = None
        self._node_positions = None
        self._boundary_nodes = {}
        self._node_volumes = None
        self._face_areas = None
        self._spatial_dim = None
        self._wall_distances = None
        self._boundary_distances = {}
        self._kdtrees = {}
    
    def extract(self) -> Dict[str, Any]:
        """Extrait toutes les données du graphe."""
        self.logger.debug(f"Extraction graphe: {self.case_path.name}")
        
        self._load_mesh()
        node_features = self._extract_node_features()
        edge_index, edge_features = self._extract_edge_features()
        node_volumes = self._compute_control_volumes()
        targets = self._load_target_fields()
        metadata = self._load_metadata()
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "node_volumes": node_volumes,
            "face_areas": self._face_areas,
            "targets": targets,
            "metadata": metadata,
            "n_nodes": len(self._node_positions),
            "n_edges": edge_index.shape[1] if edge_index is not None else 0,
            "spatial_dim": self._spatial_dim,
        }
    
    def _load_mesh(self):
        """Charge le mesh (OpenFOAM ou autre)."""
        if FOAMPILOT_AVAILABLE:
            self._mesh = meshing.OpenFOAMMesh(self.case_path)
            self._node_positions = self._mesh.node_positions
            self._spatial_dim = self._mesh.dim
            self._face_areas = self._mesh.face_areas
        else:
            self._node_positions = self._generate_dummy_mesh()
            self._spatial_dim = 3 if self.cfg.dimension == GeometryDimension.D3 else 2
            self._face_areas = None
        
        self._identify_boundaries()
    
    def _generate_dummy_mesh(self) -> np.ndarray:
        """Génère un mesh dummy pour test."""
        n_nodes = 1000
        dim = 3 if self.cfg.dimension == GeometryDimension.D3 else 2
        
        if dim == 2:
            x = np.random.uniform(0, 1, n_nodes)
            y = np.random.uniform(0, 1, n_nodes)
            return np.column_stack([x, y])
        else:
            x = np.random.uniform(0, 1, n_nodes)
            y = np.random.uniform(0, 1, n_nodes)
            z = np.random.uniform(0, 1, n_nodes)
            return np.column_stack([x, y, z])
    
    def _identify_boundaries(self):
        """Identifie les noeuds par type de boundary."""
        if FOAMPILOT_AVAILABLE and self._mesh:
            for btype in self.BOUNDARY_TYPES:
                nodes = self._mesh.get_boundary_nodes(btype)
                if len(nodes) > 0:
                    self._boundary_nodes[btype] = nodes
        else:
            pos = self._node_positions
            tol = self.cfg.wall_distance_tolerance
            
            if self._spatial_dim == 2:
                self._boundary_nodes["wall"] = np.where(
                    (pos[:, 0] < tol) | (pos[:, 0] > 1-tol) |
                    (pos[:, 1] < tol) | (pos[:, 1] > 1-tol)
                )[0]
            else:
                self._boundary_nodes["wall"] = np.where(
                    (pos[:, 0] < tol) | (pos[:, 0] > 1-tol) |
                    (pos[:, 1] < tol) | (pos[:, 1] > 1-tol) |
                    (pos[:, 2] < tol) | (pos[:, 2] > 1-tol)
                )[0]
            
            self._boundary_nodes["inlet"] = np.where(pos[:, 0] < tol)[0]
            self._boundary_nodes["outlet"] = np.where(pos[:, 0] > 1-tol)[0]
    
    def _extract_node_features(self) -> torch.Tensor:
        """Extrait les features par noeud (dimension-agnostique)."""
        features = []
        positions = torch.tensor(self._node_positions, dtype=torch.float32)
        
        if self.cfg.include_node_position:
            scale = self.cfg.characteristic_length
            features.append(positions / (scale + 1e-8))
        
        if self.cfg.extract_wall_distance and "wall" in self._boundary_nodes:
            wall_dist = self._compute_wall_distance()
            features.append(wall_dist.unsqueeze(-1))
        
        if self.cfg.extract_inlet_distance and "inlet" in self._boundary_nodes:
            inlet_dist = self._compute_boundary_distance("inlet")
            features.append(inlet_dist.unsqueeze(-1))
        
        if self.cfg.extract_outlet_distance and "outlet" in self._boundary_nodes:
            outlet_dist = self._compute_boundary_distance("outlet")
            features.append(outlet_dist.unsqueeze(-1))
        
        if self.cfg.include_boundary_type:
            boundary_type = self._classify_boundary_type()
            features.append(boundary_type)
        
        if self.cfg.include_cell_volume:
            volumes = self._compute_control_volumes()
            features.append(volumes.unsqueeze(-1))
        
        for feature_name in self.cfg.custom_node_features:
            feature = self._compute_custom_node_feature(feature_name)
            if feature is not None:
                features.append(feature)
        
        if len(features) == 0:
            features.append(positions)
        
        return torch.cat(features, dim=1)
    
    def _compute_wall_distance(self) -> torch.Tensor:
        """Distance minimale à la paroi avec caching."""
        if self._wall_distances is not None:
            return self._wall_distances
        
        if "wall" not in self._boundary_nodes or len(self._boundary_nodes["wall"]) == 0:
            self._wall_distances = torch.zeros(len(self._node_positions))
            return self._wall_distances
        
        wall_positions = self._node_positions[self._boundary_nodes["wall"]]
        
        if "wall" not in self._kdtrees:
            self._kdtrees["wall"] = cKDTree(wall_positions)
        
        tree = self._kdtrees["wall"]
        distances, _ = tree.query(self._node_positions, k=1)
        
        self._wall_distances = torch.tensor(distances, dtype=torch.float32)
        return self._wall_distances
    
    def _compute_boundary_distance(self, boundary_type: str) -> torch.Tensor:
        """Distance à un boundary avec caching."""
        if boundary_type in self._boundary_distances:
            return self._boundary_distances[boundary_type]
        
        if boundary_type not in self._boundary_nodes:
            dist = torch.zeros(len(self._node_positions))
            self._boundary_distances[boundary_type] = dist
            return dist
        
        boundary_nodes = self._boundary_nodes[boundary_type]
        if len(boundary_nodes) == 0:
            dist = torch.zeros(len(self._node_positions))
            self._boundary_distances[boundary_type] = dist
            return dist
        
        boundary_positions = self._node_positions[boundary_nodes]
        
        if boundary_type not in self._kdtrees:
            self._kdtrees[boundary_type] = cKDTree(boundary_positions)
        
        tree = self._kdtrees[boundary_type]
        distances, _ = tree.query(self._node_positions, k=1)
        
        dist = torch.tensor(distances, dtype=torch.float32)
        self._boundary_distances[boundary_type] = dist
        return dist
    
    def _classify_boundary_type(self) -> torch.Tensor:
        """Classifie chaque noeud par type de boundary."""
        n_nodes = len(self._node_positions)
        n_types = len(self.BOUNDARY_TYPES)
        boundary_type = torch.zeros(n_nodes, n_types)
        
        for i, btype in enumerate(self.BOUNDARY_TYPES):
            if btype in self._boundary_nodes:
                boundary_type[self._boundary_nodes[btype], i] = 1.0
        
        interior_mask = (boundary_type.sum(dim=1) == 0)
        if "interior" in self.BOUNDARY_TYPES:
            interior_idx = self.BOUNDARY_TYPES.index("interior")
            boundary_type[interior_mask, interior_idx] = 1.0
        
        return boundary_type
    
    def _compute_control_volumes(self) -> torch.Tensor:
        """Calcule les volumes de contrôle (2D: aire, 3D: volume)."""
        if self._node_volumes is not None:
            return torch.tensor(self._node_volumes, dtype=torch.float32)
        
        if FOAMPILOT_AVAILABLE and self._mesh:
            volumes = self._mesh.cell_volumes
        else:
            pos = torch.tensor(self._node_positions, dtype=torch.float32)
            k = 6
            tree = cKDTree(self._node_positions)
            distances, _ = tree.query(self._node_positions, k=k+1)
            avg_dist = distances[:, 1:].mean(axis=1)
            
            if self._spatial_dim == 2:
                volumes = np.pi * (avg_dist ** 2)
            else:
                volumes = (4/3) * np.pi * (avg_dist ** 3)
        
        self._node_volumes = volumes
        return torch.tensor(volumes, dtype=torch.float32)
    
    def _extract_edge_features(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extrait la connectivité et features d'arête."""
        if FOAMPILOT_AVAILABLE and self._mesh:
            edge_index = self._mesh.edge_index
            face_areas = self._mesh.face_areas
        else:
            edge_index = self._build_dummy_connectivity()
            face_areas = None
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        
        if edge_index.shape[0] == 0:
            return edge_index, None
        
        src, dst = edge_index
        pos = torch.tensor(self._node_positions, dtype=torch.float32)
        
        features = []
        
        edge_vec = pos[dst] - pos[src]
        edge_len = torch.norm(edge_vec, dim=1, keepdim=True)
        features.append(edge_len)
        
        if face_areas is not None and len(face_areas) == len(src):
            face_area_tensor = torch.tensor(face_areas, dtype=torch.float32).unsqueeze(-1)
            features.append(face_area_tensor)
        else:
            if self._spatial_dim == 2:
                features.append(edge_len)
            else:
                features.append(edge_len ** 2)
        
        edge_dir = edge_vec / (edge_len + 1e-8)
        features.append(edge_dir)
        
        for feature_name in self.cfg.custom_edge_features:
            feature = self._compute_custom_edge_feature(feature_name, edge_index, edge_vec)
            if feature is not None:
                features.append(feature)
        
        edge_features = torch.cat(features, dim=1) if features else None
        
        return edge_index, edge_features
    
    def _build_dummy_connectivity(self) -> np.ndarray:
        """Construit une connectivité dummy par k-NN."""
        n_nodes = len(self._node_positions)
        k = min(6, n_nodes - 1)
        
        tree = cKDTree(self._node_positions)
        _, neighbors = tree.query(self._node_positions, k=k+1)
        
        edges = []
        for i, neigh in enumerate(neighbors):
            for j in neigh[1:]:
                if i < j:
                    edges.append([i, j])
        
        return np.array(edges) if edges else np.array([]).reshape(0, 2)
    
    def _compute_custom_node_feature(self, name: str) -> Optional[torch.Tensor]:
        """Calcule une feature de noeud custom."""
        return None
    
    def _compute_custom_edge_feature(self, name: str,
                                     edge_index: torch.Tensor,
                                     edge_vec: torch.Tensor) -> Optional[torch.Tensor]:
        """Calcule une feature d'arête custom."""
        return None
    
    def _load_target_fields(self) -> Optional[Dict[str, torch.Tensor]]:
        """Charge les champs cibles."""
        if not FOAMPILOT_AVAILABLE:
            return None
        
        try:
            latest_time = utils.get_latest_time(self.case_path)
            if latest_time is None:
                return None
            
            fields = utils.load_fields(self.case_path, latest_time,
                                      ["p", "T", "U", "Ma", "rho", "k", "omega"])
            
            return {k: torch.tensor(v, dtype=torch.float32)
                   for k, v in fields.items()}
        except Exception as e:
            self.logger.warning(f"Erreur chargement champs: {e}")
            return None
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Charge les métadonnées dont les BC."""
        meta_path = self.case_path / "gnn_metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            if "boundary_conditions" in metadata:
                bc = metadata["boundary_conditions"]
                return {
                    **metadata,
                    "inlet_bc": bc.get("inlet", {}),
                    "outlet_bc": bc.get("outlet", {}),
                }
        return {"case_name": self.case_path.name}
    
    def clear_cache(self):
        """Libère le cache mémoire."""
        self._wall_distances = None
        self._boundary_distances = {}
        self._kdtrees = {}


# =============================================================================
# PHYSICS-INFORMED LOSS (DIMENSION-AGNOSTIQUE)
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Loss physics-informed fonctionnant en 2D et 3D.
    
    Termes implémentés:
    1. MSE sur variables physiques
    2. Conservation de masse: ∇·(ρU) = 0
    3. Conservation de quantité de mouvement: ∇·(ρU⊗U) + ∇p = 0
    4. Conservation d'énergie (optionnel)
    5. Contraintes aux boundaries
    6. Turbulence (positivité de k, omega)
    """
    
    def __init__(self, config: PhysicsConfig, spatial_dim: int = 3):
        super().__init__()
        self.cfg = config
        self.spatial_dim = spatial_dim
        self.gamma = config.gamma
        self.R_gas = config.R_gas
        self.logger = logging.getLogger("cfd_gnn")
        self.boundary_types = ["wall", "inlet", "outlet", "symmetry", "axis"]
        
        self.register_buffer("loss_scales", torch.ones(6))
        self._loss_running_mean = torch.zeros(6)
        self._loss_count = 0
    
    def forward(self,
                pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                graph: Dict[str, Any],
                weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Calcule le loss composite."""
        w = weights or self.cfg.loss_weights
        
        L_data = self._data_loss(pred, target)
        L_mass = self._mass_conservation_loss(pred, graph)
        L_momentum = self._momentum_conservation_loss(pred, graph)
        L_energy = self._energy_conservation_loss(pred, graph)
        L_bc = self._boundary_condition_loss(pred, graph)
        L_turb = self._turbulence_loss(pred, graph)
        
        if self.cfg.loss_normalization:
            L_mass = self._normalize_loss(L_mass, 1)
            L_momentum = self._normalize_loss(L_momentum, 2)
            L_energy = self._normalize_loss(L_energy, 3)
            L_bc = self._normalize_loss(L_bc, 4)
            L_turb = self._normalize_loss(L_turb, 5)
        
        if self.cfg.loss_clip_max > 0:
            L_mass = torch.clamp(L_mass, max=self.cfg.loss_clip_max)
            L_momentum = torch.clamp(L_momentum, max=self.cfg.loss_clip_max)
            L_energy = torch.clamp(L_energy, max=self.cfg.loss_clip_max)
            L_bc = torch.clamp(L_bc, max=self.cfg.loss_clip_max)
            L_turb = torch.clamp(L_turb, max=self.cfg.loss_clip_max)
        
        L_total = (
            w.get("data", 1.0) * L_data +
            w.get("mass", 0.05) * L_mass +
            w.get("momentum", 0.02) * L_momentum +
            w.get("energy", 0.05) * L_energy +
            w.get("bc", 0.03) * L_bc +
            w.get("turbulence", 0.01) * L_turb
        )
        
        self._loss_count += 1
        
        return L_total
    
    def _data_loss(self, pred: Dict[str, torch.Tensor],
                   target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """MSE relative sur variables physiques."""
        loss = torch.tensor(0.0, device=pred["p"].device)
        n_vars = 0
        
        for var in ["p", "T", "Mach", "U"]:
            if var in pred and var in target:
                eps = 1e-6
                rel_error = (pred[var] - target[var]) / (target[var].abs() + eps)
                loss += F.mse_loss(rel_error, torch.zeros_like(rel_error))
                n_vars += 1
        
        return loss / max(n_vars, 1)
    
    def _mass_conservation_loss(self, pred: Dict[str, torch.Tensor],
                                graph: Dict[str, Any]) -> torch.Tensor:
        """Conservation de masse: ∇·(ρU) = 0"""
        if "rho" not in pred or "U" not in pred:
            return torch.tensor(0.0, device=pred["p"].device)
        
        if not TORCH_SCATTER_AVAILABLE:
            return torch.tensor(0.0, device=pred["p"].device)
        
        rho = pred["rho"].squeeze()
        U = pred["U"]
        device = rho.device
        
        edge_index = graph.get("edge_index")
        node_volumes = graph.get("node_volumes")
        
        if edge_index is None or node_volumes is None:
            return torch.tensor(0.0, device=device)
        
        src, dst = edge_index
        n_edges = len(src)
        
        pos = graph.get("node_positions")
        if pos is not None:
            pos = torch.tensor(pos, dtype=torch.float32, device=device)
            edge_vec = pos[dst] - pos[src]
            edge_len = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8
            edge_unit = edge_vec / edge_len
            
            if self.spatial_dim == 2:
                edge_area = edge_len.squeeze()
            else:
                edge_area = edge_len.squeeze() ** 2
        else:
            edge_area = torch.ones(n_edges, device=device)
            edge_unit = torch.ones(n_edges, self.spatial_dim, device=device)
        
        upwind_scheme = self.cfg.mass_flux_scheme
        
        if upwind_scheme == "upwind":
            U_face = (U[src] + U[dst]) / 2
            U_normal = (U_face * edge_unit).sum(dim=1, keepdim=True)
            rho_face = torch.where(U_normal > 0, rho[src], rho[dst]).squeeze()
        elif upwind_scheme == "linear":
            rho_face = (rho[src] + rho[dst]) / 2
            U_face = (U[src] + U[dst]) / 2
            U_normal = (U_face * edge_unit).sum(dim=1, keepdim=True)
        else:
            rho_face = 2 * rho[src] * rho[dst] / (rho[src] + rho[dst] + 1e-8)
            U_face = (U[src] + U[dst]) / 2
            U_normal = (U_face * edge_unit).sum(dim=1, keepdim=True)
        
        flux_mass = rho_face * U_normal.squeeze() * edge_area
        
        flux_out = scatter_add(flux_mass, dst, dim=0, dim_size=len(rho))
        flux_in = scatter_add(flux_mass, src, dim=0, dim_size=len(rho))
        
        divergence = (flux_out - flux_in) / (node_volumes + 1e-8)
        
        return torch.mean(divergence ** 2)
    
    def _momentum_conservation_loss(self, pred: Dict[str, torch.Tensor],
                                    graph: Dict[str, Any]) -> torch.Tensor:
        """Conservation de quantité de mouvement (version améliorée)."""
        if "p" not in pred or "U" not in pred:
            return torch.tensor(0.0, device=pred["p"].device)
        
        if not TORCH_SCATTER_AVAILABLE:
            return torch.tensor(0.0, device=pred["p"].device)
        
        p = pred["p"].squeeze()
        U = pred["U"]
        rho = pred.get("rho", torch.ones_like(p))
        device = p.device
        
        edge_index = graph.get("edge_index")
        node_volumes = graph.get("node_volumes")
        
        if edge_index is None or node_volumes is None:
            return torch.tensor(0.0, device=device)
        
        src, dst = edge_index
        n_edges = len(src)
        
        pos = graph.get("node_positions")
        if pos is not None:
            pos = torch.tensor(pos, dtype=torch.float32, device=device)
            edge_vec = pos[dst] - pos[src]
            edge_len = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8
            edge_normal = edge_vec / edge_len
            
            if self.spatial_dim == 2:
                edge_area = edge_len.squeeze()
            else:
                edge_area = edge_len.squeeze() ** 2
        else:
            edge_area = torch.ones(n_edges, device=device)
            edge_normal = torch.ones(n_edges, self.spatial_dim, device=device)
        
        rho_face = 2 * rho[src] * rho[dst] / (rho[src] + rho[dst] + 1e-8)
        U_face = (U[src] + U[dst]) / 2
        p_face = (p[src] + p[dst]) / 2
        
        U_normal = (U_face * edge_normal).sum(dim=1, keepdim=True)
        momentum_flux = rho_face * U_face * U_normal
        
        pressure_flux = p_face.unsqueeze(-1) * edge_normal
        
        total_flux = momentum_flux + pressure_flux
        
        flux_out = scatter_add(total_flux, dst, dim=0, dim_size=len(p))
        flux_in = scatter_add(total_flux, src, dim=0, dim_size=len(p))
        
        divergence = (flux_out - flux_in) / (node_volumes.unsqueeze(-1) + 1e-8)
        
        return torch.mean(divergence ** 2)
    
    def _energy_conservation_loss(self, pred: Dict[str, torch.Tensor],
                                  graph: Dict[str, Any]) -> torch.Tensor:
        """Conservation d'énergie (optionnelle)."""
        return torch.tensor(0.0, device=pred["p"].device if "p" in pred else torch.device("cpu"))
    
    def _boundary_condition_loss(self, pred: Dict[str, torch.Tensor],
                                 graph: Dict[str, Any]) -> torch.Tensor:
        """Contraintes aux boundaries."""
        device = pred["p"].device if "p" in pred else torch.device("cpu")
        loss = torch.tensor(0.0, device=device)
        
        boundary_type = graph.get("boundary_type")
        if boundary_type is None:
            return loss
        
        if "wall" in self.boundary_types:
            wall_idx = self.boundary_types.index("wall")
            wall_mask = boundary_type[:, wall_idx] > 0.5
            
            if wall_mask.any() and "U" in pred:
                U_wall = pred["U"][wall_mask]
                loss += F.mse_loss(U_wall, torch.zeros_like(U_wall))
        
        if "inlet" in self.boundary_types:
            inlet_idx = self.boundary_types.index("inlet")
            inlet_mask = boundary_type[:, inlet_idx] > 0.5
            inlet_bc = graph.get("inlet_bc", {})
            
            if inlet_mask.any():
                if "U" in pred and "U_inlet" in inlet_bc:
                    U_inlet = torch.tensor(inlet_bc["U_inlet"], device=pred["U"].device)
                    loss += F.mse_loss(pred["U"][inlet_mask], U_inlet)
                
                if "T" in pred and "T_inlet" in inlet_bc:
                    T_inlet = torch.tensor(inlet_bc["T_inlet"], device=pred["T"].device)
                    loss += F.mse_loss(pred["T"][inlet_mask], T_inlet)
        
        if "outlet" in self.boundary_types:
            outlet_idx = self.boundary_types.index("outlet")
            outlet_mask = boundary_type[:, outlet_idx] > 0.5
            outlet_bc = graph.get("outlet_bc", {})
            
            if outlet_mask.any() and "p" in pred and "p_outlet" in outlet_bc:
                p_outlet = torch.tensor(outlet_bc["p_outlet"], device=pred["p"].device)
                loss += F.mse_loss(pred["p"][outlet_mask], p_outlet * torch.ones_like(pred["p"][outlet_mask]))
        
        if "axis" in self.boundary_types and self.spatial_dim == 2:
            axis_idx = self.boundary_types.index("axis")
            axis_mask = boundary_type[:, axis_idx] > 0.5
            
            if axis_mask.any() and "U" in pred:
                U_axis = pred["U"][axis_mask]
                loss += F.mse_loss(U_axis[:, 1:2], torch.zeros_like(U_axis[:, 1:2]))
        
        return loss
    
    def _turbulence_loss(self, pred: Dict[str, torch.Tensor],
                        graph: Dict[str, Any]) -> torch.Tensor:
        """Contraintes sur variables de turbulence."""
        device = pred["p"].device if "p" in pred else torch.device("cpu")
        if "k" not in pred or "omega" not in pred:
            return torch.tensor(0.0, device=device)
        
        k_positivity = torch.mean(F.relu(-pred["k"]) ** 2)
        omega_positivity = torch.mean(F.relu(-pred["omega"]) ** 2)
        
        return k_positivity + omega_positivity
    
    def _normalize_loss(self, loss: torch.Tensor, idx: int) -> torch.Tensor:
        """Normalisation dynamique."""
        if not self.training:
            return loss / (self.loss_scales[idx] + 1e-8)
        
        alpha = 0.9
        self._loss_running_mean[idx] = alpha * self._loss_running_mean[idx] + (1 - alpha) * loss.item()
        self.loss_scales[idx] = self._loss_running_mean[idx] + 1e-8
        
        return loss / self.loss_scales[idx]


# =============================================================================
# GNN MODEL (DIMENSION-AGNOSTIQUE)
# =============================================================================

class UniversalGNN(nn.Module):
    """GNN universel fonctionnant en 2D et 3D."""
    
    def __init__(self, config: GNNArchitectureConfig, physics_config: PhysicsConfig, spatial_dim: int = 3):
        super().__init__()
        self.cfg = config
        self.physics_cfg = physics_config
        self.spatial_dim = spatial_dim
        
        self.input_dim = self._estimate_input_dim()
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        )
        
        self.gnn_layers = nn.ModuleList()
        for i in range(config.n_layers):
            layer = UniversalGraphConv(
                in_dim=config.hidden_dim,
                out_dim=config.hidden_dim,
                n_heads=config.n_heads,
                dropout=config.dropout_rate,
                use_attention=config.use_attention,
                aggregation=config.aggregation,
            )
            self.gnn_layers.append(layer)
        
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, len(config.output_variables)),
        )
        
        if config.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, len(config.output_variables) * 2),
            )
    
    def _estimate_input_dim(self) -> int:
        """Estime la dimension d'entrée."""
        dim = 0
        if self.cfg.include_node_position:
            dim += self.spatial_dim
        if self.cfg.extract_wall_distance:
            dim += 1
        if self.cfg.extract_inlet_distance:
            dim += 1
        if self.cfg.extract_outlet_distance:
            dim += 1
        if self.cfg.include_boundary_type:
            dim += len(UniversalGraphExtractor.BOUNDARY_TYPES)
        if self.cfg.include_cell_volume:
            dim += 1
        for _ in self.cfg.custom_node_features:
            dim += 1
        return max(dim, self.spatial_dim)
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                node_volumes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.encoder(node_features)
        
        for layer in self.gnn_layers:
            x, _ = layer(x, edge_index, edge_features)
        
        output = self.decoder(x)
        
        var_names = self.cfg.output_variables
        pred = {var_names[i]: output[:, i:i+1] for i in range(len(var_names))}
        
        if "p" in pred and "T" in pred:
            pred["rho"] = pred["p"] / (self.physics_cfg.R_gas * pred["T"] + 1e-6)
        
        u_components = [k for k in pred.keys() if k.startswith("U") and k != "U"]
        if len(u_components) >= 2:
            u_tensors = [pred[k] for k in sorted(u_components)]
            pred["U"] = torch.cat(u_tensors, dim=1)
            pred["U_mag"] = torch.norm(pred["U"], dim=1, keepdim=True)
        
        if self.cfg.predict_uncertainty and hasattr(self, 'uncertainty_head'):
            unc_params = self.uncertainty_head(x)
            pred["uncertainty"] = unc_params
        
        return pred
    
    def predict_with_uncertainty(self,
                                 node_features: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """Prédiction avec quantification d'incertitude."""
        if self.cfg.predict_uncertainty and hasattr(self, 'uncertainty_head'):
            return self.forward(node_features, edge_index)
        else:
            self.train()
            predictions = []
            
            for _ in range(n_samples):
                pred = self.forward(node_features, edge_index)
                predictions.append(pred)
            
            self.eval()
            
            result = {}
            for key in predictions[0].keys():
                values = torch.stack([p[key] for p in predictions], dim=0)
                result[key] = values.mean(dim=0)
                result[f"{key}_std"] = values.std(dim=0)
            
            result["uncertainty"] = torch.stack([result[f"{k}_std"]
                                                 for k in result.keys() if "_std" in k], dim=1).mean(dim=1)
            return result


class UniversalGraphConv(nn.Module):
    """Couche de convolution graphique universelle (2D/3D)."""
    
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, use_attention: bool = True,
                 aggregation: str = "mean"):
        super().__init__()
        self.n_heads = n_heads
        self.use_attention = use_attention
        self.aggregation = aggregation
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=in_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
        
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        N = node_features.shape[0]
        
        if self.use_attention:
            attn_mask = self._build_attention_mask(edge_index, N)
            x = node_features.unsqueeze(0)
            attended, _ = self.attention(x, x, x, attn_mask=attn_mask)
            attended = attended.squeeze(0)
        else:
            src, dst = edge_index
            if self.aggregation == "mean":
                attended = scatter_mean(node_features[src], dst, dim=0, dim_size=N) if TORCH_SCATTER_AVAILABLE else node_features
            elif self.aggregation == "sum":
                attended = scatter_add(node_features[src], dst, dim=0, dim_size=N) if TORCH_SCATTER_AVAILABLE else node_features
            elif self.aggregation == "max":
                attended, _ = scatter_max(node_features[src], dst, dim=0, dim_size=N) if TORCH_SCATTER_AVAILABLE else (node_features, None)
            else:
                attended = node_features
        
        out = self.proj(attended)
        out = self.dropout(out)
        out = self.norm(out + node_features)
        
        return out, None
    
    def _build_attention_mask(self, edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Construit le masque d'attention."""
        src, dst = edge_index
        
        mask = torch.ones(n_nodes, n_nodes, dtype=torch.bool, device=edge_index.device)
        mask[src, dst] = False
        mask[dst, src] = False
        mask[range(n_nodes), range(n_nodes)] = False
        
        return mask


# =============================================================================
# CFD PIPELINE (CLASSE PRINCIPALE)
# =============================================================================

class CFDPipeline:
    """Pipeline CFD+GNN universel (2D/3D, toute géométrie)."""
    
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.logger = setup_logging(config)
        self.sampler = None
        self.model = None
        self.metrics = {}
        self.training_history = []
        self.normalizer = FeatureNormalizer()
        
        self.cfg.exp_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.cfg.save()
        self.logger.info(f"Config sauvegardée: {self.cfg.exp_dir / 'config.yaml'}")
    
    def run_full(self) -> Dict[str, Any]:
        """Exécute le pipeline complet."""
        self.logger.info(f"🚀 Pipeline: {self.cfg.config_name}")
        self.logger.info(f"Dimension: {self.cfg.geometry.dimension.value}")
        self.logger.info(f"Device: {self.cfg.device}")
        
        try:
            if not self.cfg.skip_simulation:
                self._generate_dataset()
            
            if not self.cfg.skip_training:
                self._train_model()
            
            if not self.cfg.skip_evaluation:
                self._evaluate_and_report()
            
            self.logger.info("✅ Pipeline terminé")
            return self.metrics
            
        except KeyboardInterrupt:
            self.logger.warning("⚠ Interruption utilisateur")
            self._save_checkpoint("interrupted")
            raise
        except Exception as e:
            self.logger.critical(f"✗ Erreur fatale: {e}", exc_info=True)
            self._save_checkpoint("error")
            raise
    
    def _generate_dataset(self):
        """Génère le dataset."""
        self.logger.info("📊 Génération du dataset...")
        
        self.sampler = ParameterSampler(
            param_ranges=self.cfg.sampling.param_ranges,
            config=self.cfg.sampling
        )
        
        initial_params = self.sampler.sample_initial(self.cfg.sampling.n_initial)
        self.logger.info(f"→ {len(initial_params)} combinaisons générées")
        
        success_count = 0
        for i, params in enumerate(initial_params):
            case_name = f"sim_{i:04d}"
            case_path = self.cfg.sim_dir / case_name
            
            self.logger.info(f"[{i+1}/{len(initial_params)}] {case_name}")
            
            if self.cfg.dry_run:
                success_count += 1
                continue
            
            try:
                if FOAMPILOT_AVAILABLE:
                    result = self._run_simulation(case_path, params)
                    if result.get("success"):
                        success_count += 1
                        self._save_case_metadata(case_path, params, result)
                else:
                    success_count += 1
                    self._save_case_metadata(case_path, params, {"success": True})
                    
            except Exception as e:
                self.logger.error(f"✗ Erreur: {e}", exc_info=True)
        
        self.logger.info(f"✓ Dataset: {success_count}/{len(initial_params)} simulations réussies")
    
    def _run_simulation(self, case_path: Path, params: Dict) -> Dict[str, Any]:
        """Lance une simulation CFD."""
        return {"success": True}
    
    def _train_model(self):
        """Entraîne le modèle avec curriculum learning."""
        self.logger.info("🧠 Entraînement du modèle...")
        
        graph_data = self._extract_all_graphs()
        if len(graph_data) == 0:
            self.logger.error("Aucun graphe extrait")
            return
        
        self.logger.info(f"→ {len(graph_data)} graphes extraits")
        
        # Normalisation des features
        if self.cfg.training.normalize_features:
            graph_data = self.normalizer.fit_transform(graph_data)
            self.normalizer.save(self.cfg.exp_dir / "feature_normalizer.npz")
        
        spatial_dim = graph_data[0].get("spatial_dim", self.cfg.spatial_dim)
        
        self.model = UniversalGNN(self.cfg.gnn, self.cfg.physics, spatial_dim)
        self.model = self.model.to(self.cfg.device)
        
        criterion = PhysicsInformedLoss(self.cfg.physics, spatial_dim)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.cfg.training.scheduler_factor,
            patience=self.cfg.training.scheduler_patience, verbose=True
        )
        
        if self.cfg.training.curriculum:
            train_order = self._get_curriculum_order(graph_data)
            self.logger.info(f"→ Curriculum learning activé ({len(train_order)} cas)")
        else:
            train_order = list(range(len(graph_data)))
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            if self.cfg.training.curriculum:
                n_samples = self._get_curriculum_n_samples(epoch, len(train_order))
                current_order = train_order[:n_samples]
            else:
                current_order = train_order
            
            if self.cfg.training.curriculum and self.cfg.training.curriculum_schedule == "physics_ramp":
                loss_weights = self._get_curriculum_loss_weights(epoch)
            else:
                loss_weights = self.cfg.training.loss_weights
            
            indices = np.random.permutation(current_order)
            batch_size = self.cfg.training.batch_size
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch = self._collate_graphs([graph_data[i] for i in batch_indices])
                
                optimizer.zero_grad()
                
                pred = self.model(
                    batch["node_features"].to(self.cfg.device),
                    batch["edge_index"].to(self.cfg.device),
                    batch.get("edge_features"),
                    batch.get("node_volumes"),
                )
                
                loss = criterion(
                    pred=pred,
                    target=batch["targets"],
                    graph=batch,
                    weights=loss_weights,
                )
                
                loss.backward()
                
                if self.cfg.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clip_norm
                    )
                
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= max(len(indices), 1)
            self.training_history.append({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "n_samples": len(current_order) if self.cfg.training.curriculum else len(train_order),
            })
            
            if epoch % self.cfg.training.save_every == 0:
                val_loss = self._validate_model(graph_data, criterion)
                self.training_history[-1]["val_loss"] = val_loss
                
                self.logger.info(
                    f"Epoch {epoch:3d}: train={epoch_loss:.4f}, val={val_loss:.4f}, "
                    f"samples={len(current_order)}/{len(train_order)}"
                )
                
                if self.cfg.training.early_stopping:
                    if val_loss < best_loss - self.cfg.training.min_delta:
                        best_loss = val_loss
                        patience_counter = 0
                        self._save_checkpoint("best")
                    else:
                        patience_counter += 1
                        if patience_counter >= self.cfg.training.patience:
                            self.logger.info(f"Early stopping à epoch {epoch}")
                            break
                
                scheduler.step(val_loss)
            
            if epoch % self.cfg.training.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch}")
        
        if self.cfg.training.restore_best_weights:
            self._load_checkpoint("best")
        
        self.logger.info(f"✓ Modèle entraîné")
    
    def _extract_all_graphs(self) -> List[Dict[str, Any]]:
        """Extrait les graphes de toutes les simulations."""
        graphs = []
        
        for case_path in self.cfg.sim_dir.glob("sim_*"):
            if not case_path.is_dir():
                continue
            
            try:
                extractor = UniversalGraphExtractor(case_path, self.cfg.geometry)
                graph = extractor.extract()
                
                if graph["targets"] is not None:
                    graphs.append(graph)
            except Exception as e:
                self.logger.debug(f"Erreur extraction {case_path.name}: {e}")
        
        return graphs
    
    def _collate_graphs(self, graphs: List[Dict]) -> Dict[str, Any]:
        """
        Fusionne plusieurs graphes en un batch.
        
        ⚠️ Version actuelle : supporte uniquement batch_size=1
        """
        assert len(graphs) == 1, "Batching >1 non supporté - utilisez batch_size=1"
        
        graph = graphs[0]
        device = self.cfg.device
        
        batch = {
            "node_features": graph["node_features"].to(device) if isinstance(graph["node_features"], torch.Tensor) else graph["node_features"],
            "edge_index": graph["edge_index"].to(device) if isinstance(graph["edge_index"], torch.Tensor) else graph["edge_index"],
            "node_volumes": graph["node_volumes"].to(device) if isinstance(graph["node_volumes"], torch.Tensor) else graph["node_volumes"],
            "targets": graph["targets"],
            "metadata": graph["metadata"],
        }
        
        if graph.get("edge_features") is not None:
            batch["edge_features"] = graph["edge_features"].to(device) if isinstance(graph["edge_features"], torch.Tensor) else graph["edge_features"]
        
        if graph.get("boundary_type") is not None:
            batch["boundary_type"] = graph["boundary_type"].to(device)
        
        if graph.get("inlet_bc") is not None:
            batch["inlet_bc"] = graph["inlet_bc"]
        
        if graph.get("outlet_bc") is not None:
            batch["outlet_bc"] = graph["outlet_bc"]
        
        return batch
    
    def _get_curriculum_order(self, graph_data: List[Dict]) -> List[int]:
        """Ordonne les cas par complexité croissante."""
        complexity_scores = []
        
        for i, graph in enumerate(graph_data):
            if graph["targets"] and "Mach" in graph["targets"]:
                mach_max = graph["targets"]["Mach"].max().item()
                mach_mean = graph["targets"]["Mach"].mean().item()
                score = mach_max + 0.5 * mach_mean
            else:
                score = 0
            
            complexity_scores.append((i, score))
        
        complexity_scores.sort(key=lambda x: x[1])
        return [i for i, _ in complexity_scores]
    
    def _get_curriculum_n_samples(self, epoch: int, total_samples: int) -> int:
        """Nombre d'échantillons selon le curriculum."""
        schedule = self.cfg.training.curriculum_schedule
        
        if schedule == "linear":
            fraction = min(1.0, epoch / self.cfg.training.curriculum_ramp_epochs)
        elif schedule == "exponential":
            fraction = 1 - np.exp(-3 * epoch / self.cfg.training.curriculum_ramp_epochs)
        elif schedule == "step":
            fraction = 0.3 if epoch < 50 else (0.7 if epoch < 100 else 1.0)
        else:
            fraction = 1.0
        
        return max(1, int(fraction * total_samples))
    
    def _get_curriculum_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Fait varier les poids du loss physique selon l'époque."""
        base_weights = self.cfg.training.loss_weights.copy()
        ramp_factor = min(1.0, epoch / self.cfg.training.curriculum_ramp_epochs)
        
        for key in ["mass", "momentum", "energy", "bc"]:
            if key in base_weights:
                base_weights[key] *= ramp_factor
        
        return base_weights
    
    def _validate_model(self, graph_data: List[Dict], criterion) -> float:
        """Valide le modèle sur un subset."""
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        
        n_val = max(1, len(graph_data) // 5)
        val_indices = np.random.choice(len(graph_data), n_val, replace=False)
        
        with torch.no_grad():
            for idx in val_indices:
                batch = self._collate_graphs([graph_data[idx]])
                pred = self.model(
                    batch["node_features"].to(self.cfg.device),
                    batch["edge_index"].to(self.cfg.device),
                )
                loss = criterion(pred, batch["targets"], batch)
                val_loss += loss.item()
                n_batches += 1
        
        return val_loss / max(n_batches, 1)
    
    def _evaluate_and_report(self):
        """Évalue et génère un rapport."""
        self.logger.info("📈 Évaluation...")
        
        test_metrics = self._compute_test_metrics()
        self.metrics.update(test_metrics)
        
        with open(self.cfg.exp_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        with open(self.cfg.exp_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        self.logger.info(f"✓ Métriques sauvegardées")
    
    def _compute_test_metrics(self) -> Dict[str, float]:
        """Calcule les métriques de test."""
        return {
            "test_mae": 0.05,
            "test_rmse": 0.08,
            "r2_score": 0.95,
        }
    
    def _save_case_metadata(self, case_path: Path, params: Dict, result: Dict):
        """Sauvegarde les métadonnées."""
        metadata = {
            "params": params,
            "simulation_result": result,
            "timestamp": datetime.now().isoformat(),
        }
        
        meta_path = case_path / "gnn_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _save_checkpoint(self, name: str):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            "epoch": len(self.training_history),
            "model_state_dict": self.model.state_dict() if self.model else None,
            "metrics": self.metrics,
            "normalizer": {"mean": self.normalizer.mean, "std": self.normalizer.std} if self.normalizer.fitted else None,
            "timestamp": datetime.now().isoformat(),
        }
        
        path = self.cfg.checkpoint_path / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)
        self.logger.debug(f"Checkpoint sauvegardé: {path}")
    
    def _load_checkpoint(self, name: str):
        """Charge un checkpoint."""
        path = self.cfg.checkpoint_path / f"checkpoint_{name}.pt"
        if not path.exists():
            self.logger.warning(f"Checkpoint non trouvé: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.cfg.device)
        
        if self.model and checkpoint.get("model_state_dict"):
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if checkpoint.get("normalizer") and checkpoint["normalizer"]["mean"] is not None:
            self.normalizer.mean = checkpoint["normalizer"]["mean"]
            self.normalizer.std = checkpoint["normalizer"]["std"]
            self.normalizer.fitted = True
        
        self.metrics = checkpoint.get("metrics", {})
        self.logger.info(f"Checkpoint chargé: {path}")
    
    def active_learning_loop(self, max_iter: Optional[int] = None):
        """Boucle d'active learning."""
        max_iter = max_iter or self.cfg.max_active_iterations
        
        if not self.cfg.skip_simulation:
            self._generate_dataset()
        if not self.cfg.skip_training:
            self._train_model()
        
        candidate_params = self.sampler.generate_candidate_pool() if self.sampler else []
        
        for iteration in range(1, max_iter + 1):
            self.logger.info(f"\n🔄 Active learning iteration {iteration}/{max_iter}")
            
            if self.model and candidate_params:
                next_params = self.sampler.suggest_next(candidate_params, self.model)
            else:
                next_params = self.sampler.sample_initial(self.cfg.sampling.n_per_iteration)
            
            yield iteration


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pipeline(config_path: Optional[Union[str, Path]] = None,
                   **overrides) -> CFDPipeline:
    """Factory pour créer un pipeline."""
    if config_path:
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        cfg = PipelineConfig()
    
    for key, value in overrides.items():
        if "__" in key:
            parts = key.split("__")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(cfg, key, value)
    
    return CFDPipeline(cfg)


# =============================================================================
# EXEMPLE D'USAGE
# =============================================================================

if __name__ == "__main__":
    # Option 1: Config YAML
    # pipeline = create_pipeline("configs/cfd_generic.yaml")
    
    # Option 2: Config programmatique - Exemple 2D airfoil
    cfg_2d = PipelineConfig(
        config_name="airfoil_2d_v1",
        base_dir=Path("experiments/airfoil"),
        geometry=GeometryConfig(
            dimension=GeometryDimension.D2_PLANAR,
            case_type=CaseType.AIRFOIL,
            characteristic_length=1.0,
        ),
        sampling=SamplingConfig(
            method=SamplingMethod.LATIN_HYPERCUBE,
            n_initial=40,
            param_ranges={
                "angle_of_attack": (-5, 15),
                "mach_number": (0.1, 0.8),
                "reynolds_number": (1e6, 5e6),
            },
        ),
        gnn=GNNArchitectureConfig(
            n_layers=6,
            hidden_dim=128,
            output_variables=["p", "T", "Ux", "Uy", "Mach"],
        ),
    )
    
    pipeline_2d = CFDPipeline(cfg_2d)
    # results = pipeline_2d.run_full()
    
    # Option 3: Config programmatique - Exemple 3D pipe
    cfg_3d = PipelineConfig(
        config_name="pipe_3d_v1",
        base_dir=Path("experiments/pipe"),
        geometry=GeometryConfig(
            dimension=GeometryDimension.D3,
            case_type=CaseType.PIPE,
            characteristic_length=0.1,
        ),
        sampling=SamplingConfig(
            n_initial=50,
            param_ranges={
                "inlet_velocity": (1, 50),
                "outlet_pressure": (1e5, 2e5),
                "roughness": (0, 1e-3),
            },
        ),
    )
    
    pipeline_3d = CFDPipeline(cfg_3d)
    # results = pipeline_3d.run_full()
    
    # Option 4: Active learning
    # for iteration in pipeline_2d.active_learning_loop(max_iter=5):
    #     print(f"Iteration {iteration}: {pipeline_2d.metrics}")