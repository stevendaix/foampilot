#!/usr/bin/env python
"""
foampilot.cases.nozzle — Configuration Laval nozzle (hors module générique)
==========================================================================

Ce fichier contient TOUTE la logique spécifique au cas nozzle :
- Lecture des paramètres depuis nozzle_config.json
- Zones d'échantillonnage adaptées
- Métriques physiques (Mach, chocs)

Usage :
    from foampilot.gnn import Experiment, GNNConfig
    from foampilot.cases.nozzle import nozzle_param_reader, nozzle_metrics, nozzle_sampler_zones
    
    cfg = GNNConfig(
        name="nozzle_v1",
        fields_out=["p", "T", "Ux", "Uy"],
        physical_params_keys=["R_throat", "R_exit", "p_total", "p_outlet", "AR_exit"],
        sampler_zones=nozzle_sampler_zones(),
        param_reader=nozzle_param_reader,
        metrics_computer=nozzle_metrics,
    )
    
    exp = Experiment(cfg)
    exp.extract_graphs()
    exp.fit()
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# 1. ZONES D'ÉCHANTILLONNAGE NOZZLE
# =============================================================================

def nozzle_sampler_zones():
    """Zones optimisées pour nozzle (couche limite µm, col raffiné)."""
    return [
        (5e-3, 1.00),   # 0-5mm paroi → 100%
        (0.05, 0.50),   # 5-50mm → 50%
        (1e9,  0.25),   # bulk → 25%
    ]


# =============================================================================
# 2. LECTEUR DE PARAMÈTRES NOZZLE
# =============================================================================

def nozzle_param_reader(case_path: Path) -> Optional[Dict[str, float]]:
    """
    Lit les paramètres physiques depuis nozzle_config.json.
    
    Returns:
        Dict avec R_throat, R_exit, p_total, p_outlet, AR_exit
        ou None si le fichier est absent
    """
    config_path = case_path / "nozzle_config.json"
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path) as f:
            raw = json.load(f)
        
        nozzle_data = raw.get("nozzle", {})
        operating_data = raw.get("operating", {})
        
        R_throat = nozzle_data.get("R_throat", 0.08)
        R_exit = nozzle_data.get("R_exit", 0.18)
        
        params = {
            "R_throat": float(R_throat),
            "R_exit": float(R_exit),
            "p_total": float(operating_data.get("p_total_inlet", 3e5)),
            "p_outlet": float(operating_data.get("p_static_outlet", 1e4)),
            "AR_exit": float((R_exit / max(R_throat, 1e-8)) ** 2),
        }
        
        return params
        
    except Exception as e:
        print(f"Erreur lecture nozzle_config.json : {e}")
        return None


# =============================================================================
# 3. MÉTRIQUES PHYSIQUES NOZZLE
# =============================================================================

def nozzle_metrics(
    pred_raw: torch.Tensor,
    true_raw: torch.Tensor,
    fields: List[str],
    node_type: torch.Tensor,
    rho: float = 1.225,
    gamma: float = 1.4,
    R_gas: float = 287.0,
) -> Dict[str, float]:
    """
    Métriques spécifiques nozzle :
    - Erreur relative sur Mach
    - Mach max prédit (détection choc)
    - Erreur température de stagnation (si disponible)
    
    Ces métriques sont spécifiques au nozzle et ne doivent pas être
    dans le module générique.
    """
    from foampilot.gnn import NODE_FLUID, NODE_OUTLET
    
    # Filtrage noeuds internes
    mask = ((node_type == NODE_FLUID) | (node_type == NODE_OUTLET)).cpu().numpy()
    pred_np = pred_raw.cpu().numpy()[mask]
    true_np = true_raw.cpu().numpy()[mask]
    
    if len(pred_np) == 0:
        return {}
    
    f_idx = {f: i for i, f in enumerate(fields)}
    metrics = {}
    
    # Vérifier que les champs nécessaires sont présents
    if "T" not in f_idx:
        return {}
    
    u_fields = [f for f in fields if f.upper().startswith("U")]
    if len(u_fields) < 1:
        return {}
    
    # Extraction
    T_pred = np.clip(pred_np[:, f_idx["T"]], 1.0, None)  # T > 0
    T_true = np.clip(true_np[:, f_idx["T"]], 1.0, None)
    
    u_indices = [f_idx[f] for f in u_fields]
    U_pred = np.linalg.norm(pred_np[:, u_indices], axis=1)
    U_true = np.linalg.norm(true_np[:, u_indices], axis=1)
    
    # Nombre de Mach
    Ma_pred = U_pred / np.sqrt(gamma * R_gas * T_pred)
    Ma_true = U_true / np.sqrt(gamma * R_gas * T_true)
    
    metrics["rel_err_Mach"] = float(
        np.abs(Ma_pred - Ma_true).mean() / (Ma_true.mean() + 1e-8)
    )
    metrics["max_Ma_pred"] = float(Ma_pred.max())
    metrics["max_Ma_true"] = float(Ma_true.max())
    
    # Température de stagnation (approximation)
    # T0 = T * (1 + (gamma-1)/2 * Ma^2)
    T0_pred = T_pred * (1 + (gamma - 1) / 2 * Ma_pred ** 2)
    T0_true = T_true * (1 + (gamma - 1) / 2 * Ma_true ** 2)
    
    metrics["rel_err_T0"] = float(
        np.abs(T0_pred - T0_true).mean() / (T0_true.mean() + 1e-8)
    )
    
    # Détection de choc (si Mach > 1 puis retombe < 1 dans la direction x)
    # C'est un indicateur qualitatif
    if Ma_pred.max() > 1.1 and Ma_pred.min() < 0.9:
        metrics["shock_detected"] = 1.0
    else:
        metrics["shock_detected"] = 0.0
    
    return metrics


# =============================================================================
# 4. HELPER : Configuration complète nozzle
# =============================================================================

def make_nozzle_config(
    name: str = "nozzle_exp",
    sim_dir: str = "simulations/nozzle",
    **overrides
) -> "GNNConfig":
    """
    Construit une config complète pour nozzle.
    
    Args:
        name : nom de l'expérience
        sim_dir : dossier des simulations
        **overrides : surcharges (ex: n_layers=8, hidden_dim=256)
    
    Returns:
        GNNConfig prête à l'emploi
    """
    from foampilot.gnn import GNNConfig
    
    cfg = GNNConfig(
        name=name,
        fields_out=["p", "T", "Ux", "Uy"],
        physical_params_keys=["R_throat", "R_exit", "p_total", "p_outlet", "AR_exit"],
        sim_dir=Path(sim_dir),
        graph_dir=Path(f"graphs/{name}"),
        model_dir=Path("models"),
        
        # Échantillonnage nozzle
        sampler_zones=nozzle_sampler_zones(),
        
        # Loss adaptée (no-slip plus fort pour nozzle)
        loss_data=1.0,
        loss_inlet=2.0,
        loss_no_slip=8.0,
        
        # Callbacks spécifiques nozzle
        param_reader=nozzle_param_reader,
        metrics_computer=nozzle_metrics,
        
        # Dimension (2D axisymétrique)
        dimension=2,
    )
    
    # Appliquer les surcharges
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"⚠ Attribut inconnu ignoré: {key}")
    
    return cfg


# =============================================================================
# 5. EXEMPLE D'USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Exemple standalone : entraînement GNN sur nozzle.
    """
    from foampilot.gnn import Experiment
    
    # Configuration nozzle avec surcharges
    cfg = make_nozzle_config(
        name="nozzle_v2",
        sim_dir="simulations/nozzle",
        n_layers=8,
        hidden_dim=256,
        epochs=200,
    )
    
    # Pipeline complet
    exp = Experiment(cfg)
    exp.extract_graphs()
    exp.fit()
    exp.evaluate()
    
    # Comparaison sur un cas
    exp.compare("simulations/nozzle/sim_0099")
    
    print(f"\n✓ Modèle sauvegardé dans {cfg.experiment_dir()}")
