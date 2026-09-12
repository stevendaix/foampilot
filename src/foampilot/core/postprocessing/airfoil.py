#!/usr/bin/env python
"""
foampilot.cases.airfoil — Configuration NACA airfoil (hors module générique)
===========================================================================

Toute la logique spécifique airfoil :
- Lecture angle_of_attack, Re, Ma depuis config
- Zones d'échantillonnage adaptées (couche limite)
- Métriques Cp, Cl, Cd

Usage :
    from foampilot.gnn import Experiment, GNNConfig
    from foampilot.cases.airfoil import make_airfoil_config
    
    cfg = make_airfoil_config(name="airfoil_v1", n_layers=6)
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
# 1. ZONES D'ÉCHANTILLONNAGE AIRFOIL
# =============================================================================

def airfoil_sampler_zones():
    """Zones optimisées pour airfoil (couche limite raffinée)."""
    return [
        (0.05, 1.00),   # 0-5cm couche limite → 100%
        (0.50, 0.33),   # 5-50cm champ proche → 33%
        (1e9,  0.10),   # champ lointain → 10%
    ]


# =============================================================================
# 2. LECTEUR DE PARAMÈTRES AIRFOIL
# =============================================================================

def airfoil_param_reader(case_path: Path) -> Optional[Dict[str, float]]:
    """
    Lit angle_of_attack, Re, Ma depuis airfoil_config.json ou config.json.
    
    Returns:
        Dict avec angle_of_attack, Re, Ma
        ou None si absent
    """
    # Chercher le fichier de config
    config_files = [
        case_path / "airfoil_config.json",
        case_path / "config.json",
    ]
    
    config_path = None
    for f in config_files:
        if f.exists():
            config_path = f
            break
    
    if config_path is None:
        return None
    
    try:
        with open(config_path) as f:
            raw = json.load(f)
        
        sim_data = raw.get("simulation", {})
        
        params = {
            "angle_of_attack": float(sim_data.get("angle_of_attack", 0.0)),
            "Re": float(sim_data.get("Re", 1e6)),
            "Ma": float(sim_data.get("Ma", 0.1)),
        }
        
        return params
        
    except Exception as e:
        print(f"Erreur lecture airfoil config : {e}")
        return None


# =============================================================================
# 3. MÉTRIQUES PHYSIQUES AIRFOIL
# =============================================================================

def airfoil_metrics(
    pred_raw: torch.Tensor,
    true_raw: torch.Tensor,
    fields: List[str],
    node_type: torch.Tensor,
    rho: float = 1.225,
    U_inf: float = 10.0,  # vitesse amont (à paramétrer si nécessaire)
    chord: float = 1.0,   # corde de référence
) -> Dict[str, float]:
    """
    Métriques spécifiques airfoil :
    - Coefficient de pression Cp
    - Coefficient de portance Cl (approximation)
    - Coefficient de traînée Cd (approximation)
    
    Ces métriques sont spécifiques à l'airfoil et ne doivent pas être
    dans le module générique.
    """
    from foampilot.gnn import NODE_FLUID, NODE_OUTLET, NODE_WALL
    
    # Filtrage noeuds internes
    mask = ((node_type == NODE_FLUID) | (node_type == NODE_OUTLET)).cpu().numpy()
    pred_np = pred_raw.cpu().numpy()[mask]
    true_np = true_raw.cpu().numpy()[mask]
    node_type_np = node_type.cpu().numpy()
    
    if len(pred_np) == 0:
        return {}
    
    f_idx = {f: i for i, f in enumerate(fields)}
    metrics = {}
    
    # Coefficient de pression
    if "p" in f_idx:
        q_inf = 0.5 * rho * U_inf ** 2
        
        Cp_pred = pred_np[:, f_idx["p"]] / q_inf
        Cp_true = true_np[:, f_idx["p"]] / q_inf
        
        metrics["rel_err_Cp"] = float(
            np.abs(Cp_pred - Cp_true).mean() / (np.abs(Cp_true).mean() + 1e-8)
        )
        metrics["max_err_Cp"] = float(np.abs(Cp_pred - Cp_true).max())
        
        # Cp sur la paroi uniquement (pour analyse de la distribution)
        wall_mask = (node_type_np == NODE_WALL)
        if wall_mask.any():
            # On extrait pred/true complets (pas juste internes)
            pred_full = pred_raw.cpu().numpy()
            true_full = true_raw.cpu().numpy()
            
            Cp_wall_pred = pred_full[wall_mask, f_idx["p"]] / q_inf
            Cp_wall_true = true_full[wall_mask, f_idx["p"]] / q_inf
            
            metrics["rel_err_Cp_wall"] = float(
                np.abs(Cp_wall_pred - Cp_wall_true).mean() / (np.abs(Cp_wall_true).mean() + 1e-8)
            )
    
    # Approximation très simplifiée de Cl et Cd
    # (nécessiterait l'intégration sur la paroi + orientation des normales)
    # Ici on donne juste une indication qualitative
    u_fields = [f for f in fields if f.upper().startswith("U")]
    if "p" in f_idx and len(u_fields) >= 2:
        # Moyenne de pression sur paroi supérieure vs inférieure
        # (approximation TRÈS grossière — ne pas utiliser pour validation quantitative)
        
        wall_mask = (node_type_np == NODE_WALL)
        if wall_mask.any():
            pred_full = pred_raw.cpu().numpy()
            
            p_wall_pred = pred_full[wall_mask, f_idx["p"]]
            
            # Approximation : différence de pression moyenne
            # (dans un vrai code, il faudrait intégrer avec les normales)
            delta_p_pred = np.std(p_wall_pred)  # écart-type comme proxy
            
            # Portance approximative ~ intégrale pression
            Cl_approx = delta_p_pred / (q_inf * chord)
            metrics["Cl_approx"] = float(Cl_approx)
    
    return metrics


# =============================================================================
# 4. HELPER : Configuration complète airfoil
# =============================================================================

def make_airfoil_config(
    name: str = "airfoil_exp",
    sim_dir: str = "simulations/airfoil",
    **overrides
) -> "GNNConfig":
    """
    Construit une config complète pour airfoil.
    
    Args:
        name : nom de l'expérience
        sim_dir : dossier des simulations
        **overrides : surcharges
    
    Returns:
        GNNConfig prête à l'emploi
    """
    from foampilot.gnn import GNNConfig
    
    cfg = GNNConfig(
        name=name,
        fields_out=["p", "Ux", "Uy"],
        physical_params_keys=["angle_of_attack", "Re", "Ma"],
        sim_dir=Path(sim_dir),
        graph_dir=Path(f"graphs/{name}"),
        model_dir=Path("models"),
        
        # Échantillonnage airfoil
        sampler_zones=airfoil_sampler_zones(),
        
        # Loss adaptée
        loss_data=1.0,
        loss_inlet=2.0,
        loss_no_slip=5.0,
        
        # Callbacks spécifiques airfoil
        param_reader=airfoil_param_reader,
        metrics_computer=airfoil_metrics,
        
        # Dimension (2D par extrusion)
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
    """Exemple standalone : entraînement GNN sur airfoil."""
    from foampilot.gnn import Experiment
    
    cfg = make_airfoil_config(
        name="airfoil_v1",
        sim_dir="simulations/airfoil",
        n_layers=6,
        hidden_dim=128,
        epochs=150,
    )
    
    exp = Experiment(cfg)
    exp.extract_graphs()
    exp.fit()
    exp.evaluate()
    
    print(f"\n✓ Modèle sauvegardé dans {cfg.experiment_dir()}")
