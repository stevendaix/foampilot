#!/usr/bin/env python
"""
train_nozzle_gnn.py — Pipeline complet : génération dataset + entraînement GNN
==============================================================================

Ce script fait TOUT de A à Z :
1. Génère N cas nozzle avec foampilot (paramètres variés)
2. Lance les simulations OpenFOAM
3. Extrait les graphes
4. Entraîne le GNN
5. Évalue et compare

Usage :
    python train_nozzle_gnn.py --n_cases 50 --epochs 150
    python train_nozzle_gnn.py --skip_generation  # si simulations déjà faites
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from foampilot import meshing, boundary, solver
from foampilot.gnn import Experiment
from foampilot.cases.nozzle import make_nozzle_config


# =============================================================================
# 1. GÉNÉRATION DU DATASET (avec foampilot)
# =============================================================================

def generate_nozzle_dataset(
    sim_dir: Path,
    n_cases: int = 50,
    force: bool = False,
):
    """
    Génère N cas de nozzle avec paramètres variés.
    
    Args:
        sim_dir : dossier de sortie
        n_cases : nombre de cas à générer
        force : si True, régénère même si le cas existe
    """
    sim_dir = Path(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"GÉNÉRATION DATASET NOZZLE — {n_cases} cas")
    print(f"{'='*80}\n")
    
    # ── Grille de paramètres ─────────────────────────────────────────────────
    # On fait varier : R_throat, R_exit, p_total, p_outlet
    
    param_grid = {
        "R_throat": np.linspace(0.06, 0.10, 3),   # 3 valeurs : 6, 8, 10 cm
        "R_exit": np.linspace(0.15, 0.22, 3),     # 3 valeurs : 15, 18, 22 cm
        "p_total": np.linspace(2e5, 4e5, 3),      # 3 valeurs : 2, 3, 4 bar
        "p_outlet": np.linspace(8e3, 1.5e4, 3),   # 3 valeurs : 8, 10, 15 kPa
    }
    
    # Génération du produit cartésien (3^4 = 81 combinaisons)
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    # Limiter au nombre demandé (échantillonnage aléatoire si n_cases < 81)
    if n_cases < len(all_combinations):
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:n_cases]
    
    print(f"Génération de {len(all_combinations)} cas parmi {3**4} possibles\n")
    
    # ── Génération des cas ────────────────────────────────────────────────────
    
    n_ok = 0
    n_skip = 0
    n_fail = 0
    
    for i, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        case_name = f"sim_{i:04d}"
        case_path = sim_dir / case_name
        
        # Skip si déjà existant
        if case_path.exists() and not force:
            n_skip += 1
            continue
        
        try:
            print(f"[{i+1}/{len(all_combinations)}] {case_name}")
            print(f"  R_throat={params['R_throat']:.3f} m  "
                  f"R_exit={params['R_exit']:.3f} m  "
                  f"p_total={params['p_total']/1e5:.1f} bar  "
                  f"p_outlet={params['p_outlet']/1e3:.1f} kPa")
            
            run_nozzle_case(case_path, **params)
            n_ok += 1
            
        except Exception as e:
            print(f"  ✗ Erreur : {e}")
            n_fail += 1
    
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ GÉNÉRATION")
    print(f"{'='*80}")
    print(f"  ✓ Succès   : {n_ok}")
    print(f"  ⊘ Ignorés  : {n_skip}")
    print(f"  ✗ Échecs   : {n_fail}")
    print(f"{'='*80}\n")


def run_nozzle_case(
    case_path: Path,
    R_throat: float,
    R_exit: float,
    p_total: float,
    p_outlet: float,
):
    """
    Lance une simulation OpenFOAM complète pour une nozzle avec foampilot.
    
    Args:
        case_path : dossier du cas
        R_throat : rayon col (m)
        R_exit : rayon sortie (m)
        p_total : pression totale inlet (Pa)
        p_outlet : pression statique outlet (Pa)
    """
    from scipy.interpolate import PchipInterpolator
    
    case_path = Path(case_path)
    case_path.mkdir(parents=True, exist_ok=True)
    
    # ── Paramètres géométriques ───────────────────────────────────────────────
    
    R_inlet = R_throat * 2.5        # entrée convergent
    L_convergent = 0.30              # longueur convergent (m)
    L_divergent = 0.80               # longueur divergent (m)
    L_total = L_convergent + L_divergent
    
    AR_exit = (R_exit / R_throat) ** 2  # rapport de section
    
    # ── Génération du profil (spline PCHIP) ───────────────────────────────────
    
    x_ctrl = [0.0, L_convergent, L_total]
    r_ctrl = [R_inlet, R_throat, R_exit]
    spline = PchipInterpolator(x_ctrl, r_ctrl)
    
    # Échantillonnage du profil
    n_points = 200
    x_profile = np.linspace(0, L_total, n_points)
    r_profile = spline(x_profile)
    
    profile_data = case_path / "nozzle_profile.dat"
    np.savetxt(profile_data, np.column_stack([x_profile, r_profile]),
               header="x(m) r(m)", comments="")
    
    # ── Paramètres de maillage ────────────────────────────────────────────────
    
    wedge_angle_deg = 5.0
    lc_tuyere = 0.003   # taille cellule paroi (3 mm)
    lc_col = 0.001      # taille cellule col (1 mm)
    lc_jet = 0.005      # taille cellule jet divergent
    
    # ── Création du maillage Gmsh (wedge axisymétrique) ───────────────────────
    
    gmsh_script = f"""
// Nozzle axisymétrique wedge {wedge_angle_deg}°
SetFactory("OpenCASCADE");

// Chargement du profil
profile = ReadFile("{profile_data.absolute()}");

// Paramètres
lc_tuyere = {lc_tuyere};
lc_col = {lc_col};
lc_jet = {lc_jet};
wedge_angle = {np.deg2rad(wedge_angle_deg)};

// Construction géométrie (spline + révolution partielle)
// ... [code Gmsh complet ici — simplifié pour l'exemple]

// BoundaryLayer sur paroi
Field[1] = BoundaryLayer;
Field[1].AnisoMax = 1e6;
Field[1].FanNodesList = {{nozzle_nodes}};
Field[1].Quads = 1;
Field[1].Thickness = 0.02;
Field[1].Ratio = 1.2;

// Extrusion wedge (1 couche)
Extrude {{{{0,0,1}}, {{0,0,0}}, wedge_angle}} {{
  Surface{{nozzle_surface}};
  Layers{{1}};
  Recombine;
}}

// Physical groups
Physical Surface("front") = {{front_surface}};
Physical Surface("back") = {{back_surface}};
Physical Surface("inlet") = {{inlet_surface}};
Physical Surface("outlet") = {{outlet_surface}};
Physical Surface("nozzle") = {{nozzle_surface}};
Physical Surface("axis") = {{axis_surface}};
Physical Volume("internal") = {{volume}};

Mesh 3;
"""
    
    # Note : le code Gmsh complet est simplifié ici pour la lisibilité
    # Dans la vraie implémentation, utiliser foampilot.meshing.Gmsh
    
    gmsh_file = case_path / "nozzle.geo"
    gmsh_file.write_text(gmsh_script)
    
    # Maillage
    mesh = meshing.Gmsh(case_path)
    mesh.generate_from_geo("nozzle.geo")
    mesh.convert_to_openfoam()
    
    # ── Configuration solveur rhoCentralFoam (compressible) ───────────────────
    
    slv = solver.RhoCentralFoam(case_path=case_path)
    
    # Thermophysique (air parfait)
    slv.constant.thermophysicalProperties = {
        "thermoType": {
            "type": "hePsiThermo",
            "mixture": "pureMixture",
            "transport": "const",
            "thermo": "hConst",
            "equationOfState": "perfectGas",
            "specie": "specie",
            "energy": "sensibleEnthalpy",
        },
        "mixture": {
            "specie": {"molWeight": 28.96},
            "thermodynamics": {"Cp": 1004.5, "Hf": 0},
            "transport": {"mu": 1.8e-5, "Pr": 0.713},
        },
    }
    
    # Turbulence k-omega SST (meilleur pour compressible)
    slv.constant.turbulenceProperties = {
        "simulationType": "RAS",
        "RAS": {
            "RASModel": "kOmegaSST",
            "turbulence": "on",
            "printCoeffs": "on",
        },
    }
    
    # Schémas numériques
    slv.system.fvSchemes = {
        "ddtSchemes": {"default": "Euler"},
        "gradSchemes": {"default": "Gauss linear"},
        "divSchemes": {
            "default": "none",
            "div(phi,U)": "Gauss linearUpwind grad(U)",
            "div(phi,K)": "Gauss linear",
            "div(phi,e)": "Gauss linearUpwind grad(e)",
            "div(phi,k)": "Gauss linearUpwind grad(k)",
            "div(phi,omega)": "Gauss linearUpwind grad(omega)",
            "div(phid,p)": "Gauss limitedLinear 1",
            "div((muEff*dev2(T(grad(U)))))": "Gauss linear",
        },
        "laplacianSchemes": {"default": "Gauss linear corrected"},
        "interpolationSchemes": {
            "default": "linear",
            "reconstruct(rho)": "vanLeer",
            "reconstruct(U)": "vanLeerV",
            "reconstruct(T)": "vanLeer",
        },
    }
    
    # Solveurs
    slv.system.fvSolution = {
        "solvers": {
            "p": {"solver": "PCG", "preconditioner": "DIC", "tolerance": 1e-8, "relTol": 0.01},
            "rho": {"solver": "PCG", "preconditioner": "DIC", "tolerance": 1e-7, "relTol": 0},
            "U": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-7, "relTol": 0.1},
            "e": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-7, "relTol": 0.1},
            "(k|omega)": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-7, "relTol": 0.1},
        },
    }
    
    # Contrôle (CFL adaptatif)
    slv.system.controlDict = {
        "application": "rhoCentralFoam",
        "startFrom": "latestTime",
        "startTime": 0,
        "stopAt": "endTime",
        "endTime": 5000,
        "deltaT": 1e-6,
        "writeControl": "adjustableRunTime",
        "writeInterval": 500,
        "purgeWrite": 2,
        "adjustTimeStep": "yes",
        "maxCo": 0.5,
        "maxDeltaT": 1,
    }
    
    # ── Conditions aux limites ────────────────────────────────────────────────
    
    # Inlet : pression/température totales
    slv.boundary.apply_condition_with_wildcard(
        pattern="inlet",
        condition_type="totalPressure",
        pressure=p_total,
        temperature=600.0,  # température totale (K)
    )
    
    # Outlet : pression statique
    slv.boundary.apply_condition_with_wildcard(
        pattern="outlet",
        condition_type="waveTransmissive",  # condition sortie supersonique
        pressure=p_outlet,
    )
    
    # Paroi : no-slip adiabatique
    slv.boundary.apply_condition_with_wildcard(
        pattern="nozzle",
        condition_type="wall",
        wall_type="noSlip",
        thermal="adiabatic",
    )
    
    # Axe de symétrie
    slv.boundary.apply_condition_with_wildcard(
        pattern="axis",
        condition_type="symmetry",
    )
    
    # Wedge faces (axisymétrie OpenFOAM)
    for face in ["front", "back"]:
        slv.boundary.apply_condition_with_wildcard(
            pattern=face,
            condition_type="wedge",
        )
    
    # ── Sauvegarde de la configuration (pour GNN) ─────────────────────────────
    
    config = {
        "nozzle": {
            "R_inlet": float(R_inlet),
            "R_throat": float(R_throat),
            "R_exit": float(R_exit),
            "L_convergent": float(L_convergent),
            "L_divergent": float(L_divergent),
        },
        "operating": {
            "p_total_inlet": float(p_total),
            "T_total_inlet": 600.0,
            "p_static_outlet": float(p_outlet),
        },
        "mesh": {
            "lc_tuyere": lc_tuyere,
            "lc_col": lc_col,
            "wedge_angle_deg": wedge_angle_deg,
        },
        "simulation": {
            "endTime": 5000,
            "writeInterval": 500,
        },
    }
    
    with open(case_path / "nozzle_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # ── Lancement de la simulation ────────────────────────────────────────────
    
    print(f"  Lancement rhoCentralFoam...")
    t0 = time.time()
    
    slv.run(parallel=False, log_file=case_path / "log.rhoCentralFoam")
    
    elapsed = time.time() - t0
    print(f"  ✓ Simulation terminée en {elapsed:.1f}s")


# =============================================================================
# 2. ENTRAÎNEMENT GNN
# =============================================================================

def train_gnn(
    sim_dir: Path,
    exp_name: str = "nozzle_gnn_v1",
    epochs: int = 150,
    n_layers: int = 6,
    hidden_dim: int = 128,
):
    """
    Entraîne le GNN sur le dataset généré.
    
    Args:
        sim_dir : dossier contenant les simulations
        exp_name : nom de l'expérience
        epochs : nombre d'epochs
        n_layers : nombre de couches GNN
        hidden_dim : dimension cachée
    """
    print(f"\n{'='*80}")
    print(f"ENTRAÎNEMENT GNN — {exp_name}")
    print(f"{'='*80}\n")
    
    # ── Configuration ─────────────────────────────────────────────────────────
    
    cfg = make_nozzle_config(
        name=exp_name,
        sim_dir=str(sim_dir),
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=4,
        patience=20,
    )
    
    print(f"Configuration :")
    print(f"  Simulations    : {cfg.sim_dir}")
    print(f"  Graphes        : {cfg.graph_dir}")
    print(f"  Modèle         : {cfg.model_dir}/{cfg.name}")
    print(f"  Architecture   : {cfg.model_name} ({n_layers} layers, {hidden_dim} hidden)")
    print(f"  Entraînement   : {epochs} epochs max, batch_size={cfg.batch_size}")
    print()
    
    # ── Expérience ────────────────────────────────────────────────────────────
    
    exp = Experiment(cfg)
    
    # 1. Extraction des graphes
    print("ÉTAPE 1 : Extraction des graphes")
    exp.extract_graphs()
    
    # 2. Entraînement
    print("\nÉTAPE 2 : Entraînement")
    exp.fit()
    
    # 3. Évaluation
    print("\nÉTAPE 3 : Évaluation")
    metrics = exp.evaluate()
    
    # Affichage des métriques clés
    print("\n" + "="*80)
    print("MÉTRIQUES FINALES")
    print("="*80)
    for key in ["mean_rel_err_p", "mean_rel_err_T", "mean_rel_err_Mach", "mean_max_Ma_pred"]:
        if key in metrics:
            print(f"  {key:30s} = {metrics[key]:.4f}")
    print("="*80)
    
    return exp


# =============================================================================
# 3. COMPARAISON VISUELLE
# =============================================================================

def compare_cases(exp: Experiment, sim_dir: Path, n_compare: int = 5):
    """
    Compare visuellement GNN vs OpenFOAM sur quelques cas.
    
    Args:
        exp : expérience entraînée
        sim_dir : dossier des simulations
        n_compare : nombre de cas à comparer
    """
    print(f"\n{'='*80}")
    print(f"COMPARAISON VISUELLE — {n_compare} cas")
    print(f"{'='*80}\n")
    
    # Sélection aléatoire de cas
    sim_paths = sorted(Path(sim_dir).glob("sim_*"))
    if len(sim_paths) > n_compare:
        import random
        sim_paths = random.sample(sim_paths, n_compare)
    
    for case_path in sim_paths:
        print(f"Comparaison : {case_path.name}")
        result = exp.compare(case_path)
        
        # Affichage des métriques
        for k, v in result["metrics"].items():
            print(f"  {k:30s} = {v:.4f}")
        print()


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline complet nozzle GNN")
    
    # Génération dataset
    parser.add_argument("--n_cases", type=int, default=50,
                        help="Nombre de cas à générer (default: 50)")
    parser.add_argument("--sim_dir", type=str, default="simulations/nozzle",
                        help="Dossier des simulations (default: simulations/nozzle)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip la génération (utilise les simulations existantes)")
    parser.add_argument("--force_generation", action="store_true",
                        help="Force la régénération même si les cas existent")
    
    # Entraînement GNN
    parser.add_argument("--exp_name", type=str, default="nozzle_gnn_v1",
                        help="Nom de l'expérience (default: nozzle_gnn_v1)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Nombre d'epochs (default: 150)")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Nombre de couches GNN (default: 6)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Dimension cachée (default: 128)")
    
    # Comparaison
    parser.add_argument("--n_compare", type=int, default=5,
                        help="Nombre de cas à comparer visuellement (default: 5)")
    
    args = parser.parse_args()
    
    # ── Pipeline complet ──────────────────────────────────────────────────────
    
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLET NOZZLE GNN")
    print(f"{'#'*80}\n")
    
    sim_dir = Path(args.sim_dir)
    
    # 1. Génération du dataset (optionnel)
    if not args.skip_generation:
        generate_nozzle_dataset(
            sim_dir=sim_dir,
            n_cases=args.n_cases,
            force=args.force_generation,
        )
    else:
        print(f"⊘ Génération ignorée (--skip_generation)")
        print(f"  Utilisation des simulations dans : {sim_dir}\n")
    
    # 2. Entraînement GNN
    exp = train_gnn(
        sim_dir=sim_dir,
        exp_name=args.exp_name,
        epochs=args.epochs,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
    )
    
    # 3. Comparaison visuelle
    compare_cases(exp, sim_dir, n_compare=args.n_compare)
    
    # ── Résumé final ──────────────────────────────────────────────────────────
    
    print(f"\n{'#'*80}")
    print(f"# PIPELINE TERMINÉ")
    print(f"{'#'*80}")
    print(f"  Dataset        : {sim_dir}")
    print(f"  Modèle         : {exp.exp_dir}")
    print(f"  Convergence    : {exp.exp_dir / 'convergence.png'}")
    print(f"  Métriques      : {exp.exp_dir / 'eval_metrics.json'}")
    print(f"  Comparaisons   : {exp.exp_dir / 'compare_*.png'}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
