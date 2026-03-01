#!/usr/bin/env python3
"""
run_parametric_study.py - Lance une étude paramétrique sur le silencieux
"""

import sys
from pathlib import Path
import json
import subprocess
from train_cfd_gnn import PipelineConfig, CFDPipeline

class MufflerPipeline(CFDPipeline):
    """
    Pipeline personnalisé pour le silencieux.
    Surcharge _run_simulation pour lancer le script externe.
    """
    
    def _run_simulation(self, case_path: Path, params: dict) -> dict:
        """
        Lance une simulation en appelant run_muffler_simulation.py
        """
        self.logger.info(f"  🏃 Lancement simulation avec {len(params)} paramètres")
        
        # Convertir les paramètres en JSON
        params_json = json.dumps(params, default=str)
        
        # Commande à exécuter
        cmd = [
            sys.executable,
            "run_muffler_simulation.py",
            "--case-dir", str(case_path),
            "--params", params_json
        ]
        
        try:
            # Exécuter avec timeout de 2h
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=7200  # 2 heures max
            )
            
            if result.returncode != 0:
                self.logger.error(f"  ❌ Erreur simulation: {result.stderr[:200]}")
                return {"success": False, "error": result.stderr[:500]}
            
            # Lire le fichier de résultat
            result_file = case_path / "result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    sim_result = json.load(f)
                
                if sim_result.get("success"):
                    self.logger.info(f"  ✅ Simulation réussie: {sim_result.get('n_cells', 0)} cellules")
                else:
                    self.logger.error(f"  ❌ Échec: {sim_result.get('error', 'unknown')}")
                
                return sim_result
            else:
                self.logger.warning("  ⚠️ Fichier result.json non trouvé")
                return {"success": True}  # Assume success
                
        except subprocess.TimeoutExpired:
            self.logger.error("  ⏰ Timeout simulation (2h dépassé)")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            self.logger.error(f"  💥 Exception: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Point d'entrée principal"""
    
    # Vérifier les fichiers nécessaires
    required_files = ["run_muffler_simulation.py"]
    for f in required_files:
        if not Path(f).exists():
            print(f"❌ Fichier requis non trouvé: {f}")
            return
    
    # Configuration
    config_path = Path("configs/muffler_3d.yaml")
    if not config_path.exists():
        print(f"❌ Config non trouvée: {config_path}")
        print("   Création du répertoire configs/...")
        config_path.parent.mkdir(exist_ok=True)
        print("   Copiez la configuration YAML fournie dans configs/muffler_3d.yaml")
        return
    
    print("\n" + "="*70)
    print("🔧 FRAMEWORK CFD+GNN - ÉTUDE PARAMÉTRIQUE SILENCIEUX")
    print("="*70)
    
    # Charger config
    print(f"\n📄 Chargement configuration: {config_path}")
    cfg = PipelineConfig.from_yaml(config_path)
    
    # Aperçu des paramètres
    print(f"\n📊 Paramètres variables ({len(cfg.sampling.param_ranges)}):")
    for name, (min_val, max_val) in cfg.sampling.param_ranges.items():
        print(f"   • {name}: [{min_val}, {max_val}]")
    
    print(f"\n🎯 Objectif: {cfg.n_initial} simulations initiales")
    if cfg.active_learning:
        print(f"🔄 Active learning: {cfg.max_active_iterations} itérations")
    
    # Créer pipeline personnalisé
    pipeline = MufflerPipeline(cfg)
    
    # Menu
    print("\n" + "-"*50)
    print("Options disponibles:")
    print("  1. Exécution complète (simulations + entraînement)")
    print("  2. Génération dataset uniquement (simulations)")
    print("  3. Active learning complet")
    print("  4. Entraînement uniquement (si dataset existe)")
    print("  5. Analyser dataset existant")
    print("  q. Quitter")
    print("-"*50)
    
    choice = input("\nVotre choix [1]: ").strip() or "1"
    
    if choice == "1":
        # Exécution complète
        print("\n🚀 Lancement du pipeline complet...")
        results = pipeline.run_full()
        print(f"\n✅ Résultats finaux: {results}")
        
    elif choice == "2":
        # Dataset uniquement
        print("\n📊 Génération du dataset uniquement...")
        pipeline._generate_dataset()
        print(f"\n✅ Dataset généré dans: {pipeline.cfg.sim_dir}")
        
        # Afficher statistiques
        sim_dirs = list(pipeline.cfg.sim_dir.glob("sim_*"))
        print(f"   {len(sim_dirs)} simulations créées")
        
    elif choice == "3":
        # Active learning
        print("\n🔄 Active learning...")
        
        # Phase 1: Dataset initial
        print("\n📊 Phase 1: Génération dataset initial")
        pipeline._generate_dataset()
        
        # Phase 2: Premier entraînement
        print("\n🧠 Phase 2: Premier entraînement")
        pipeline._train_model()
        
        # Phase 3: Itérations active learning
        for iteration in pipeline.active_learning_loop():
            print(f"\n🔄 Itération {iteration}:")
            print(f"   - Nouvelles simulations avec paramètres sélectionnés")
            pipeline._generate_dataset()  # Génère les nouvelles simulations
            pipeline._train_model()       # Ré-entraîne avec toutes les données
            
    elif choice == "4":
        # Entraînement uniquement
        print("\n🧠 Entraînement uniquement...")
        pipeline._train_model()
        
    elif choice == "5":
        # Analyser dataset
        print("\n📊 Analyse du dataset...")
        graph_data = pipeline._extract_all_graphs()
        print(f"   {len(graph_data)} graphes extraits")
        
        if graph_data:
            # Stats sur le premier graphe
            g = graph_data[0]
            print(f"\n   Exemple (sim_0000):")
            print(f"   • Noeuds: {g['n_nodes']}")
            print(f"   • Arêtes: {g['n_edges']}")
            print(f"   • Dimension: {g['spatial_dim']}D")
            if g['targets']:
                print(f"   • Champs: {list(g['targets'].keys())}")
        
    elif choice.lower() == "q":
        print("Au revoir !")
        return
    
    else:
        print("Choix invalide")

if __name__ == "__main__":
    main()