#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Génère un rapport PDF de validation du modèle Windkessel.
Utilise typst pour créer un document scientifique avec:
- Paramètres du modèle
- Métriques de validation
- Comparaison des formes d'onde
- Résultats de validation
"""

from pathlib import Path
import json

# Add foampilot to path
import sys
sys.path.insert(0, '/home/steven/foampilot')

from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

# Base path
base_path = Path(__file__).parent

# Path to validation results
results_dir = base_path / 'results'

# Load validation metrics
metrics_path = results_dir / 'validation_metrics.json'
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Extract parameters
params = metrics['parameters']

# Create scientific document
doc = ScientificDocument(
    title="Validation du Modèle Windkessel",
    author="Foampilot Automated Report"
)

# =========================================
# 1. TITRE ET RÉSUMÉ
# =========================================
doc.add_section("Validation du Modèle Windkessel", "", level=1)

# Abstract
model_desc = "5 éléments (Rc, Rp, C, L, Cprox)" if 'Cprox' in params else "4 éléments (Rc, Rp, C, L)"
abstract = f"""
Cette étude présente la validation du modèle Windkessel à {model_desc} 
contre des données de référence expérimentales. Le modèle est un modèle lumped-parameter 
cardiovasculaire utilisé pour simuler la pressurisation aortique.
"""
doc.add_section("Résumé", abstract)

# =========================================
# 2. THÉORIE DU MODÈLE
# =========================================
doc.add_section("Théorie du Modèle Windkessel", "", level=1)

# Analogie électrique-hydraulique
analogie = """Le modèle Windkessel est un modèle lumped-parameter cardiovasculaire basé sur l'analogie entre les systèmes hydraulique et électrique :

- Pression P (Pa) <-> Tension V (unités : Pa ou mmHg)
- Débit Q (m³/s) <-> Courant I (unités : m³/s)
- Résistance R <-> Résistance R (unités : Pa·s/m³)
- Compliance C <-> Capacitance C (unités : m³/Pa)
- Inertance L <-> Inductance L (unités : Pa·s²/m³)"""
doc.add_section(analogie, "", level=2)

# 4-element model equations
doc.add_section("Modèle à 4 Éléments (Rc, Rp, C, L)", "Le modèle 4-éléments inclut une inertance L en série avec Rc.", level=2)

# Equations using add_equation - using Typst math syntax
doc.add_equation("C ((d P_2) / (d t)) + P_2 / R_p = Q(t)", caption="Équation différentielle au nœud de compliance", label="eq:4e_ode")
doc.add_equation("P_1(t) = P_2(t) + R_c Q(t) + L ((d Q) / (d t))", caption="Reconstruction de la pression aortique", label="eq:4e_recon")
doc.add_equation("tau = R_p C", caption="Constante de temps diastolique (tau = Rp x C)", label="eq:tau")

# 5-element model equations (if Cprox is used)
if 'Cprox' in params:
    doc.add_section("Modèle à 5 Éléments (Rc, Rp, C, L, Cprox)", "Le modèle 5-éléments ajoute une compliance proximale Cprox.", level=2)
    # Use "prox" in quotes for text subscript in Typst
    doc.add_equation("C_(\"prox\") ((d P_(\"prox\")) / (d t)) = Q(t) - (P_(\"prox\") - P_2) / R_c", caption="Équation pour la compliance proximale", label="eq:5e_prox")
    doc.add_equation("C ((d P_2) / (d t)) = (P_(\"prox\") - P_2) / R_c - P_2 / R_p", caption="Équation pour la compliance distale", label="eq:5e_dist")
    doc.add_equation("P_1(t) = P_(\"prox\")(t) + L ((d Q) / (d t))", caption="Reconstruction de la pression aortique", label="eq:5e_recon")

# =========================================
# 3. PARAMÈTRES DU MODÈLE
# =========================================
doc.add_section("Paramètres du Modèle", "", level=1)

# Model parameters
model_data = [
    ["Paramètre", "Valeur", "Unité"],
    ["Résistance caractéristique (Rc)", f"{params['Rc']:.2e}", "Pa·s/m³"],
    ["Résistance périphérique (Rp)", f"{params['Rp']:.2e}", "Pa·s/m³"],
    ["Compliance distale (C)", f"{params['C']:.2e}", "m³/Pa"],
    ["Inertance (L)", f"{params['L']:.2e}", "Pa·s²/m³"],
]

# Add Cprox if present (5-element model)
if 'Cprox' in params:
    model_data.append(["Compliance proximale (Cprox)", f"{params['Cprox']:.2e}", "m³/Pa"])
    model_type = "5 éléments (avec Cprox)"
else:
    model_type = "4 éléments"

doc.add_table(
    model_data,
    headers=["Paramètre", "Valeur", "Unité"],
    caption=f"Paramètres du modèle Windkessel {model_type}",
    label="tab:parameters"
)

# Theoretical time constant
tau_theoretical = params['Rp'] * params['C']
doc.add_section("Constante de Temps Théorique", "", level=2)
tau_text = f"""
La constante de temps diastolique théorique est calculée par τ = Rp × C.

τ = {params['Rp']:.2e} × {params['C']:.2e} = {tau_theoretical:.4f} s
"""
doc.add_section(tau_text, "", level=3)

# =========================================
# 3. MÉTRIQUES DE VALIDATION
# =========================================
doc.add_section("Métriques de Validation", "", level=1)

# Validation metrics table
validation_data = [
    ["Métrique", "Valeur", "Seuil", "Statut"],
]

# NRMS error
nrms_status = "✅ PASS" if metrics['nrms'] <= 0.15 else "❌ FAIL"
validation_data.append([
    "NRMS error",
    f"{metrics['nrms']*100:.2f}%",
    "≤ 15%",
    nrms_status
])

# Peak timing
dt_peak_status = "✅ PASS" if abs(metrics['dt_peak_ms']) <= 50 else "❌ FAIL"
validation_data.append([
    "Peak time shift",
    f"{metrics['dt_peak_ms']:.2f} ms",
    "≤ 50 ms",
    dt_peak_status
])

# Tau error
if metrics['tau_error_pct'] is not None:
    tau_error_status = "✅ PASS" if metrics['tau_error_pct'] <= 20 else "❌ FAIL"
    validation_data.append([
        "Relative tau error",
        f"{metrics['tau_error_pct']:.2f}%",
        "≤ 20%",
        tau_error_status
    ])
else:
    validation_data.append([
        "Relative tau error",
        "N/A",
        "≤ 20%",
        "⚠️ fitting failed"
    ])

doc.add_table(
    validation_data,
    headers=["Métrique", "Valeur", "Seuil", "Statut"],
    caption="Métriques de validation du modèle",
    label="tab:metrics"
)

# =========================================
# 4. ANALYSE DÉTAILLÉE
# =========================================
doc.add_section("Analyse Détaillée", "", level=1)

# Diastolic time constant analysis
doc.add_section("Constante de Temps Diastolique", "", level=2)
tau_analysis = f"""
La constante de temps diastolique (τ) caractérise la décroissance exponentielle 
de la pression durante la diastole.

- τ simulé : {metrics['tau_sim_s']:.4f} s
- τ référence : {metrics['tau_ref_s']:.4f} s
"""
doc.add_section(tau_analysis, "", level=3)

# Waveform comparison
doc.add_section("Comparaison des Formes d'Onde", "", level=2)
waveform_text = f"""
La forme d'onde de pression simulée est comparée à la référence après 
mise à l'échelle affine pour s'affranchir des différences d'amplitude.

L'erreur NRMS de {metrics['nrms']*100:.2f}% indique un écart global 
{"acceptable" if metrics['nrms'] <= 0.15 else "significatif"} entre les deux signaux.
"""
doc.add_section(waveform_text, "", level=3)

# =========================================
# 5. VISUALISATIONS
# =========================================
doc.add_section("Visualisations", "", level=1)

# Waveform comparison figure
waveform_fig = results_dir / 'validation_waveform.png'
if waveform_fig.exists():
    doc.add_figure(
        "validation_waveform.png",
        caption="Comparaison des formes d'onde de pression - Simulé vs Référence",
        label="fig:waveform",
        width="80%"
    )

# =========================================
# 6. CONCLUSION
# =========================================
doc.add_section("Conclusion", "", level=1)

if metrics['validation_passed']:
    conclusion = """La validation du modèle Windkessel est ACCEPTÉE. Toutes les métriques sont dans les tolérances définies:
- L'erreur NRMS est inférieure à 15%
- Le décalage temporel du pic est inférieur à 50ms
- L'erreur relative sur τ est inférieure à 20%

Le modèle Windkessel avec les paramètres définis reproduit correctement la dynamique de pressurisation aortique."""
else:
    conclusion = """La validation du modèle Windkessel est REJETÉE. Une ou plusieurs métriques dépassent les tolérances définies:"""
    if metrics['nrms'] > 0.15:
        conclusion += f"- NRMS error: {metrics['nrms']*100:.2f}% > 15%\\n"
    if abs(metrics['dt_peak_ms']) > 50:
        conclusion += f"- Peak timing: {metrics['dt_peak_ms']:.2f} ms > 50 ms\\n"
    if metrics['tau_error_pct'] and metrics['tau_error_pct'] > 20:
        conclusion += f"- Tau error: {metrics['tau_error_pct']:.2f}% > 20%\\n"
    
    conclusion += """
Une optimisation des paramètres (Rc, Rp, C, L) pourrait être nécessaire 
pour améliorer la correspondance avec les données de référence.
"""

doc.add_section(conclusion, "", level=2)

# =========================================
# 7. COMPILATION PDF
# =========================================
print("Génération du rapport PDF...")

# Create report directory
report_dir = base_path / 'report'
report_dir.mkdir(exist_ok=True)

# Copy images to report folder
import shutil
waveform_src = results_dir / 'validation_waveform.png'
if waveform_src.exists():
    shutil.copy(waveform_src, report_dir / 'validation_waveform.png')

renderer = TypstRenderer()
renderer.compile_pdf(
    doc, 
    output_pdf=str(report_dir / "rapport_windkessel_validation.pdf")
)

print(f"Rapport généré: {report_dir / 'rapport_windkessel_validation.pdf'}")
