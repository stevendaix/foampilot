#!/usr/bin/env python3
"""Validation minimale du couplage OpenFOAM–JOS-3.

Exécution depuis la racine du dépôt FoamPilot :

    PYTHONPATH=foampilot/src:../JOS-3/src python validate_coupling.py

Dans un cas OpenFOAM réel, remplacer exchange_arrays(...) par
exchange_from_openfoam(...), après avoir produit les champs point h, Ta et T.
"""

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np

# Autorise l'exécution directe depuis examples/thermoregulation/openfoam_jos3_coupling.
HERE = Path(__file__).resolve()
FOAMPILOT_SRC = HERE.parents[3] / "foampilot" / "src"
JOS3_SRC = HERE.parents[4] / "JOS-3" / "src"
for path in (FOAMPILOT_SRC, JOS3_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jos3

# Le paquet FoamPilot complet charge aussi ses extensions CAD. Pour que ce
# petit test de calcul fonctionne sur une installation minimale, on charge
# le module ciblé avec un lecteur OpenFOAM substituable ; l’installation
# complète utilise automatiquement OpenFOAMDirectReader réel.
_openfoam_stub = types.ModuleType("foampilot.postprocess.openfoam_direct")
_openfoam_stub.OpenFOAMDirectReader = object
sys.modules.setdefault("foampilot", types.ModuleType("foampilot"))
postprocess_pkg = types.ModuleType("foampilot.postprocess")
postprocess_pkg.__path__ = []
sys.modules["foampilot.postprocess"] = postprocess_pkg
sys.modules["foampilot.postprocess.openfoam_direct"] = _openfoam_stub
_spec = importlib.util.spec_from_file_location(
    "foampilot.postprocess.jos3_openfoam",
    FOAMPILOT_SRC / "foampilot" / "postprocess" / "jos3_openfoam.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
OpenFOAMJOS3Coupler = _module.OpenFOAMJOS3Coupler


def main() -> None:
    n_nodes = 34
    segment_ids = np.arange(n_nodes) % 17
    node_areas = np.full(n_nodes, 0.01)
    h = np.full(n_nodes, 8.0)
    ta = np.full(n_nodes, 22.0)

    # Cas de référence JOS-3 sans flux CFD externe.
    reference = jos3.JOS3(ex_output="all")
    reference.To = 22.0
    reference.RH = 50.0
    reference.Va = 0.1
    reference.PAR = 1.25
    reference.simulate(times=5, dtime=60)

    # Cas couplé : T = Ta implique q = h(T-Ta) = 0 ; les résultats doivent coïncider.
    coupled = jos3.JOS3(ex_output="all")
    coupled.To = 22.0
    coupled.RH = 50.0
    coupled.Va = 0.1
    coupled.PAR = 1.25
    adapter = OpenFOAMJOS3Coupler(coupled, segment_ids, node_areas)
    exchange = adapter.exchange_arrays(h, ta, ta, time_step="0", apply=True)
    coupled.simulate(times=5, dtime=60)

    np.testing.assert_allclose(exchange.outward_flux, 0.0, atol=1e-14)
    np.testing.assert_allclose(exchange.segment_heat, 0.0, atol=1e-14)
    np.testing.assert_allclose(reference.Tsk, coupled.Tsk, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(reference.Tcr, coupled.Tcr, rtol=0.0, atol=1e-12)

    # Vérification indépendante de l’échange : flux entrant de 40 W/m2.
    hot_surface = ta + 5.0
    exchange = adapter.exchange_arrays(h, ta, hot_surface, apply=False)
    expected = np.zeros(17)
    np.add.at(expected, segment_ids, -h * 5.0 * node_areas)
    np.testing.assert_allclose(exchange.segment_heat, expected)

    out = HERE.parent / "results"
    out.mkdir(exist_ok=True)
    adapter.write_point_scalar_field(
        exchange.body_flux,
        out / "qJOS3",
        field_name="qJOS3",
        time_name="0",
    )
    print("Validation JOS-3/OpenFOAM réussie.")
    print(f"Flux segmenté exemple [W] : {exchange.segment_heat.tolist()}")
    print(f"Champ retour OpenFOAM : {out / 'qJOS3'}")


if __name__ == "__main__":
    main()
