# Convertisseur VOF vers DPM

`VofToDpmConverter` est le convertisseur Python déterministe de foampilot. Il sélectionne les cellules selon `alpha >= alpha_threshold`, regroupe les cellules connectées et calcule chaque fragment avec le poids physique `alpha × V`.

```text
V_fragment = somme(alpha_i × V_i)
centre = somme(alpha_i × V_i × centre_i) / V_fragment
U_fragment = somme(alpha_i × V_i × U_i) / V_fragment
d_eq = (6 V_fragment / pi)^(1/3)
```

Le convertisseur lit les champs OpenFOAM ASCII, refuse explicitement les champs binaires non décodés et écrit des positions, des propriétés de fragments et un rapport JSON. Il ne modifie pas `alpha` et n’insère pas directement de parcels dans un cloud vivant.

```python
from foampilot.utilities.vof_to_dpm import VofToDpmConverter

converter = VofToDpmConverter(alpha_threshold=0.5)
fragments = converter.extract_case("case", time_directory="0.01")
outputs = converter.write_openfoam_outputs(fragments, "case/constant")
```

Pour l’installation et les commandes OpenFOAM 13, consulter le [guide complet](vof_to_dpm_openfoam13.md). Pour les limites et la vérification scientifique, consulter l’[audit technique](audit_implementation_vof_to_dpm.md).
