# Sources externes enregistrées

## Article fourni par l’utilisateur

Source locale : `/home/ubuntu/upload/pasted_content.txt`

Référence : Yoshito Takahashi, Akihisa Nomoto, Shu Yoda, Ryo Hisayama, Masayuki Ogata, Yoshiichi Ozeki, Shin-ichi Tanabe, “Thermoregulation model JOS-3 with new open source code”, Energy & Buildings 231 (2021), article 110575, DOI: https://doi.org/10.1016/j.enbuild.2020.110575.

L’extrait indique que JOS-3 est dérivé de JOS-2, comprend 83 nœuds et calcule les réponses physiologiques et les températures par une méthode de différence arrière. Les extensions annoncées sont l’activité du tissu adipeux brun, les effets du vieillissement et le gain solaire de courte longueur d’onde à la peau. Les méthodes du frisson, de la distribution de la sudation et du métabolisme basal sont modifiées par rapport à JOS-2.

L’article rapporte une validation sur des essais humains stables et transitoires. Pour neuf conditions transitoires, les RMSE annoncées entre prédictions et expériences sont de 0,12–0,38 °C pour la température rectale et de 0,58–0,83 °C pour la température cutanée moyenne. L’introduction rappelle que les modèles calculent la conduction et le débit sanguin dans les tissus, ainsi que la convection, l’évaporation et le rayonnement entre le corps et l’ambiance. Les réponses physiologiques incluent notamment sudation, vasodilatation, vasoconstriction et frisson.

## Sources web consultées

1. Article ScienceDirect : https://www.sciencedirect.com/science/article/pii/S0378778820333612
2. Dépôt officiel JOS-3 : https://github.com/TanabeLab/JOS-3
3. Documentation OpenFOAM externalCoupled : https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/externalCoupled/
4. OpenFOAM 13 : https://openfoam.org/version/13/

La documentation OpenFOAM externalCoupled décrit un échange texte avec une ligne par face, des fichiers `.out` écrits par OpenFOAM, des fichiers `.in` écrits par l’application externe, et un fichier `OpenFOAM.lock` utilisé pour synchroniser le cycle. La géométrie collectée est stockée dans `patchFaces` et `patchPoints`.
