# À propos de foampilot

## Pourquoi créer foampilot ?

L’idée initiale est née du constat que **le coût d’apprentissage d’OpenFOAM est élevé**.  
Sa structure, organisée autour de nombreux dossiers et fichiers dictionnaires, peut être difficile à appréhender, à vérifier et à maintenir, en particulier dans un contexte de bureau d’études où les délais sont courts et les itérations fréquentes.

J’ai donc envisagé qu’une **surcouche Python orientée objet** pourrait faciliter cette compréhension :  
- centraliser l’information dans des fichiers Python lisibles,  
- refléter explicitement la structure d’un cas OpenFOAM,  
- réduire les erreurs liées à la manipulation manuelle des dictionnaires.

Grâce à l’écosystème Python, il devient possible de **piloter l’ensemble de la chaîne CFD**.  
L’objectif de foampilot est ainsi de proposer une **plateforme open‑source de calcul CFD**, couvrant tout le cycle de simulation :
- création de la géométrie,
- génération du maillage,
- configuration et exécution du solveur,
- post‑traitement,
- génération de rapports.

---

## Qui suis‑je ?

Je me nomme **Steven Daix** et je pratique la CFD depuis plus de **20 ans**.

J’ai travaillé dans différents secteurs d’activité :
- nucléaire,
- automobile,
- oil & gas,
- bâtiment,

au sein de structures de tailles variées, allant de la startup aux grands groupes, en passant par des bureaux d’études.

Je suis principalement utilisateur de **Fluent (ANSYS)** et **STAR‑CCM+**.  
Mon expérience couvre des projets très divers, notamment :
- études de tirage thermique dans des bâtiments,
- optimisation thermique dans des musées,
- mélange dans des bains de verre agités,
- études aérodynamiques sous capot de tracteur,
- optimisation de procédés industriels, comme le séchage de papier toilette (fun fact).

La difficulté d’intégrer OpenFOAM dans un contexte de bureau d’études — notamment pour les vérifications rapides et les demandes urgentes — m’a longtemps empêché d’utiliser des outils open‑source en production.

C’est dans ce contexte que j’ai décidé de développer **foampilot**, sur mon temps libre, en m’appuyant sur mon expérience CFD et sur la manière dont j’aurais souhaité utiliser OpenFOAM au quotidien.

---

## Rôle de l’intelligence artificielle

Le développement de foampilot a été **assisté par plusieurs outils d’intelligence artificielle**, notamment :
- ChatGPT,
- Gemini,
- Mistral,
- DeepSeek,
- Manus.

Ces IA ont été utilisées comme **outils d’assistance** pour :
- la structuration du code,
- la clarification de concepts,
- l’amélioration de la documentation,
- la reformulation et la pédagogie.

Les choix techniques, l’architecture globale, les concepts CFD et la vision du projet restent **guidés par mon expérience d’ingénieur CFD**.  
Les IA sont ici des **accélérateurs de réflexion et de productivité**, et non des substituts à l’expertise métier.

---

## Objectif du projet

L’objectif de foampilot est de proposer une **interface Python claire, reproductible et automatisée** pour OpenFOAM, permettant :
- de fiabiliser les études CFD,
- de faciliter les audits et vérifications,
- de rendre OpenFOAM plus accessible en environnement industriel,
- tout en restant fidèle aux concepts fondamentaux de la CFD et d’OpenFOAM.

> foampilot est avant tout un projet d’ingénieur, conçu par un utilisateur pour des utilisateurs.
