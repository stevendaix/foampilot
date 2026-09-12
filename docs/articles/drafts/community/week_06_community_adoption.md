# Rendre un outil CFD adoptable : leçons tirées de 6 mois de foampilot sur GitHub

*Vous avez passé 6 mois à construire un outil formidable. Vous le partagez sur GitHub. Six mois plus tard : 3 stars, 0 issue, 0 contributeur. Le problème ? Un projet scientifique open-source est un produit à deux faces.*

---

## Introduction : L'outil invisible

Storytelling : Vous avez passé 6 mois à construire un outil formidable. Il automatise 90% de votre workflow CFD. Vous le partagez sur GitHub. Six mois plus tard : 3 stars, 0 issue, 0 contributeur.

**Le problème** : Un projet scientifique open-source est un produit à deux faces :
1. La face technique (code, algorithmes, rigueur)
2. La face communication (README, exemples, documentation, démo)

La plupart des chercheurs excellent sur la première. Peu investissent suffisamment la seconde.

**Promesse** : Les 5 leçons que j'ai tirées de foampilot pour rendre un outil adoptable.

---

## Leçon 1 : Le README comme promesse, pas comme documentation

**Leçon 1** : Le README doit répondre à 3 questions en 30 secondes :
1. Qu'est-ce que c'est ?
2. À quoi ça sert ?
3. Pourquoi c'est mieux que l'alternative ?

Montrer l'exemple de foampilot :
- Titre + emoji
- Elevator pitch
- Features avec verbes d'action
- Section "Ce que ce n'est pas" pour éviter les malentendus

**Erreur à éviter** : Un README de 200 lignes qui explique l'architecture avant d'expliquer l'usage.

---

## Leçon 2 : La documentation multilingue comme signal

**Leçon 2** : Si votre outil est utilisé mondialement, traduisez la vitrine.

- OpenFOAM est global, mais beaucoup d'utilisateurs ne sont pas anglophones
- Un README en 3 langues (EN/FR/ZH) envoie un message : *"ce projet est fait pour vous"*
- La traduction n'a pas à être parfaite — elle doit être compréhensible

---

## Leçon 3 : Les exemples comme argument de vente

**Leçon 3** : Ne dites pas que votre outil est puissant. Montrez-le.

- Le répertoire `examples/` est votre meilleur argument
- Chaque exemple répond à une question : "est-ce que ça peut faire ça ?"
- Un exemple complet > 10 pages de documentation
- Les captures d'écran et résultats visuels sont des preuves

**Conseil** : Pour chaque exemple, ajoutez un `README.md` qui explique le cas, les paramètres, et le résultat attendu.

---

## Leçon 4 : La documentation technique comme mémoire

**Leçon 4** : Un projet open-source a deux publics : les utilisateurs et les contributeurs.

- Documentation utilisateur : tutoriels, guides, FAQ
- Documentation contributeur : architecture, conventions, workflow de dev
- MkDocs ou Sphinx pour structurer
- Liens croisés entre les pages

---

## Leçon 5 : Les tests comme preuve de maturité

**Leçon 5** : Dans le monde scientifique, la reproductibilité est reine.

- Un projet sans tests est un projet non fiable
- Les tests documentent le comportement attendu
- Les CI/CD (GitHub Actions) prouvent que ça marche sur tout environnement

Montrer l'exemple de `AGENTS.md` dans foampilot : il documente comment exécuter les tests, ce qui est rare dans les projets scientifiques.

---

## Leçon 6 : La communauté comme levier

**Leçon 6** : Votre première communauté est votre première audience.

- Répondez aux issues rapidement (même pour dire "c'est un bug, merci")
- Écrivez des release notes pour chaque version
- Utilisez GitHub Discussions pour les questions générales
- Mettez en avant les contributeurs dans le README

---

## Conclusion — L'outil comme produit de communication

**Message final** : Un projet scientifique open-source est un produit à deux faces :
1. La face technique (code, algorithmes, rigueur)
2. La face communication (README, exemples, documentation, démo)

La plupart des chercheurs excellent sur la première. Peu investissent suffisamment la seconde.

**CTA** : *"Si vous avez construit un outil CFD qui mérite d'être partagé, publiez-le. Mais publiez-le avec la même rigueur que votre code."*

---

*Article en cours d'amélioration — version 1.0*
