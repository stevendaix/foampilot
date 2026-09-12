# Brief — Semaine 6 : "Rendre un outil CFD adoptable"

**Objectif** : Expliquer comment présenter et documenter un projet open-source scientifique pour maximiser l'adoption.
**Angle** : Communication, open-source, documentation, communauté.

---

## Structure proposée

### Introduction — L'outil invisible

Storytelling : Vous avez passé 6 mois à construire un outil formidable. Il automatise 90% de votre workflow CFD. Vous le partagez sur GitHub. Six mois plus tard : 3 stars, 0 issue, 0 contributeur.

**Le problème** : Un outil scientifique ne vit pas seulement par sa qualité technique. Il vit par sa présentation.

**Promesse** : Les 5 leçons que j'ai tirées de foampilot pour rendre un outil adoptable.

### Section 1 — Le README comme promesse, pas comme documentation

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

### Section 2 — La documentation multilingue comme signal

**Leçon 2** : Si votre outil est utilisé mondialement, traduisez la vitrine.

- OpenFOAM est global, mais beaucoup d'utilisateurs ne sont pas anglophones
- Un README en 3 langues (EN/FR/ZH) envoie un message : *"ce projet est fait pour vous"*
- La traduction n'a pas à être parfaite — elle doit être compréhensible

### Section 3 — Les exemples comme argument de vente

**Leçon 3** : Ne dites pas que votre outil est puissant. Montrez-le.

- Le répertoire `examples/` est votre meilleur argument
- Chaque exemple répond à une question : "est-ce que ça peut faire ça ?"
- Un exemple complet > 10 pages de documentation
- Les captures d'écran et résultats visuels sont des preuves

**Conseil** : Pour chaque exemple, ajoutez un `README.md` qui explique le cas, les paramètres, et le résultat attendu.

### Section 4 — La documentation technique comme mémoire

**Leçon 4** : Un projet open-source a deux publics : les utilisateurs et les contributeurs.

- Documentation utilisateur : tutoriels, guides, FAQ
- Documentation contributeur : architecture, conventions, workflow de dev
- MkDocs ou Sphinx pour structurer
- Liens croisés entre les pages

### Section 5 — Les tests comme preuve de maturité

**Leçon 5** : Dans le monde scientifique, la reproductibilité est reine.

- Un projet sans tests est un projet non fiable
- Les tests documentent le comportement attendu
- Les CI/CD (GitHub Actions) prouvent que ça marche sur tout environnement

Montrer l'exemple de `AGENTS.md` dans foampilot : il documente comment exécuter les tests, ce qui est rare dans les projets scientifiques.

### Section 6 — La communauté comme levier

**Leçon 6** : Votre première communauté est votre première audience.

- Répondez aux issues rapidement (même pour dire "c'est un bug, merci")
- Écrivez des release notes pour chaque version
- Utilisez GitHub Discussions pour les questions générales
- Mettez en avant les contributeurs dans le README

### Conclusion — L'outil comme produit de communication

**Message final** : Un projet scientifique open-source est un produit à deux faces :
1. La face technique (code, algorithmes, rigueur)
2. La face communication (README, exemples, documentation, démo)

La plupart des chercheurs excellent sur la première. Peu investissent suffisamment la seconde.

**CTA** : *"Si vous avez construit un outil CFD qui mérite d'être partagé, publiez-le. Mais publiez-le avec la même rigueur que votre code."*

---

## Code à préparer

- [ ] Extrait du README.md (section promesse)
- [ ] Extrait de la structure `docs/` (MkDocs)
- [ ] Extrait de `AGENTS.md` (conventions de dev)
- [ ] Extrait de `test/` (exemple de test)

## Images à préparer

- [ ] Schéma de la structure du projet (`src/`, `docs/`, `examples/`, `test/`)
- [ ] Capture d'écran du README.md (sur GitHub)
- [ ] Capture d'écran du site de documentation (MkDocs)
- [ ] Capture d'écran des GitHub Actions (si présentes)
- [ ] Schéma "2 faces d'un projet open-source"

## Ressources à citer

- "The Cathedral and the Bazaar" (Eric Raymond)
- "Don't Make Me Think" (Steve Krug) — pour le README
- "Clean Code" (Robert Martin) — pour les tests
