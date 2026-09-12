# Template d'article Medium — Format standardisé

Copier ce template pour chaque nouvel article. Respecter la structure et le ton pour créer une ligne éditoriale cohérente.

---

## Métadonnées (à compléter avant rédaction)

- **Titre** :
- **Sous-titre** (optionnel) :
- **Mots-clés** (3-5) :
- **Semaine #** :
- **Public cible** :
- **Angle** :
- **Temps de lecture estimé** :
- **CTA (call to action)** : 

---

## Structure standard

### 1. Titre et illustration

- Titre en **gras**, accrocheur, < 70 caractères si possible
- Sous-titre optionnel pour préciser l'angle
- Image d'en-tête (capture d'écran, schéma, ou photo libre de droits)

### 2. Introduction — Le hook (paragraphe 1-2)

**Règle d'or** : La première phrase doit répondre à "pourquoi devrais-je lire cet article ?"

Formule à utiliser :
- Problème concret que le lecteur reconnaît
- Promesse de solution
- Preview de ce qu'il va apprendre

Exemple :
> *"Si vous avez déjà édité un fichier `controlDict` à 2h du matin pour corriger une simulation qui refuse de démarrer, cet article est pour vous."*

### 3. Section 1 — Le contexte / Le problème

- Expliquer le problème en termes simples
- Utiliser des métaphores concrètes (éviter le jargon excessif)
- Montrer la douleur du lecteur : frustration, temps perdu, erreurs répétitives
- **Pas de code ici** — juste du storytelling

### 4. Section 2 — La solution / Le concept

- Introduire la solution (foampilot, un pattern, une technique)
- **Un seul concept par section**
- Alterner texte et blocs de code
- Chaque bloc de code doit avoir :
  - Un commentaire en français expliquant ce qu'il fait
  - Un commentaire en anglais (optionnel) pour les termes techniques
  - Une phrase après le bloc expliquant le résultat

### 5. Section 3 — L'exemple concret

- Code réel, minimaliste, qui fonctionne
- Output attendu (capture d'écran ou texte)
- Comparaison avant/après si possible
- Numéros de ligne et références au repo si pertinent

### 6. Section 4 — Le conseil / La leçon

- Tirer une leçon générale du cas concret
- Donner un conseil actionnable
- Connecter au problème initial
- Éviter le "et en conclusion..." → préférer une phrase forte qui clôt

### 7. Conclusion et CTA

- Résumer en 2-3 phrases
- CTA clair : vers quel article de la série, quel outil, quelle action
- Lien vers le repo GitHub, la documentation, les exemples
- Invitation à commenter / partager

---

## Règles de rédaction

### Ton
- Professionnel mais accessible
- Didactique : expliquer comme à un collègue compétent mais pas expert
- Enthousiaste sans être promotionnel
- Éviter le "je" excessif → préférer "on", "nous", ou le impersonnel
- Éviter le tutoiement

### Code
- Blocs de code avec langage `python` systématiquement
- Indentation 4 espaces
- Commentaires concis
- Pas de code inutile — chaque ligne doit servir l'explication
- Si le code est long (> 20 lignes), extraire l'essentiel et renvoyer vers le repo

### Formatting Medium
- Titres : `##` pour les sections, `###` pour les sous-sections
- Listes à puces pour les concepts clés
- **Gras** pour les termes importants (première occurrence)
- _Italique_ pour les nuances
- Blocs de citation `>` pour les leçons importantes
- Séparateurs `---` entre les sections principales
- Pas de tableaux complexes (mal supportés sur Medium mobile)

### Images
- Privilégier des schémas simples (draw.io, Excalidraw) plutôt que des captures d'écran
- Si capture d'écran : annoter avec des flèches et des numéros
- Légendes systématiques sous chaque image
- Ratio 16:9 pour les schémas, 1:1 pour les captures

### Longueur
- **Cible** : 1200-1800 mots
- **Minimum** : 900 mots
- **Maximum** : 2500 mots (sauf article long format)

---

## Checklist pré-publication

### Contenu
- [ ] Titre accrocheur et < 70 caractères
- [ ] Introduction qui répond à "pourquoi lire cet article ?"
- [ ] Un seul concept par section
- [ ] Exemples de code commentés et testés
- [ ] Leçon ou conseil actionnable
- [ ] CTA clair en fin d'article

### Technique
- [ ] Code exécuté et vérifié
- [ ] Liens internes vers le repo fonctionnels
- [ ] Images optimisées (< 500 KB chacune)
- [ ] Tags Medium ajoutés (3-5)
- [ ] Publication programmée ou immédiate

### Editorial
- [ ] Relecture orthographique (français)
- [ ] Pas de jargon excessif sans explication
- [ ] Ton cohérent avec les articles précédents
- [ ] Pas de contenu promotionnel excessif

---

## Publication

1. **Sauvegarder** l'article dans `articles/week_XX_topic.md`
2. **Copier-coller** dans l'éditeur Medium
3. **Ajouter** les images et le formatage inline
4. **Vérifier** la prévisualisation mobile
5. **Publier** avec les tags appropriés
6. **Partager** sur Twitter/LinkedIn avec un extrait accrocheur

---

## Exemple d'article structuré

```markdown
# Titre accrocheur en gras

*Sous-titre optionnel qui précise l'angle.*

---

## Introduction

[Storytelling, problème, promesse]

## Le problème que vous connaissez

[Contexte, douleur, frustration]

## La solution : [nom du concept]

[Explication simple, métaphore]

```python
# Code exemple
```

[Explication du résultat]

## L'exemple concret

[Code complet, output, comparaison]

## La leçon à retenir

[Conseil actionnable]

## Conclusion

[Résumé, CTA, liens]
```
