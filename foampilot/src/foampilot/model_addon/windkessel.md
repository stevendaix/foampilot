
# Modèle Windkessel : Théorie, Implémentation et Validation

## Table des Matières

1. [Introduction aux Modèles Cardiovasculaires](#introduction)
2. [Théorie du Modèle Windkessel](#theorie)
   - [2.1 Analogie Électrique-Hydraulique](#analogie)
   - [2.2 Modèle à 2 Éléments](#windkessel-2e)
   - [2.3 Modèle à 3 Éléments](#windkessel-3e)
   - [2.4 Modèle à 4 Éléments](#windkessel-4e)
3. [Formulation Mathématique](#mathematiques)
4. [Implémentation Numérique](#implementation)
5. [Validation du Modèle](#validation)
6. [Guide d'Utilisation](#utilisation)
7. [Annexes](#annexes)

---

## 1. Introduction aux Modèles Cardiovasculaires {#introduction}

### Contexte Physiologique

Le système cardiovasculaire est un système complexe de pompes (cœur) et de conduits (vaisseaux) qui distribue le sang oxygéné à l'ensemble de l'organisme. La modélisation de ce système est essentielle pour :

- **Comprendre** la physiologie cardiovasculaire
- **Diagnostiquer** des pathologies (hypertension, insuffisance cardiaque)
- **Optimiser** des dispositifs médicaux (prothèses valvulaires, assistance ventriculaire)
- **Prédire** les effets de traitements pharmacologiques

### Pourquoi un Modèle "Lumped" ?

Les modèles **lumped-parameter** (à paramètres localisés) simplifient le système cardiovasculaire en le représentant comme un circuit électrique équivalent :

| Domaine Hydraulique | Domaine Électrique | Unités SI |
|---------------------|-------------------|-----------|
| Pression $P$        | Tension $V$       | Pa (ou mmHg) |
| Débit $Q$           | Courant $I$       | m³/s |
| Résistance $R$      | Résistance $R$    | Pa·s/m³ |
| Compliance $C$      | Capacitance $C$   | m³/Pa |
| Inertance $L$       | Inductance $L$    | Pa·s²/m³ |

**Avantages** :
- Calcul rapide (quelques millisecondes)
- Peu de paramètres à identifier
- Interprétation physiologique claire

**Limites** :
- Ne capture pas les effets de propagation d'onde
- Simplification géométrique importante

---

## 2. Théorie du Modèle Windkessel {#theorie}

### 2.1 Analogie Électrique-Hydraulique {#analogie}

Le modèle Windkessel (de l'allemand "chambre à air") a été introduit par **Otto Frank** en 1899 pour décrire la relation pression-débit dans l'aorte.

#### Composants de Base

**1. Résistance ($R$)**
```
P1 ----[R]---- P2
```
- **Loi** : $P_1 - P_2 = R \cdot Q$
- **Physiologie** : Frottements visqueux dans les vaisseaux
- **Dépendance** : Rayon du vaisseau ($R \propto 1/r^4$, loi de Poiseuille)

**2. Compliance ($C$)**
```
      |
P ----| C
      |
```
- **Loi** : $Q = C \cdot \frac{dP}{dt}$
- **Physiologie** : Élasticité des parois artérielles
- **Effet** : Stockage d'énergie pendant la systole, restitution en diastole

**3. Inertance ($L$)**
```
P1 ----[L]---- P2
```
- **Loi** : $P_1 - P_2 = L \cdot \frac{dQ}{dt}$
- **Physiologie** : Inertie du sang en accélération
- **Importance** : Significative dans les grosses artères et à haute fréquence

---

### 2.2 Modèle à 2 Éléments (Windkessel Classique) {#windkessel-2e}

#### Schéma du Circuit

```
        Q(t)
    ──────────►
         │
         ├───[Rp]───┐
         │          │
        [C]        P(t)
         │          │
         ┴          ┴
```

#### Équation Différentielle

**Bilan de débit au nœud :**
$$Q(t) = Q_C + Q_R$$

où :
- $Q_C = C \frac{dP}{dt}$ (débit dans la compliance)
- $Q_R = \frac{P}{R_p}$ (débit dans la résistance périphérique)

**Équation finale :**
$$\boxed{C \frac{dP}{dt} + \frac{P}{R_p} = Q(t)}$$

#### Solution Analytique (pour Q constant)

Pour un débit constant $Q_0$ et condition initiale $P(0) = P_0$ :

$$P(t) = R_p Q_0 + (P_0 - R_p Q_0) e^{-t/\tau}$$

avec la **constante de temps diastolique** :
$$\tau = R_p \cdot C$$

**Interprétation physiologique** :
- $\tau \approx 1.5 - 2.0$ s pour un adulte sain
- Détermine la pente de décroissance de pression en diastole
- Diminué dans l'hypertension artérielle

#### Limites du Modèle 2É

❌ **Problèmes majeurs** :
1. Impédance infinie à haute fréquence (non physiologique)
2. Ne prédit pas l'augmentation de pression systolique avec l'âge
3. Forme d'onde trop lissée

---

### 2.3 Modèle à 3 Éléments (Windkessel Modifié) {#windkessel-3e}

#### Schéma du Circuit

```
        Q(t)
    ──────────►
         │
        [Rc]
         │
         ├───[Rp]───┐
         │          │
        [C]        P2(t)
         │          │
         ┴          ┴
         
P1(t) ◄─────────────┘
```

**Nouveauté** : Ajout d'une **résistance caractéristique $R_c$** en série, représentant l'impédance de l'aorte proximale.

#### Système d'Équations

**Variables** :
- $P_1(t)$ : Pression aortique (mesurée cliniquement)
- $P_2(t)$ : Pression après $R_c$ (aux bornes de $C$)

**Équations** :

1. **Compliance** (nœud après $R_c$) :
   $$C \frac{dP_2}{dt} + \frac{P_2}{R_p} = Q(t)$$

2. **Reconstruction de $P_1$** (algébrique) :
   $$\boxed{P_1(t) = P_2(t) + R_c \cdot Q(t)}$$

#### Avantages par rapport au 2É

✅ **Améliorations** :
1. Impédance finie à haute fréquence : $Z(\omega \to \infty) = R_c$
2. Meilleure reproduction de la pression systolique
3. Séparation des effets résistifs (Rp) et pulsatoires (Rc)

**Ordres de grandeur typiques** :
- $R_c \approx 0.01 - 0.1 \cdot R_p$ (environ 5-10% de Rp)
- $R_p \approx 1.0 - 2.0$ mmHg·s/mL ($\approx 1.3-2.6 \times 10^8$ Pa·s/m³)
- $C \approx 1.0 - 2.0$ mL/mmHg ($\approx 1.3-2.6 \times 10^{-9}$ m³/Pa)

---

### 2.4 Modèle à 4 Éléments (avec Inertance) {#windkessel-4e}

#### Schéma du Circuit

```
        Q(t)
    ──────────►
         │
        [Rc]
         │
        [L]
         │
         ├───[Rp]───┐
         │          │
        [C]        P2(t)
         │          │
         ┴          ┴
         
P1(t) ◄─────────────┘
```

**Nouveauté** : Ajout d'une **inertance $L$** en série avec $R_c$, représentant l'inertie du sang.

#### Système d'Équations Complet

1. **Équation différentielle** (identique au 3É) :
   $$C \frac{dP_2}{dt} + \frac{P_2}{R_p} = Q(t)$$

2. **Reconstruction de $P_1$** (avec terme inertiel) :
   $$\boxed{P_1(t) = P_2(t) + R_c \cdot Q(t) + L \cdot \frac{dQ}{dt}(t)}$$

#### Interprétation du Terme Inertiel

Le terme $L \cdot \frac{dQ}{dt}$ représente :
- **Accélération** du sang pendant la systole ($dQ/dt > 0$) → augmentation de $P_1$
- **Décélération** en fin de systole ($dQ/dt < 0$) → diminution de $P_1$

**Effet visible** : 
- Pic de pression plus aigu
- Incisure dicrote plus marquée

**Ordre de grandeur** :
- $L \approx 10^3 - 10^5$ Pa·s²/m³
- Souvent négligé dans les applications cliniques (sauf pathologies spécifiques)

---

## 3. Formulation Mathématique {#mathematiques}

### 3.1 Problème aux Valeurs Initiales

Le modèle Windkessel se résout comme un **problème de Cauchy** :

$$\begin{cases}
\displaystyle \frac{dP_2}{dt} = \frac{Q(t)}{C} - \frac{P_2}{R_p C}, & t \in [0, T] \\
P_2(0) = P_{2,0} & \text{(condition initiale)}
\end{cases}$$

### 3.2 Solution Générale

L'équation différentielle linéaire du premier ordre admet la solution :

$$P_2(t) = e^{-t/\tau} \left[ P_{2,0} + \frac{1}{C} \int_0^t Q(s) e^{s/\tau} ds \right]$$

où $\tau = R_p C$ est la constante de temps.

**Cas périodique** : Si $Q(t)$ est périodique de période $T$, après un transitoire exponentiel, $P_2(t)$ devient périodique.

### 3.3 Analyse de Stabilité

**Condition de stabilité** : Le système est **asymptotiquement stable** car :
- La constante de temps $\tau = R_p C > 0$ (paramètres physiques positifs)
- La solution homogène $P_2^h(t) = P_{2,0} e^{-t/\tau}$ décroît exponentiellement

**Temps de convergence** vers le régime permanent :
$$t_{95\%} \approx 3\tau = 3 R_p C$$

Pour $\tau \approx 1.5$ s → convergence en ~4.5 s (3-4 cycles cardiaques)

### 3.4 Réponse en Fréquence (Analyse Harmonique)

Pour une entrée sinusoïdale $Q(t) = Q_0 e^{i\omega t}$, l'impédance d'entrée $Z(\omega) = \frac{P_1(\omega)}{Q(\omega)}$ vaut :

**Modèle 3É** :
$$Z_{3E}(\omega) = R_c + \frac{R_p}{1 + i\omega R_p C}$$

**Module** :
$$|Z_{3E}(\omega)| = \sqrt{R_c^2 + \frac{R_p^2}{1 + (\omega \tau)^2} + \frac{2 R_c R_p}{1 + (\omega \tau)^2}}$$

**Limites** :
- $\omega \to 0$ (continu) : $|Z| \to R_c + R_p$
- $\omega \to \infty$ : $|Z| \to R_c$

**Modèle 4É** :
$$Z_{4E}(\omega) = R_c + i\omega L + \frac{R_p}{1 + i\omega R_p C}$$

---

## 4. Implémentation Numérique {#implementation}

### 4.1 Architecture de la Classe `Windkessel`

```python
class Windkessel:
    """
    Modèle Windkessel avec inertance série.
    
    Attributs principaux :
    - Rc, Rp, C, L : paramètres physiques
    - _Q_spline : interpolation cubique du débit
    - periodic : gestion du signal périodique
    """
```

### 4.2 Interpolation du Débit par Spline Cubique

**Problème** : Les données de débit sont discrètes ($t_i, Q_i$), mais le solveur ODE nécessite $Q(t)$ continu et dérivable.

**Solution** : Interpolation par **spline cubique** $C^2$ :

```python
from scipy.interpolate import CubicSpline

bc_type = "periodic" if periodic else "natural"
self._Q_spline = CubicSpline(t_flow, q_flow, bc_type=bc_type)
```

**Avantages** :
- Continuité $C^2$ (nécessaire pour calculer $dQ/dt$ sans bruit)
- Condition périodique : $Q(0) = Q(T)$, $Q'(0) = Q'(T)$
- Évaluation rapide ($O(1)$ par point)

**Dérivée** :
```python
def dQdt(self, t):
    return self._Q_spline(t, 1)  # ordre=1 pour la dérivée première
```

### 4.3 Résolution de l'EDO avec `solve_ivp`

**Méthode** : Runge-Kutta explicite d'ordre 5(4) (RK45)

```python
from scipy.integrate import solve_ivp

sol = solve_ivp(
    fun=self._rhs,           # fonction f(t, y)
    t_span=(t_start, t_end), # intervalle de temps
    y0=[p2_init],            # condition initiale
    t_eval=t_eval,           # points de sortie
    method="RK45",           # schéma numérique
    rtol=1e-6,               # tolérance relative
    atol=1e-9,               # tolérance absolue
)
```

**Choix du pas de temps** :
- Adaptatif (contrôlé par les tolérances)
- Typiquement $\Delta t \approx 0.1 - 1$ ms pour un cycle cardiaque de 800 ms

**Estimation de la condition initiale** :

Pour réduire le transitoire, on initialise $P_2$ près du régime permanent :

```python
def estimate_steady_state_p2(self):
    """Approximation quasi-statique : P2 ≈ Rp * Q"""
    q_samples = self.Q(np.linspace(0, T, 100))
    return np.mean(self.Rp * q_samples)
```

### 4.4 Reconstruction Algébrique de $P_1$

Une fois $P_2(t)$ calculé, on obtient $P_1(t)$ **sans intégration supplémentaire** :

```python
def p1_from_p2(self, t, p2):
    """P1 = P2 + Rc*Q + L*dQ/dt"""
    return p2 + self.Rc * self.Q(t) + self.L * self.dQdt(t)
```

**Avantage** : Évite l'accumulation d'erreurs numériques.

### 4.5 Gestion de la Périodicité

Pour simuler plusieurs cycles cardiaques :

```python
def _wrap_time(self, t):
    """Ramène t dans [0, T] par modulo"""
    if not self.periodic:
        return t
    return np.mod(t, self._t_max)
```

**Application** :
- Permet d'évaluer $Q(t)$ pour $t > T$ sans extrapolation
- Essentiel pour les simulations longues (5-10 cycles)

---

## 5. Validation du Modèle {#validation}

### 5.1 Objectifs de la Validation

Valider un modèle Windkessel consiste à vérifier que :

1. **Morphologie** : La forme d'onde simulée ressemble à la référence
2. **Timing** : Les pics systoliques sont alignés temporellement
3. **Diastole** : La décroissance exponentielle est correcte
4. **Amplitude** : Les pressions systolique/diastolique sont réalistes

### 5.2 Métriques de Validation

#### 5.2.1 Erreur RMS Normalisée (NRMS)

$$\text{NRMS} = \frac{\|P_{\text{sim}} - P_{\text{ref}}\|_2}{\|P_{\text{ref}}\|_2} = \sqrt{\frac{\sum_{i=1}^n (P_{\text{sim},i} - P_{\text{ref},i})^2}{\sum_{i=1}^n P_{\text{ref},i}^2}}$$

**Interprétation** :
- NRMS < 5% : Excellent accord
- NRMS < 10% : Bon accord
- NRMS < 15% : Acceptable pour un modèle lumped
- NRMS > 20% : Problème de calibration

**Avantage** : Sans dimension, comparable entre différents patients.

#### 5.2.2 Décalage Temporel du Pic Systolique

$$\Delta t_{\text{peak}} = t_{\text{peak}}^{\text{sim}} - t_{\text{peak}}^{\text{ref}}$$

**Seuil acceptable** : $|\Delta t_{\text{peak}}| < 50$ ms

**Causes d'erreur** :
- Mauvaise estimation de $R_c$ ou $L$
- Déphasage dans l'interpolation du débit

#### 5.2.3 Constante de Temps Diastolique $\tau$

**Méthode d'estimation** : Ajustement exponentiel sur la phase diastolique :

$$P(t) = P_0 \cdot e^{-(t-t_0)/\tau} + P_{\infty}$$

**Implémentation** :
```python
def diastolic_tau(t, p):
    peak_idx = np.argmax(p)
    # Commencer après l'incisure dicrote (~30% après le pic)
    start_idx = peak_idx + int(0.3 * (len(p) - peak_idx))
    
    t_d = t[start_idx:] - t[start_idx]
    p_d = p[start_idx:] - np.min(p[start_idx:])
    
    # Ajustement par moindres carrés
    popt, _ = curve_fit(
        lambda t, A, tau: A * np.exp(-t / tau),
        t_d, p_d,
        p0=[np.max(p_d), 1.0],
        bounds=([0, 0.01], [np.inf, 10.0])
    )
    return popt[1]  # tau
```

**Validation** :
$$\text{Erreur relative} = \frac{|\tau_{\text{sim}} - \tau_{\text{ref}}|}{\tau_{\text{ref}}} \times 100\%$$

**Seuil** : < 20%

**Interprétation physiologique** :
- $\tau$ trop faible → $R_p$ ou $C$ sous-estimés
- $\tau$ trop élevé → $R_p$ ou $C$ surestimés

### 5.3 Mise à l'Échelle Affine (Affine Matching)

**Problème** : Les pressions simulées et mesurées peuvent différer en amplitude absolue (calibration du capteur, unités).

**Solution** : Comparer les **formes d'onde** indépendamment de l'échelle :

```python
def affine_match(reference, target):
    """
    Rescale reference pour matcher l'amplitude de target.
    
    P_ref_scaled = a * P_ref + b
    avec a = (max(target) - min(target)) / (max(ref) - min(ref))
         b = min(target) - a * min(ref)
    """
    ref = reference - np.min(reference)
    ref = ref / np.max(ref)  # Normalisation [0, 1]
    return ref * (np.max(target) - np.min(target)) + np.min(target)
```

**Utilité** :
- Valider la morphologie avant la calibration absolue
- Identifier les erreurs de forme vs. erreurs de gain

### 5.4 Protocole de Validation Complet

```python
# 1. Chargement des données
flow_data = np.loadtxt("data_typec_q.csv", delimiter=",")
pressure_data = np.loadtxt("data_typec_p.csv", delimiter=",")

# 2. Conversion d'unités (SI)
q_flow = flow_data[:, 1] * 1e-6  # mL/s → m³/s
p_ref_pa = pressure_data[:, 1] * 133.322  # mmHg → Pa

# 3. Création du modèle
wk = Windkessel(
    t_flow=t_flow,
    q_flow=q_flow,
    Rc=1.2e7,   # Pa·s/m³
    Rp=1.5e8,   # Pa·s/m³
    C=1.8e-9,   # m³/Pa
    L=5e4,      # Pa·s²/m³
)

# 4. Simulation (5 cycles pour atteindre le régime permanent)
sol = wk.solve(t_end=5*T, n_steps=5000)

# 5. Suppression du transitoire (garder les 20% derniers points)
startN = int(0.8 * len(sol.t))
t_sim = sol.t[startN:] - sol.t[startN]
P_sim = sol.p1[startN:]

# 6. Interpolation de la référence sur la grille de simulation
P_ref_interp = np.interp(t_sim, t_p, p_ref_pa)

# 7. Matching affine pour comparaison de forme
P_ref_matched = affine_match(P_ref_interp, P_sim)

# 8. Calcul des métriques
nrms = normalized_rms_error(P_ref_matched, P_sim)
dt_peak = t_sim[np.argmax(P_ref_matched)] - t_sim[np.argmax(P_sim)]
tau_sim = diastolic_tau(t_sim, P_sim)
tau_ref = diastolic_tau(t_sim, P_ref_matched)

# 9. Validation des seuils
assert nrms < 0.15, f"NRMS trop élevé: {nrms:.2%}"
assert abs(dt_peak) < 0.050, f"Décalage pic: {dt_peak*1000:.1f} ms"
assert abs(tau_sim - tau_ref) / tau_ref < 0.20, f"Erreur tau: {...:.2%}"
```

### 5.5 Visualisation des Résultats

```python
plt.figure(figsize=(10, 5))
plt.plot(t_sim, P_sim/133.322, label="Simulation", linewidth=2)
plt.plot(t_sim, P_ref_matched/133.322, '--', label="Référence (scalée)", color='gray')
plt.xlabel("Temps [s]")
plt.ylabel("Pression [mmHg]")
plt.title(f"Validation Windkessel - NRMS = {nrms:.2%}")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, T)  # Un cycle cardiaque
plt.show()
```

**Éléments à inspecter visuellement** :
1. **Pente systolique** : Montée rapide de pression
2. **Pic systolique** : Amplitude et position temporelle
3. **Incisure dicrote** : Présence et profondeur (fermeture aortique)
4. **Pente diastolique** : Décroissance exponentielle
5. **Pression diastolique minimale** : Niveau de base

### 5.6 Calibration des Paramètres

Si la validation échoue, ajuster les paramètres :

| Symptôme | Paramètre à ajuster | Effet |
|----------|---------------------|-------|
| Pression systolique trop basse | ↑ $R_p$ ou ↑ $R_c$ | Augmente l'amplitude |
| Pression diastolique trop basse | ↑ $R_p$ ou ↑ $C$ | Ralentit la décroissance |
| Pente diastolique trop raide | ↑ $R_p$ ou ↑ $C$ | Augmente $\tau$ |
| Pic trop tardif | ↓ $R_c$ ou ↓ $L$ | Réduit le déphasage |
| Incisure absente | ↑ $L$ ou ↓ $R_c$ | Accentue les effets dynamiques |
| Oscillations non physiques | ↓ $L$ ou ↑ $R_c$ | Amortit le système |

**Méthode systématique** :
1. Ajuster $R_p$ pour matcher la pression moyenne : $P_{\text{moy}} \approx R_p \cdot Q_{\text{moy}}$
2. Ajuster $C$ pour matcher $\tau$ diastolique : $C = \tau / R_p$
3. Ajuster $R_c$ pour matcher l'amplitude systolique
4. Ajuster $L$ (si nécessaire) pour affiner la forme du pic

---

## 6. Guide d'Utilisation {#utilisation}

### 6.1 Installation et Prérequis

```bash
pip install numpy scipy matplotlib
```

### 6.2 Exemple Complet

```python
import numpy as np
from windkessel import Windkessel

# 1. Générer un débit cardiaque synthétique (pour démonstration)
T = 0.8  # Période cardiaque [s] (75 bpm)
t = np.linspace(0, T, 500)

# Débit triangulaire simplifié (systole 0.3s, diastole 0.5s)
q = np.zeros_like(t)
idx_systole = t < 0.3
q[idx_systole] = 5e-4 * (1 - np.abs(t[idx_systole] - 0.15) / 0.15)  # Pic à 500 mL/s

# 2. Créer le modèle Windkessel 4 éléments
wk = Windkessel(
    t_flow=t,
    q_flow=q,
    Rc=1.0e7,    # 1.0e7 Pa·s/m³ ≈ 75 mmHg·s/L
    Rp=1.5e8,    # 1.5e8 Pa·s/m³ ≈ 1125 mmHg·s/L
    C=1.5e-9,    # 1.5e-9 m³/Pa ≈ 1.1 mL/mmHg
    L=5e4,       # 5e4 Pa·s²/m³
    periodic=True
)

print(wk)
# Windkessel(model='4-element', Rc=1.000e+07 Pa·s/m³, ...)
print(f"Constante de temps τ = {wk.time_constant:.3f} s")

# 3. Simuler sur 3 cycles
result = wk.solve(t_end=3*T, n_steps=3000)

print(f"Pression systolique: {np.max(result.p1)/133.322:.1f} mmHg")
print(f"Pression diastolique: {np.min(result.p1)/133.322:.1f} mmHg")

# 4. Visualiser le dernier cycle (régime permanent)
idx_last_cycle = result.t > 2*T
t_plot = result.t[idx_last_cycle] - 2*T
p_plot = result.p1[idx_last_cycle] / 133.322  # Conversion en mmHg

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(t_plot, p_plot, 'b-', linewidth=2)
plt.xlabel("Temps [s]")
plt.ylabel("Pression [mmHg]")
plt.title("Pression aortique simulée (Windkessel 4 éléments)")
plt.grid(alpha=0.3)
plt.show()
```

### 6.3 Cas d'Usage Avancés

#### 6.3.1 Simulation d'Hypertension

```python
# Hypertension artérielle : Rp augmentée, C diminuée
wk_hta = Windkessel(
    t_flow=t, q_flow=q,
    Rc=1.2e7,    # Légèrement augmentée
    Rp=2.5e8,    # ×1.67 (hypertension)
    C=0.8e-9,    # ÷1.87 (rigidité artérielle)
    L=5e4,
)

# Résultat attendu : 
# - Pression systolique augmentée
# - Pression diastolique augmentée
# - τ diminué (décroissance plus rapide)
```

#### 6.3.2 Vieillissement Artériel

```python
# Vieillissement : C diminuée (rigidité), Rp stable
wk_aged = Windkessel(
    t_flow=t, q_flow=q,
    Rc=1.5e7,    # Augmentée (perte de compliance aortique)
    Rp=1.5e8,    # Stable
    C=0.6e-9,    # Fortement diminuée
    L=5e4,
)

# Résultat attendu :
# - Pression pulsée augmentée (PP = Ps - Pd)
# - Pic systolique plus aigu
# - Incisure dicrote plus marquée
```

#### 6.3.3 Insuffisance Aortique

```python
# Fuite aortique : diminution de Rc (moins de résistance à l'éjection)
wk_ia = Windkessel(
    t_flow=t, q_flow=q,
    Rc=0.5e7,    # Diminuée de moitié
    Rp=1.2e8,    # Légèrement diminuée (baisse de post-charge)
    C=2.0e-9,    # Augmentée (dilatation aortique)
    L=5e4,
)
```

### 6.4 Bonnes Pratiques

✅ **À faire** :
- Utiliser `estimate_ic=True` pour converger plus vite vers le régime permanent
- Simuler au moins 3-5 cycles avant d'analyser les résultats
- Vérifier que `np.all(np.diff(t_flow) > 0)` (temps croissants)
- Convertir systématiquement en unités SI avant de créer le modèle
- Sauvegarder les métriques de validation dans un fichier JSON

❌ **À éviter** :
- Utiliser des données de débit bruitées (lisser avant interpolation)
- Initialiser avec $P_2(0) = 0$ si le débit moyen est élevé
- Oublier de vérifier `result.success` après `solve()`
- Comparer des pressions en mmHg avec des paramètres en unités SI

---

## 7. Annexes {#annexes}

### 7.1 Conversion d'Unités

| Grandeur | Unités Cliniques | Unités SI | Facteur |
|----------|------------------|-----------|---------|
| Pression | mmHg | Pa | × 133.322 |
| Débit | mL/s | m³/s | × 1e-6 |
| Débit | L/min | m³/s | × 1.667e-5 |
| Résistance | mmHg·s/mL | Pa·s/m³ | × 1.333e8 |
| Résistance | mmHg·s/L | Pa·s/m³ | × 1.333e5 |
| Compliance | mL/mmHg | m³/Pa | × 7.5e-9 |
| Inertance | mmHg·s²/mL | Pa·s²/m³ | × 1.333e8 |

### 7.2 Valeurs de Référence (Adulte Sain au Repos)

| Paramètre | Valeur Typique | Intervalle Normal |
|-----------|----------------|-------------------|
| Fréquence cardiaque | 75 bpm | 60-100 bpm |
| Pression systolique | 120 mmHg | 90-140 mmHg |
| Pression diastolique | 80 mmHg | 60-90 mmHg |
| Débit cardiaque | 5.0 L/min | 4.0-8.0 L/min |
| $R_p$ | 1.5e8 Pa·s/m³ | 1.0-2.0e8 |
| $C$ | 1.5e-9 m³/Pa | 1.0-2.5e-9 |
| $R_c$ | 1.0e7 Pa·s/m³ | 0.5-2.0e7 |
| $L$ | 5e4 Pa·s²/m³ | 1e4-1e5 |
| $\tau$ | 1.8 s | 1.2-2.5 s |

### 7.3 Ressources Complémentaires

**Livres** :
- "Cardiovascular Fluid Mechanics" par Roger Kamm
- "Hemodynamics" par Wilmer W. Nichols

**Articles fondateurs** :
- Frank, O. (1899). "Die Grundform des arteriellen Pulses"
- Westerhof, N. et al. (1969). "An artificial arterial system for pumping hearts"

**Logiciels** :
- PySeuille (https://github.com/TS-CUBED/PySeuille)
- OpenBF (Open-source Blood Flow)

### 7.4 Glossaire

- **Afterload** : Résistance à l'éjection ventriculaire (Rp + Rc)
- **Compliance** : Capacité d'un vaisseau à se distendre sous pression
- **Impédance** : Rapport pression/débit en régime sinusoïdal
- **Incisure dicrote** : Petit rebond de pression marquant la fermeture aortique
- **Post-charge** : Contrainte pariétale en fin de systole
- **Précharge** : Volume télédiastolique ventriculaire
- **Windkessel** : Allemand pour "chambre à air" (analogie avec les pompes à incendie du 19ème siècle)

---

## Conclusion

Le modèle Windkessel reste un outil fondamental en biomécanique cardiovasculaire malgré sa simplicité. Sa force réside dans :

1. **Son interprétabilité** : Chaque paramètre a une signification physiologique claire
2. **Sa rapidité** : Simulation en temps réel possible
3. **Sa robustesse** : Peu de paramètres à identifier

Cependant, il ne remplace pas les modèles 1D ou 3D pour :
- L'étude de la propagation d'onde de pouls
- Les écoulements complexes (bifurcations, sténoses)
- L'interaction fluide-structure détaillée

**Perspectives** :
- Couplage avec des modèles de fonction ventriculaire
- Identification automatique des paramètres par optimisation
- Extension à des réseaux multi-compartiments (circulation systémique + pulmonaire)

---

*Document rédigé pour la formation à la modélisation cardiovasculaire - 2026*
