# Diagnostic : PNG noirs avec PyVista/VTK sous WSLg

## Résumé

Les captures d'écran PyVista en mode off-screen étaient **entièrement noires** sous WSLg avec VTK 9.3.1. Le problème est maintenant résolu par une mise à jour de VTK et l'activation du backend EGL.

## Chronologie du diagnostic

### 1. Observations initiales

- Tous les PNG générés par `pv.Plotter(off_screen=True)` étaient noirs
- Le système est sous **WSL/WSLg** (Ubuntu 22.04, kernel 6.18.33.2-microsoft-standard-WSL2)
- Messages `D3D12: Removing Device` dans la sortie VTK
- `/dev/dxg` existe → accès GPU Windows via D3D12 confirmé

### 2. Tests sécurisés (sans off-screen)

| Test | Résultat |
|------|----------|
| Environnement WSLg | ✅ `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`, `/dev/dxg` présent |
| Bibliothèques Wayland/EGL | ✅ `libwayland-egl.so.1`, `libEGL.so.1` présentes |
| VTK RenderWindow | ✅ `vtkXOpenGLRenderWindow`, OpenGL supporté |
| Rendu VTK interactif | ✅ `rw.Render()` fonctionne |
| OSMesa | ❌ pas dans ldconfig |

### 3. Tests off-screen VTK natif (tests décisifs)

| Test | Résultat |
|------|----------|
| Test A: `OffScreenRenderingOn()` + `Render()` | ✅ OK |
| Test B: + `vtkWindowToImageFilter` + PNG | ❌ **PNG 100% noir** |
| Test 1: `ReadFrontBufferOff` | ❌ noir |
| Test 2: `ReadFrontBufferOn` | ❌ noir |
| Test 3: Buffer type par défaut | ❌ noir |
| Test 4: `SetInputBufferTypeToRGBA` | ❌ noir |

**Conclusion** : `Render()` off-screen fonctionne, mais la lecture du framebuffer retourne systématiquement des zéros. Le problème n'est pas PyVista, ni l'écriture PNG, ni la sélection du buffer.

### 4. Vérification EGL système

```python
libEGL.so           → chargeable
eglGetDisplay(NULL) → 0x6175679fab00
eglInitialize()     → échec (major=0, minor=0)
```

EGL existe au niveau système mais **ne s'initialise pas** avec le display par défaut dans cette configuration WSLg.

### 5. Vérification backend EGL dans VTK

```python
from vtkmodules.vtkRenderingOpenGL2 import vtkEGLRenderWindow
```

- **VTK 9.3.1** : ❌ `ImportError` — backend EGL absent
- **VTK 9.6.2** : ✅ disponible

`VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow` avec VTK 9.3.1 reste bloqué sur `vtkXOpenGLRenderWindow`.

### 6. Test VTK 9.6.2 + EGL (solution)

| Configuration | RenderWindow | Off-screen PNG |
|---------------|--------------|----------------|
| VTK 9.3.1 (défaut) | `vtkXOpenGLRenderWindow` | ❌ noir |
| VTK 9.6.2 + EGL | `vtkEGLRenderWindow` | ✅ correct |

Résultat VTK 9.6.2 + EGL :
```text
shape: (300, 400, 3)
min: 0, max: 255, mean: 219.18
ratio_noir: 0.0000
```

### 7. Mise à jour effectuée

| Package | Avant | Après |
|---------|-------|-------|
| VTK | 9.3.1 | **9.6.2** |
| PyVista | 0.46.3 | **0.48.4** |
| build123d | 0.10.0 | **0.11.1** |
| cadquery-ocp | 7.8.1.1.post1 | **7.9.3.1.1** |
| numpy | 1.26.4 | **2.2.6** |

### 8. Compatibilité FOAMPilot vérifiée

| Module | Tests | Résultat |
|--------|-------|----------|
| `test_boundary_viewer.py` | 12 | ✅ 12 passed |
| `test_direct_openfoam_export.py` | 3 | ✅ 3 passed |
| `test_postprocess_cube.py` | 15 | ✅ 15 passed |
| `test_topology_with_centerline.py` | 6 | ✅ 6 passed |
| **Total** | **36** | **✅ 36 passed** |

Les 7 échecs dans `test_boundary_class.py` sont préexistants et sans rapport avec VTK/PyVista.

## Cause racine

**VTK 9.3.1** ne supportait pas le backend EGL et utilisait `vtkXOpenGLRenderWindow` par défaut. Sous WSLg, le rendu off-screen X11/GLX ne permet pas la lecture du framebuffer par `vtkWindowToImageFilter`, produisant des PNG noirs silencieux.

## Solution

```bash
# Dans ~/.bashrc
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
```

Avec VTK ≥ 9.6.2, le backend EGL fonctionne correctement sous WSLg pour le rendu off-screen.

## Pipeline confirmé

```
PyVista/VTK
   ↓
vtkEGLRenderWindow (EGL)
   ↓
Mesa/WSLg
   ↓
/dev/dxg
   ↓
D3D12
   ↓
GPU Windows
```
