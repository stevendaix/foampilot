"""
foampilot.constant.constantDirectory
-----------------------------------

Manager for the 'constant' directory in an OpenFOAM case.

Ce module gère l'ensemble des fichiers habituellement présents dans
constant/: transportProperties, turbulenceProperties, physicalProperties,
g (gravity), pRef (si compressible) et optionnellement radiationProperties
+ fvModels.

La méthode write() tente d'appeler la méthode write() de chaque fichier de
façon robuste (soit en lui passant un "file path", soit en lui passant le
répertoire parent) et utilise un fallback sur write_file() si disponible.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Dict, Union
import logging

from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.constant.physicalProperties import PhysicalPropertiesFile
from foampilot.constant.gravityFile import GravityFile
from foampilot.constant.pRefFile import PRefFile
from foampilot.constant.radiationPropertiesFile import RadiationPropertiesFile, FvModelsFile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ConstantDirectory:
    """
    Manager for the 'constant' directory in an OpenFOAM case.

    Usage:
        cd = ConstantDirectory(parent_foam_instance)
        cd.enable_radiation(model="fvDOM", absorptivity=0.7, emissivity=0.7)
        cd.write()
    """

    def __init__(self, parent: Any, *, with_radiation: bool = False):
        self.parent = parent

        # composants par défaut (instances prêtes à l'emploi)
        self._transportProperties = TransportPropertiesFile()
        self._turbulenceProperties = TurbulencePropertiesFile()
        self._physicalProperties = PhysicalPropertiesFile()
        self._gravity = GravityFile()
        self._pRef = PRefFile()

        # radiation (optionnel)
        self.with_radiation: bool = False
        self._radiation: Optional[RadiationPropertiesFile] = None
        self._fvmodels: Optional[FvModelsFile] = None

        if with_radiation:
            # activation avec paramètres par défaut
            self.enable_radiation()

    # -------------------
    # Properties (getters / setters)
    # -------------------
    @property
    def transportProperties(self) -> TransportPropertiesFile:
        return self._transportProperties

    @transportProperties.setter
    def transportProperties(self, value: TransportPropertiesFile) -> None:
        self._transportProperties = value

    @property
    def turbulenceProperties(self) -> TurbulencePropertiesFile:
        return self._turbulenceProperties

    @turbulenceProperties.setter
    def turbulenceProperties(self, value: TurbulencePropertiesFile) -> None:
        self._turbulenceProperties = value

    @property
    def physicalProperties(self) -> PhysicalPropertiesFile:
        return self._physicalProperties

    @physicalProperties.setter
    def physicalProperties(self, value: PhysicalPropertiesFile) -> None:
        self._physicalProperties = value

    @property
    def gravity(self) -> GravityFile:
        return self._gravity

    @gravity.setter
    def gravity(self, value: GravityFile) -> None:
        self._gravity = value

    @property
    def pRef(self) -> PRefFile:
        return self._pRef

    @pRef.setter
    def pRef(self, value: PRefFile) -> None:
        self._pRef = value

    @property
    def radiation(self) -> Optional[RadiationPropertiesFile]:
        return self._radiation

    @radiation.setter
    def radiation(self, value: Optional[RadiationPropertiesFile]) -> None:
        self._radiation = value
        self.with_radiation = value is not None

    # -------------------
    # Radiation helpers
    # -------------------
    def enable_radiation(self, model: str = "P1", absorptivity: float = 0.5,
                         emissivity: float = 0.5, E: float = 0.0, **kwargs) -> None:
        """
        Activer la radiation et créer les instances nécessaires.

        Args:
            model: "P1" ou "fvDOM"
            absorptivity, emissivity, E: constantes pour constantCoeffs
            kwargs: passe d'autres paramètres (nPhi, nTheta, tolerance, maxIter)
        """
        logger.debug("Enabling radiation: model=%s, absorptivity=%s, emissivity=%s", model, absorptivity, emissivity)
        self.with_radiation = True
        self._radiation = RadiationPropertiesFile(
            model=model,
            absorptivity=absorptivity,
            emissivity=emissivity,
            E=E,
            **kwargs
        )
        self._fvmodels = FvModelsFile()

    def disable_radiation(self) -> None:
        """Désactive la radiation (les fichiers ne seront pas écrits)."""
        logger.debug("Disabling radiation")
        self.with_radiation = False
        self._radiation = None
        self._fvmodels = None

    # -------------------
    # Internal utility: write wrapper robuste
    # -------------------
    def _safe_write(self, obj: Any, target_path: Union[str, Path]) -> None:
        """
        Appelle la méthode write de l'objet de la façon la plus robuste possible.

        Tentatives (dans l'ordre):
          1. obj.write(target_path)  # la plupart des classes (file path)
          2. obj.write(target_path.parent)  # si l'objet attend un dossier
          3. obj.write_file(target_path)  # fallback direct sur write_file si disponible

        Lève l'exception finale si aucune tentative ne fonctionne.
        """
        target = Path(target_path)
        logger.debug("Trying to write %s -> %s", getattr(obj, "object_name", type(obj)), target)
        # 1) Essayer directement
        try:
            obj.write(target)
            logger.info("Wrote %s -> %s", getattr(obj, "object_name", type(obj)), target)
            return
        except TypeError as exc_type:
            logger.debug("obj.write(target) raised TypeError: %s", exc_type)
        except Exception as exc:
            # Si l'objet a sa propre erreur d'écriture, on essaie quand même d'autres approches
            logger.debug("obj.write(target) raised: %s", exc)

        # 2) Essayer de passer le dossier parent (cas où write attend le dossier)
        try:
            obj.write(target.parent)
            logger.info("Wrote %s -> %s (using parent dir)", getattr(obj, "object_name", type(obj)), target.parent)
            return
        except Exception as exc:
            logger.debug("obj.write(target.parent) also failed: %s", exc)

        # 3) Fallback: si l'objet expose write_file(file_path)
        if hasattr(obj, "write_file"):
            try:
                obj.write_file(target)
                logger.info("Wrote %s -> %s (using write_file)", getattr(obj, "object_name", type(obj)), target)
                return
            except Exception as exc:
                logger.debug("obj.write_file(target) failed: %s", exc)

        # Rien n'a fonctionné : propager une erreur claire
        raise RuntimeError(f"Impossible d'écrire l'objet {obj} vers {target}; "
                           "méthodes write(...) / write_file(...) ont toutes échoué.")

    # -------------------
    # Write method principal
    # -------------------
    def write(self) -> None:
        """
        Écrit tous les fichiers nécessaires dans le dossier constant/ du cas.

        - crée constant/ et constant/polyMesh si nécessaire
        - écrit turbulenceProperties (toujours)
        - si compressible: écrit physicalProperties + pRef
          sinon: écrit transportProperties
        - écrit g si with_gravity est True
        - écrit radiationProperties et fvModels si with_radiation est True
        """
        base_path = Path(self.parent.case_path)
        constant_path = base_path / "constant"
        polyMesh_path = constant_path / "polyMesh"

        # create directories
        polyMesh_path.mkdir(parents=True, exist_ok=True)
        constant_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directories exist: %s, %s", constant_path, polyMesh_path)

        # turbulenceProperties (toujours)
        try:
            self._safe_write(self.turbulenceProperties, constant_path / "turbulenceProperties")
        except Exception as exc:
            logger.error("Erreur écriture turbulenceProperties: %s", exc)
            raise

        # compressible vs incompressible
        if getattr(self.parent, "compressible", False):
            try:
                self._safe_write(self.physicalProperties, constant_path / "physicalProperties")
                self._safe_write(self.pRef, constant_path / "pRef")
            except Exception as exc:
                logger.error("Erreur écriture fichiers compressible: %s", exc)
                raise
        else:
            try:
                self._safe_write(self.transportProperties, constant_path / "transportProperties")
            except Exception as exc:
                logger.error("Erreur écriture transportProperties: %s", exc)
                raise

        # Gravity (g) si activé sur le parent ou si le flag with_gravity n'existe pas (on écrit par défaut)
        with_gravity = getattr(self.parent, "with_gravity", True)
        if with_gravity:
            try:
                self._safe_write(self.gravity, constant_path / "g")
            except Exception as exc:
                logger.error("Erreur écriture gravity (g): %s", exc)
                raise

        # Radiation (optionnel)
        if self.with_radiation:
            # Si absence d'instances, créer des instances par défaut
            if self._radiation is None:
                self._radiation = RadiationPropertiesFile()
            if self._fvmodels is None:
                self._fvmodels = FvModelsFile()

            try:
                # _safe_write essaiera d'appeler obj.write(file_path) puis obj.write(dir)
                self._safe_write(self._radiation, constant_path / "radiationProperties")
                self._safe_write(self._fvmodels, constant_path / "fvModels")
            except Exception as exc:
                logger.error("Erreur écriture radiation: %s", exc)
                raise

        logger.info("Écriture du répertoire constant terminée pour %s", base_path)