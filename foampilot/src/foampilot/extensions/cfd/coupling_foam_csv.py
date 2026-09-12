import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from functools import lru_cache
from typing import Dict, List, Optional, Union

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CSVFoamIntegrator:
    """
    Intégrateur pour mapper des données CSV vers des patches OpenFOAM.
    
    Structure attendue du cas OpenFOAM :
        case_path/
        ├── constant/
        │   ├── polyMesh/
        │   │   ├── points
        │   │   ├── faces
        │   │   └── boundary
        │   └── boundaryData/  (créé automatiquement)
        └── 0/  (ou autre dossier de temps)
    
    Usage :
        integrator = CSVFoamIntegrator("./myCase")
        integrator.load_mesh()
        df_patch = integrator.get_patch_dataframe("wall1")
        df_mapped = integrator.map_csv_to_patch(df_patch, df_csv)
        bc_str = integrator.write_nonuniform_bc("wall1", df_mapped, {'Ta': 'scalar'})
    """
    
    def __init__(self, case_path: str):
        self.case_path = Path(case_path)
        self.poly_mesh = self.case_path / "constant" / "polyMesh"
        self._points: Optional[np.ndarray] = None
        self._faces: Optional[List[List[int]]] = None
        self._boundary: Optional[Dict] = None
        self._mesh_loaded = False

    # ---------- Mesh Reading (Robust Method) ----------

    @staticmethod
    def _clean_openfoam_lines(lines: List[str]) -> List[str]:
        """Supprime commentaires et lignes vides des fichiers OpenFOAM."""
        cleaned = []
        for line in lines:
            stripped = line.split('//')[0].strip()  # Remove inline comments
            if stripped and stripped not in ('(', ')', '{', '}'):
                cleaned.append(stripped)
        return cleaned

    def _read_points(self) -> np.ndarray:
        path = self.poly_mesh / "points"
        if not path.exists():
            raise FileNotFoundError(f"Points file not found: {path}")
        
        with open(path, "r") as f:
            lines = f.readlines()
        
        cleaned = self._clean_openfoam_lines(lines)
        try:
            start_idx = cleaned.index('(') + 1
            end_idx = cleaned.index(')')
        except ValueError as e:
            raise ValueError(f"Cannot parse points file format: {e}")
        
        points = []
        for i, line in enumerate(cleaned[start_idx:end_idx], start=start_idx):
            tokens = line.strip('()').split()
            if len(tokens) != 3:
                logger.warning(f"Skipping malformed line {i} in points: {line}")
                continue
            try:
                points.append([float(t) for t in tokens])
            except ValueError:
                logger.warning(f"Cannot convert to float line {i}: {line}")
                continue
        
        if not points:
            raise ValueError("No valid points found in mesh file")
        
        logger.info(f"Read {len(points)} points from {path}")
        return np.asarray(points)

    def _read_faces(self) -> List[List[int]]:
        path = self.poly_mesh / "faces"
        if not path.exists():
            raise FileNotFoundError(f"Faces file not found: {path}")
        
        with open(path, "r") as f:
            lines = f.readlines()
        
        cleaned = self._clean_openfoam_lines(lines)
        try:
            start_idx = cleaned.index('(') + 1
            end_idx = cleaned.index(')')
        except ValueError as e:
            raise ValueError(f"Cannot parse faces file format: {e}")
        
        faces = []
        for line in cleaned[start_idx:end_idx]:
            tokens = line.strip('()').split()
            if not tokens:
                continue
            try:
                n = int(tokens[0])
                face = [int(t) for t in tokens[1:1+n]]
                if len(face) == n:
                    faces.append(face)
                else:
                    logger.warning(f"Face vertex count mismatch: expected {n}, got {len(face)}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed face: {line} - {e}")
                continue
        
        logger.info(f"Read {len(faces)} faces from {path}")
        return faces

    def _read_boundary(self) -> Dict[str, Dict]:
        path = self.poly_mesh / "boundary"
        if not path.exists():
            raise FileNotFoundError(f"Boundary file not found: {path}")
        
        with open(path, "r") as f:
            content = f.read()
        
        # Remove comments
        content = '\n'.join(line.split('//')[0] for line in content.split('\n'))
        
        boundary = {}
        # Simple parser for OpenFOAM boundary dictionary
        try:
            start = content.index('(') + 1
            end = content.rindex(')')
            body = content[start:end]
            
            # Split by patch blocks (simplified)
            patches = body.split(';')
            current_patch = None
            current_entry = {}
            
            for line in patches:
                line = line.strip()
                if not line or line in '{}':
                    continue
                if '{' in line and '}' not in line:
                    # Start of patch definition
                    current_patch = line.split('{')[0].strip()
                    current_entry = {}
                elif '}' in line:
                    # End of patch definition
                    if current_patch:
                        boundary[current_patch] = current_entry
                        current_patch = None
                elif current_patch:
                    # Parse key-value
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        value = parts[1]
                        # Try to convert to int if possible
                        try:
                            current_entry[key] = int(value)
                        except ValueError:
                            current_entry[key] = value
        except Exception as e:
            logger.error(f"Failed to parse boundary file: {e}")
            raise
        
        logger.info(f"Read {len(boundary)} patches from {path}")
        return boundary

    def load_mesh(self, force: bool = False) -> None:
        """Charge les données du mesh en mémoire."""
        if self._mesh_loaded and not force:
            logger.debug("Mesh already loaded, use force=True to reload")
            return
        
        logger.info(f"Loading mesh from {self.case_path}")
        self._points = self._read_points()
        self._faces = self._read_faces()
        self._boundary = self._read_boundary()
        self._mesh_loaded = True
        logger.info(f"✓ Mesh loaded: {len(self._points)} points, {len(self._faces)} faces, {len(self._boundary)} patches")

    @lru_cache(maxsize=32)
    def _get_patch_face_ids(self, patch_name: str) -> np.ndarray:
        """Cache des IDs de faces pour un patch donné."""
        if self._boundary is None:
            self.load_mesh()
        if patch_name not in self._boundary:
            available = list(self._boundary.keys())
            raise ValueError(f"Patch '{patch_name}' not found. Available: {available}")
        
        info = self._boundary[patch_name]
        start = info.get("startFace")
        nfaces = info.get("nFaces")
        
        if start is None or nfaces is None:
            raise ValueError(f"Patch '{patch_name}' missing startFace or nFaces")
        
        return np.arange(start, start + nfaces)

    def get_patch_dataframe(self, patch_name: str) -> pd.DataFrame:
        """Retourne un DataFrame avec les centres de faces d'un patch."""
        if not self._mesh_loaded:
            self.load_mesh()
        
        face_ids = self._get_patch_face_ids(patch_name)
        rows = []
        
        for local_id, face_id in enumerate(face_ids):
            face = self._faces[face_id]
            if not face:
                continue
            center = np.mean(self._points[face], axis=0)
            rows.append({
                "localId": local_id,
                "faceId": int(face_id),
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2])
            })
        
        df = pd.DataFrame(rows)
        logger.debug(f"Patch '{patch_name}': {len(df)} faces")
        return df

    # ---------- Data Mapping ----------

    def map_csv_to_patch(self, df_patch: pd.DataFrame, df_csv: pd.DataFrame, 
                        coord_cols: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Map des données CSV vers les faces d'un patch via nearest-neighbor.
        
        Args:
            df_patch: DataFrame avec colonnes x, y, z (centres de faces)
            df_csv: DataFrame source avec coordonnées et données
            coord_cols: Dict pour renommer les colonnes de coords dans df_csv 
                       ex: {'x': 'lon', 'y': 'lat', 'z': 'alt'}
        
        Returns:
            DataFrame avec les données CSV alignées sur df_patch
        """
        # Validation des colonnes requises
        required_patch_cols = {'x', 'y', 'z'}
        if not required_patch_cols.issubset(df_patch.columns):
            raise ValueError(f"df_patch must contain: {required_patch_cols}")
        
        # Gestion des noms de colonnes de coordonnées dans le CSV
        coord_map = coord_cols or {'x': 'x', 'y': 'y', 'z': 'z'}
        csv_coords = [coord_map[c] for c in ['x', 'y', 'z']]
        
        if not set(csv_coords).issubset(df_csv.columns):
            available = list(df_csv.columns)
            raise ValueError(f"CSV must contain columns: {csv_coords}. Available: {available}")
        
        # Build KDTree sur les données CSV
        csv_points = df_csv[csv_coords].values.astype(float)
        patch_points = df_patch[['x', 'y', 'z']].values.astype(float)
        
        tree = cKDTree(csv_points)
        _, indices = tree.query(patch_points, k=1)
        
        # Sélection et reset index
        df_mapped = df_csv.iloc[indices].reset_index(drop=True)
        
        # Ajouter les colonnes de référence du patch
        df_mapped['patch_localId'] = df_patch['localId'].values
        df_mapped['patch_faceId'] = df_patch.get('faceId', pd.Series([None]*len(df_patch))).values
        
        logger.info(f"Mapped {len(df_mapped)} patch faces to CSV data")
        return df_mapped

    # ---------- BC Generation Methods ----------

    def write_nonuniform_bc(self, patch_name: str, df_mapped: pd.DataFrame, 
                           fields_config: Dict[str, str],
                           bc_type: str = "externalWallHeatFluxTemperature",
                           mode: str = "coefficient",
                           default_value: float = 300.0) -> str:
        """
        Génère un bloc de boundary condition 'nonuniform List'.
        
        Args:
            patch_name: Nom du patch OpenFOAM
            df_mapped: DataFrame avec les valeurs à écrire
            fields_config: Dict {field_name: type} ex: {'Ta': 'scalar', 'U': 'vector'}
            bc_type: Type de BC OpenFOAM
            mode: Mode de la BC (pour externalWallHeatFluxTemperature)
            default_value: Valeur par défaut pour le champ 'value'
        
        Returns:
            String au format OpenFOAM
        """
        bc_str = f"    {patch_name}\n    {{\n"
        bc_str += f"        type            {bc_type};\n"
        if mode:
            bc_str += f"        mode            {mode};\n"
        
        for field, ftype in fields_config.items():
            if field not in df_mapped.columns:
                logger.warning(f"Field '{field}' not found in df_mapped, skipping")
                continue
                
            values = df_mapped[field].dropna().values
            n_values = len(values)
            
            bc_str += f"        {field}            nonuniform List<{ftype}> {n_values}\n        (\n"
            for v in values:
                if pd.isna(v):
                    bc_str += "            nan\n"
                elif ftype == 'scalar':
                    bc_str += f"            {float(v):.6g}\n"
                elif ftype == 'vector':
                    # Assume vector is stored as 3 separate columns or tuple
                    bc_str += f"            {v}\n"  # Adapt as needed
                else:
                    bc_str += f"            {v}\n"
            bc_str += "        );\n"
            
        bc_str += "        kappaMethod     fluidThermo;\n"
        bc_str += f"        value           uniform {default_value};\n"
        bc_str += "    }\n"
        
        logger.debug(f"Generated BC block for patch '{patch_name}' with {len(fields_config)} fields")
        return bc_str

    def export_to_boundary_data(self, patch_name: str, df_mapped: pd.DataFrame, 
                               field_name: str, time: Union[str, float] = "0",
                               output_dir: Optional[str] = None) -> Path:
        """
        Exporte vers le format constant/boundaryData (spatio-temporel).
        
        Args:
            patch_name: Nom du patch
            df_mapped: DataFrame avec colonnes [x, y, z, field_name, (optionnel: time)]
            field_name: Nom du champ à exporter
            time: Identifiant du pas de temps
            output_dir: Dossier de sortie (défaut: case_path/constant/boundaryData)
        
        Returns:
            Path du dossier créé
        """
        # Travailler sur une copie pour éviter les effets de bord
        df_work = df_mapped.copy()
        
        # Récupérer les coordonnées du patch (sans écraser celles du CSV !)
        df_patch_coords = self.get_patch_dataframe(patch_name)[['localId', 'x', 'y', 'z']]
        
        # Merge sur localId si présent, sinon on suppose que l'ordre correspond
        if 'patch_localId' in df_work.columns:
            df_work = df_work.merge(df_patch_coords, left_on='patch_localId', right_on='localId', 
                                   how='left', suffixes=('_csv', '_patch'))
            # Priorité aux coords du patch
            for c in ['x', 'y', 'z']:
                df_work[c] = df_work[f'{c}_patch'].combine_first(df_work[f'{c}_csv'])
        else:
            # Fallback: on suppose que l'ordre des lignes correspond
            if len(df_work) == len(df_patch_coords):
                for c in ['x', 'y', 'z']:
                    df_work[c] = df_patch_coords[c].values
            else:
                logger.warning("Row count mismatch: using CSV coordinates as-is")
        
        # Ajouter le temps si absent
        if 'time' not in df_work.columns:
            df_work['time'] = time
        
        # Déterminer le dossier de sortie
        out_dir = Path(output_dir) if output_dir else self.case_path / "constant" / "boundaryData"
        
        # Appel à la fonction utilitaire
        df_to_boundary_data_timeseries(
            df=df_work,
            patch_name=patch_name,
            field_name=field_name,
            time_col='time',
            output_dir=str(out_dir)
        )
        
        result_path = out_dir / patch_name / f"{time:g}" if isinstance(time, (int, float)) else out_dir / patch_name / str(time)
        logger.info(f"Exported boundaryData to {result_path}")
        return result_path


# ---------- Utility Functions ----------

def write_openfoam_header(f, class_name: str, object_name: str, location: str) -> None:
    """Écrit l'en-tête standard OpenFOAM dans un fichier."""
    header = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    location    "{location}";
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
    f.write(header)


def df_to_boundary_data_timeseries(df: pd.DataFrame, patch_name: str, field_name: str, 
                                  time_col: str = 'time', output_dir: str = "constant/boundaryData") -> None:
    """
    Convertit un DataFrame multi-pas de temps vers le format boundaryData OpenFOAM.
    
    Format attendu du DataFrame :
    - Champ scalaire : [time, x, y, z, {field_name}]
    - Champ vectoriel : [time, x, y, z, {field_name}_x, {field_name}_y, {field_name}_z]
      OU [time, x, y, z, vx, vy, vz] avec is_vector=True
    """
    patch_dir = Path(output_dir) / patch_name
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Détection automatique champ vectoriel
    vector_suffixes = [f'{field_name}_x', f'{field_name}_y', f'{field_name}_z']
    legacy_vector_cols = ['vx', 'vy', 'vz']
    
    is_vector = all(c in df.columns for c in vector_suffixes) or \
                all(c in df.columns for c in legacy_vector_cols)
    
    class_name = "vectorField" if is_vector else "scalarField"
    logger.info(f"Exporting {'vector' if is_vector else 'scalar'} field '{field_name}' for patch '{patch_name}'")

    times = sorted(df[time_col].dropna().unique())
    logger.info(f"Found {len(times)} time steps: {times[:3]}{'...' if len(times) > 3 else ''}")
    
    for t in times:
        t_str = f"{float(t):g}" if isinstance(t, (int, float)) else str(t)
        time_dir = patch_dir / t_str
        time_dir.mkdir(parents=True, exist_ok=True)
        
        df_t = df[df[time_col] == t].copy()
        n_points = len(df_t)
        
        if n_points == 0:
            logger.warning(f"No data for time {t_str}, skipping")
            continue
        
        # 1. Écriture du fichier points
        points_path = time_dir / "points"
        with open(points_path, "w") as f:
            write_openfoam_header(f, "vectorField", "points", 
                                location=f"{output_dir}/{patch_name}/{t_str}")
            f.write(f"{n_points}\n(\n")
            for _, row in df_t.iterrows():
                f.write(f"({row['x']:.6g} {row['y']:.6g} {row['z']:.6g})\n")
            f.write(")\n\n// ************************************************************************* //\n")

        # 2. Écriture du fichier de champ
        field_path = time_dir / field_name
        with open(field_path, "w") as f:
            write_openfoam_header(f, class_name, field_name,
                                location=f"{output_dir}/{patch_name}/{t_str}")
            f.write(f"{n_points}\n(\n")
            for _, row in df_t.iterrows():
                if is_vector:
                    if all(c in row for c in vector_suffixes):
                        val = (row[f'{field_name}_x'], row[f'{field_name}_y'], row[f'{field_name}_z'])
                    elif all(c in row for c in legacy_vector_cols):
                        val = (row['vx'], row['vy'], row['vz'])
                    else:
                        val = (0, 0, 0)  # Fallback
                    f.write(f"({val[0]:.6g} {val[1]:.6g} {val[2]:.6g})\n")
                else:
                    val = row.get(field_name, 0)
                    f.write(f"{float(val) if pd.notna(val) else 0:.6g}\n")
            f.write(")\n\n// ************************************************************************* //\n")
        
        logger.debug(f"✓ Wrote {n_points} points to {time_dir}")
    
    logger.info(f"✓ Completed export to {patch_dir}")


# --- Example Usage ---
if __name__ == "__main__":
    # Exemple complet d'utilisation
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Créer une structure de cas minimale pour le test
        case = Path(tmpdir) / "testCase"
        (case / "constant" / "polyMesh").mkdir(parents=True)
        
        # Écrire un fichier points minimal
        (case / "constant" / "polyMesh" / "points").write_text("""FoamFile
{
    version 2.0;
    format ascii;
    class vectorField;
    object points;
}
4
(
(0 0 0)
(1 0 0)
(0 1 0)
(1 1 0)
)
""")
        
        # Écrire un fichier faces minimal
        (case / "constant" / "polyMesh" / "faces").write_text("""FoamFile
{
    version 2.0;
    format ascii;
    class faceList;
    object faces;
}
2
(
3(0 1 2)
3(1 3 2)
)
""")
        
        # Écrire un fichier boundary minimal
        (case / "constant" / "polyMesh" / "boundary").write_text("""FoamFile
{
    version 2.0;
    format ascii;
    class polyBoundaryMesh;
    object boundary;
}
2
(
    wall1
    {
        type            wall;
        nFaces          2;
        startFace       0;
    }
    defaultFaces
    {
        type            empty;
        nFaces          0;
        startFace       2;
    }
)
""")
        
        # Tester l'intégrateur
        integrator = CSVFoamIntegrator(str(case))
        integrator.load_mesh()
        
        df_patch = integrator.get_patch_dataframe("wall1")
        print(f"\nPatch faces:\n{df_patch}")
        
        # Créer des données CSV fictives
        df_csv = pd.DataFrame({
            'x': [0.1, 0.9, 0.2, 0.8],
            'y': [0.1, 0.9, 0.8, 0.2], 
            'z': [0, 0, 0, 0],
            'Ta': [350, 400, 375, 390],
            'h': [10, 15, 12, 14]
        })
        
        df_mapped = integrator.map_csv_to_patch(df_patch, df_csv)
        print(f"\nMapped data:\n{df_mapped[['patch_localId', 'Ta', 'h']]}")
        
        # Générer une BC
        bc = integrator.write_nonuniform_bc("wall1", df_mapped, {'Ta': 'scalar', 'h': 'scalar'})
        print(f"\nGenerated BC:\n{bc}")
        
        # Export boundaryData
        result = integrator.export_to_boundary_data("wall1", df_mapped, field_name="Ta", time=0)
        print(f"\n✓ Exported to: {result}")