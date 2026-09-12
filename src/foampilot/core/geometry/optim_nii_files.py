"""
Pipeline CFD ultra-optimisé pour la mémoire
Auteur: Expert en optimisation mémoire
Version: 5.0
Date: 2024
"""

import numpy as np
import nibabel as nib
import trimesh
import gc
import time
import warnings
import logging
import psutil
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager
import zarr
from numcodecs import Blosc
from scipy.ndimage import (
    distance_transform_edt, 
    binary_fill_holes, 
    label, 
    zoom, 
    gaussian_filter,
    median_filter
)
from joblib import Parallel, delayed, Memory
from skimage.measure import marching_cubes
from trimesh import repair

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cfd_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppression des warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CLASSES DE GESTION DE MÉMOIRE
# ============================================================================

@dataclass
class MemoryConfig:
    """Configuration de la mémoire."""
    limit_gb: float = None
    cache_dir: str = None
    chunk_size: int = 64
    compression_level: int = 3
    max_blocks_in_memory: int = 10
    
    def __post_init__(self):
        if self.limit_gb is None:
            available_gb = psutil.virtual_memory().available / 1024**3
            self.limit_gb = available_gb * 0.7  # Utiliser 70% de la mémoire disponible
            
        if self.cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="cfd_cache_")
            
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

class MemoryAwareArray:
    """Tableau intelligent avec awareness mémoire."""
    
    def __init__(self, shape: Tuple[int, ...], dtype=np.float32, 
                 config: MemoryConfig = None, name: str = "array"):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.config = config or MemoryConfig()
        
        # Calculer la taille
        self.bytes_per_element = np.dtype(dtype).itemsize
        self.total_elements = np.prod(shape)
        self.size_gb = (self.total_elements * self.bytes_per_element) / 1024**3
        
        # Décider du backend
        if self.size_gb > self.config.limit_gb * 0.3:  # Si > 30% de la limite
            self.backend = 'zarr'
            self._init_zarr_backend()
        else:
            self.backend = 'numpy'
            self._init_numpy_backend()
        
        logger.info(f"Created {self.name}: {shape} ({self.size_gb:.2f} GB) -> backend: {self.backend}")
    
    def _init_numpy_backend(self):
        """Initialise le backend numpy."""
        self.data = np.zeros(self.shape, dtype=self.dtype)
        self.is_in_memory = True
    
    def _init_zarr_backend(self):
        """Initialise le backend Zarr."""
        # Créer le store Zarr
        store_path = Path(self.config.cache_dir) / f"{self.name}.zarr"
        store = zarr.DirectoryStore(str(store_path))
        
        # Déterminer la taille des chunks
        chunk_shape = tuple(min(self.config.chunk_size, s) for s in self.shape)
        
        # Créer le tableau Zarr
        compressor = Blosc(cname='zstd', clevel=self.config.compression_level, shuffle=Blosc.SHUFFLE)
        self.data = zarr.zeros(
            self.shape, 
            dtype=self.dtype, 
            store=store,
            chunks=chunk_shape,
            compressor=compressor
        )
        self.is_in_memory = False
        self.store_path = store_path
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __array__(self):
        """Conversion implicite en numpy array."""
        return self.to_numpy()
    
    def to_numpy(self) -> np.ndarray:
        """Convertit en numpy array si possible."""
        if self.backend == 'zarr':
            if self.size_gb < self.config.limit_gb * 0.5:
                logger.info(f"Loading {self.name} to memory ({self.size_gb:.2f} GB)")
                self.data = self.data[:]
                self.backend = 'numpy'
                self.is_in_memory = True
                # Nettoyer le fichier Zarr
                if hasattr(self, 'store_path') and self.store_path.exists():
                    shutil.rmtree(self.store_path)
            else:
                logger.warning(f"Cannot load {self.name} to memory, keeping Zarr backend")
        return self.data if self.is_in_memory else self.data[:]
    
    def free_memory(self):
        """Libère la mémoire si possible."""
        if self.backend == 'numpy' and self.size_gb > 0.1:  # Si > 100MB
            logger.info(f"Freeing memory for {self.name} ({self.size_gb:.2f} GB)")
            del self.data
            self.data = None
            self.is_in_memory = False
            force_garbage_collection()
    
    def persist_to_disk(self, path: str = None):
        """Persiste le tableau sur disque."""
        if path is None:
            path = Path(self.config.cache_dir) / f"{self.name}_persisted.npy"
        
        if self.is_in_memory:
            np.save(path, self.data)
        else:
            # Pour Zarr, copier le store
            import shutil
            if hasattr(self, 'store_path'):
                shutil.copytree(self.store_path, path)
        
        logger.info(f"Persisted {self.name} to {path}")
        return path

class MemoryManager:
    """Gestionnaire central de mémoire."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.arrays = {}
        self.peak_memory = 0
        self.start_memory = psutil.Process().memory_info().rss
        
        logger.info(f"MemoryManager initialized with limit: {self.config.limit_gb:.1f} GB")
        logger.info(f"Cache directory: {self.config.cache_dir}")
    
    def create_array(self, shape: Tuple[int, ...], dtype=np.float32, 
                    name: str = None) -> MemoryAwareArray:
        """Crée un tableau géré."""
        if name is None:
            name = f"array_{len(self.arrays)}"
        
        array = MemoryAwareArray(shape, dtype, self.config, name)
        self.arrays[name] = array
        
        # Mettre à jour la mémoire peak
        current_memory = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return array
    
    def get_current_memory(self) -> float:
        """Retourne l'utilisation mémoire actuelle en GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Retourne les statistiques d'utilisation mémoire."""
        process = psutil.Process()
        system = psutil.virtual_memory()
        
        return {
            'process_rss_gb': process.memory_info().rss / 1024**3,
            'process_vms_gb': process.memory_info().vms / 1024**3,
            'process_percent': process.memory_percent(),
            'system_available_gb': system.available / 1024**3,
            'system_total_gb': system.total / 1024**3,
            'system_percent': system.percent,
            'peak_gb': self.peak_memory,
            'limit_gb': self.config.limit_gb
        }
    
    def check_memory_safe(self, required_gb: float) -> bool:
        """Vérifie si on peut allouer en sécurité."""
        current = self.get_current_memory()
        available = self.config.limit_gb - current
        
        if required_gb > available:
            logger.warning(f"Memory unsafe: required {required_gb:.2f} GB, available {available:.2f} GB")
            return False
        
        return True
    
    def cleanup(self, force: bool = False):
        """Nettoie tous les tableaux."""
        logger.info("Cleaning up MemoryManager...")
        
        for name, array in list(self.arrays.items()):
            array.free_memory()
            del self.arrays[name]
        
        # Forcer le garbage collection
        force_garbage_collection()
        
        # Nettoyer le répertoire cache si demandé
        if force and Path(self.config.cache_dir).exists():
            shutil.rmtree(self.config.cache_dir)
            logger.info(f"Removed cache directory: {self.config.cache_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def __del__(self):
        self.cleanup()

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def force_garbage_collection():
    """Force le garbage collection de manière agressive."""
    for i in range(3):
        gc.collect(generation=i)
    
    # Essayer de libérer la mémoire au niveau OS (Linux/Unix)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

@contextmanager
def memory_context(name: str = "operation"):
    """Contexte pour le monitoring mémoire."""
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss / 1024**3
    
    logger.info(f"Starting {name}...")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / 1024**3
        delta_mem = end_mem - start_mem
        
        logger.info(f"Finished {name} in {end_time - start_time:.1f}s, "
                   f"memory Δ: {delta_mem:+.2f} GB")

def estimate_array_memory(shape: Tuple[int, ...], dtype=np.float32) -> float:
    """Estime la mémoire nécessaire pour un tableau."""
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = np.prod(shape) * bytes_per_element
    return total_bytes / 1024**3

def optimal_block_size(shape: Tuple[int, ...], memory_limit_gb: float) -> int:
    """Calcule la taille optimale des blocs."""
    # Basé sur la mémoire disponible et la forme
    total_voxels = np.prod(shape)
    
    # Cible: chaque bloc doit tenir dans 10% de la mémoire disponible
    target_block_voxels = (memory_limit_gb * 0.1) * (1024**3) / 4  # Pour float32
    
    # Calculer la dimension moyenne
    avg_dim = np.mean(shape)
    
    # Bloquer à une taille raisonnable
    block_size = int((target_block_voxels ** (1/3)) * 0.8)
    
    # Limites
    block_size = max(16, min(128, block_size))
    
    # Ajuster pour les dimensions petites
    for dim in shape:
        if dim < block_size * 2:
            block_size = max(16, dim // 4)
    
    logger.info(f"Optimal block size for shape {shape}: {block_size}")
    return block_size

def detect_image_type(data: np.ndarray) -> Tuple[bool, float]:
    """Détecte le type d'image et calcule un seuil optimal."""
    # Analyser l'histogramme
    data_flat = data.flatten()
    
    # Vérifier les caractéristiques CT
    has_negative = np.sum(data_flat < -50) > len(data_flat) * 0.01
    wide_range = np.percentile(data_flat, 99) - np.percentile(data_flat, 1) > 1000
    
    is_ct = has_negative or wide_range
    
    if is_ct:
        # Méthode adaptative pour CT
        # Filtrer les valeurs extrêmes
        valid_values = data_flat[(data_flat > -200) & (data_flat < 3000)]
        
        if len(valid_values) > 1000:
            # Utiliser la méthode d'Otsu approximative
            hist, bins = np.histogram(valid_values, bins=256)
            
            # Normaliser l'histogramme
            hist = hist.astype(float) / hist.sum()
            
            # Trouver le seuil optimal (simplifié)
            cumsum = np.cumsum(hist)
            mean = np.cumsum(hist * bins[:-1]) / cumsum
            
            variance = np.zeros_like(hist)
            for i in range(len(hist)):
                if cumsum[i] > 0 and cumsum[i] < 1:
                    variance[i] = (cumsum[i] * (1 - cumsum[i]) * 
                                 (mean[i] - mean[-1])**2)
            
            threshold_idx = np.argmax(variance)
            threshold = bins[threshold_idx]
        else:
            threshold = 200  # Valeur par défaut pour l'os
    else:
        # Image binaire ou segmentée
        threshold = 0.5
    
    return is_ct, threshold

def load_nifti_intelligent(path: str, max_memory_gb: float = 4.0) -> Tuple[np.ndarray, Dict]:
    """Charge un fichier NIfTI de manière intelligente."""
    logger.info(f"Loading NIfTI: {path}")
    
    with memory_context("load_nifti"):
        img = nib.load(path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        shape = data.shape
        dtype = data.dtype
        spacing = header.get_zooms()[:3]
        
        memory_estimate = estimate_array_memory(shape, dtype)
        logger.info(f"  Shape: {shape}, Spacing: {spacing}")
        logger.info(f"  Memory estimate: {memory_estimate:.2f} GB")
        
        # Si trop gros, downsampler
        if memory_estimate > max_memory_gb:
            logger.warning(f"Data too large ({memory_estimate:.2f} GB), downsampling...")
            down_factor = int(np.ceil(memory_estimate / max_memory_gb))
            logger.info(f"  Downsampling factor: {down_factor}")
            
            data = zoom(data, 1/down_factor, order=1)
            spacing = [s * down_factor for s in spacing]
        
        metadata = {
            'original_shape': shape,
            'affine': affine,
            'spacing': spacing,
            'dtype': dtype,
            'filename': Path(path).name
        }
        
        return data, metadata

# ============================================================================
# CLASSE PRINCIPALE SDF REFINER
# ============================================================================

class UltraMemoryEfficientSDFRefiner:
    """
    Raffineur SDF ultra-optimisé pour la mémoire.
    Supporte le traitement out-of-core et le streaming.
    """
    
    def __init__(self, memory_manager: MemoryManager, refine_factor: int = 2):
        self.mm = memory_manager
        self.factor = refine_factor
        self.halo = 4
        
        # État interne
        self.current_sdf = None
        self.current_mask = None
        self.iteration = 0
        
        logger.info(f"Initialized SDFRefiner with factor={refine_factor}")
    
    def process_mask(self, mask: np.ndarray) -> MemoryAwareArray:
        """Traite un masque complet."""
        logger.info(f"Processing mask of shape {mask.shape}")
        
        # 1. Nettoyer le masque
        with memory_context("clean_mask"):
            cleaned = self._clean_mask(mask)
        
        # 2. Calculer SDF initiale
        with memory_context("initial_sdf"):
            sdf = self._compute_full_sdf(cleaned)
        
        # 3. Itération de raffinement
        with memory_context("refinement_iteration"):
            sdf = self._refine_sdf(sdf)
        
        self.current_sdf = sdf
        return sdf
    
    def _clean_mask(self, mask: np.ndarray) -> MemoryAwareArray:
        """Nettoie le masque (remplissage + composante principale)."""
        logger.info("Cleaning mask...")
        
        # Créer un tableau géré
        mask_array = self.mm.create_array(mask.shape, dtype=np.bool_, name="mask_cleaned")
        
        # Si petit, traiter en mémoire
        if mask_array.size_gb < 2.0:
            mask_data = mask.astype(bool)
            
            # Remplir les trous
            mask_filled = binary_fill_holes(mask_data)
            
            # Composante principale
            labels, n_components = label(mask_filled)
            if n_components > 1:
                sizes = np.bincount(labels.ravel())
                # Ignorer le background (label 0)
                main_component = np.argmax(sizes[1:]) + 1
                mask_filled = labels == main_component
            
            mask_array[:] = mask_filled
            
        else:
            # Traitement par blocs pour gros volumes
            self._clean_mask_blocks(mask, mask_array)
        
        return mask_array
    
    def _clean_mask_blocks(self, mask: np.ndarray, output: MemoryAwareArray):
        """Nettoie le masque par blocs."""
        block_size = optimal_block_size(mask.shape, self.mm.config.limit_gb)
        
        logger.info(f"Cleaning mask by blocks (size={block_size})...")
        
        for i in range(0, mask.shape[0], block_size):
            for j in range(0, mask.shape[1], block_size):
                for k in range(0, mask.shape[2], block_size):
                    # Définir le bloc
                    i_end = min(i + block_size, mask.shape[0])
                    j_end = min(j + block_size, mask.shape[1])
                    k_end = min(k + block_size, mask.shape[2])
                    
                    block_slice = (slice(i, i_end), slice(j, j_end), slice(k, k_end))
                    
                    # Extraire et traiter le bloc
                    block = mask[block_slice]
                    block_filled = binary_fill_holes(block)
                    
                    # Stocker le résultat
                    output[block_slice] = block_filled
        
        # Composante principale (nécessite traitement global)
        # Pour les très gros volumes, on saute cette étape
        if output.size_gb < 10.0:
            logger.info("Finding main component...")
            mask_data = output.to_numpy()
            labels, n_components = label(mask_data)
            
            if n_components > 1:
                sizes = np.bincount(labels.ravel())
                main_component = np.argmax(sizes[1:]) + 1
                output[:] = labels == main_component
    
    def _compute_full_sdf(self, mask: MemoryAwareArray) -> MemoryAwareArray:
        """Calcule la SDF complète."""
        logger.info("Computing full SDF...")
        
        sdf_array = self.mm.create_array(mask.shape, dtype=np.float32, name="sdf")
        
        # Si petit, calculer en mémoire
        if mask.size_gb < 2.0:
            mask_data = mask.to_numpy()
            
            pos = distance_transform_edt(~mask_data)
            neg = distance_transform_edt(mask_data)
            sdf = pos - neg
            
            sdf_array[:] = sdf.astype(np.float32)
            
        else:
            # Calcul par blocs avec halo
            self._compute_sdf_blocks(mask, sdf_array)
        
        return sdf_array
    
    def _compute_sdf_blocks(self, mask: MemoryAwareArray, output: MemoryAwareArray):
        """Calcule la SDF par blocs."""
        block_size = optimal_block_size(mask.shape, self.mm.config.limit_gb)
        
        logger.info(f"Computing SDF by blocks (size={block_size})...")
        
        for i in range(0, mask.shape[0], block_size):
            for j in range(0, mask.shape[1], block_size):
                for k in range(0, mask.shape[2], block_size):
                    # Bloc avec halo
                    i_start = max(i - self.halo, 0)
                    j_start = max(j - self.halo, 0)
                    k_start = max(k - self.halo, 0)
                    
                    i_end = min(i + block_size + self.halo, mask.shape[0])
                    j_end = min(j + block_size + self.halo, mask.shape[1])
                    k_end = min(k + block_size + self.halo, mask.shape[2])
                    
                    halo_slice = (slice(i_start, i_end), 
                                 slice(j_start, j_end), 
                                 slice(k_start, k_end))
                    
                    # Extraire le bloc avec halo
                    mask_block = mask[halo_slice]
                    
                    # Calculer SDF sur le bloc avec halo
                    pos = distance_transform_edt(~mask_block)
                    neg = distance_transform_edt(mask_block)
                    sdf_block = (pos - neg).astype(np.float32)
                    
                    # Extraire la partie centrale sans halo
                    i_center_start = i - i_start
                    j_center_start = j - j_start
                    k_center_start = k - k_start
                    
                    i_center_end = i_center_start + min(block_size, mask.shape[0] - i)
                    j_center_end = j_center_start + min(block_size, mask.shape[1] - j)
                    k_center_end = k_center_start + min(block_size, mask.shape[2] - k)
                    
                    center_slice = (slice(i_center_start, i_center_end),
                                   slice(j_center_start, j_center_end),
                                   slice(k_center_start, k_center_end))
                    
                    sdf_center = sdf_block[center_slice]
                    
                    # Stocker le résultat
                    output_slice = (slice(i, i + sdf_center.shape[0]),
                                   slice(j, j + sdf_center.shape[1]),
                                   slice(k, k + sdf_center.shape[2]))
                    
                    output[output_slice] = sdf_center
        
        logger.info("SDF computation complete")
    
    def _refine_sdf(self, sdf: MemoryAwareArray) -> MemoryAwareArray:
        """Raffine la SDF."""
        logger.info(f"Refining SDF with factor={self.factor}...")
        
        # 1. Identifier la narrow band
        with memory_context("narrow_band"):
            band = self._find_narrow_band(sdf)
        
        # 2. Raffiner par blocs
        with memory_context("block_refinement"):
            refined_mask = self._refine_blocks(sdf, band)
        
        # 3. Calculer nouvelle SDF
        with memory_context("new_sdf"):
            new_sdf = self._compute_full_sdf(refined_mask)
        
        # Nettoyer
        band.free_memory()
        refined_mask.free_memory()
        
        return new_sdf
    
    def _find_narrow_band(self, sdf: MemoryAwareArray) -> MemoryAwareArray:
        """Trouve la narrow band."""
        logger.info("Finding narrow band...")
        
        dmax = max(2, self.factor)
        band = self.mm.create_array(sdf.shape, dtype=np.bool_, name="narrow_band")
        
        # Si petit, traiter en mémoire
        if sdf.size_gb < 4.0:
            sdf_data = sdf.to_numpy()
            band_data = np.abs(sdf_data) < dmax
            band[:] = band_data
        else:
            # Par blocs
            block_size = optimal_block_size(sdf.shape, self.mm.config.limit_gb)
            
            for i in range(0, sdf.shape[0], block_size):
                for j in range(0, sdf.shape[1], block_size):
                    for k in range(0, sdf.shape[2], block_size):
                        block_slice = (slice(i, min(i + block_size, sdf.shape[0])),
                                      slice(j, min(j + block_size, sdf.shape[1])),
                                      slice(k, min(k + block_size, sdf.shape[2])))
                        
                        sdf_block = sdf[block_slice]
                        band_block = np.abs(sdf_block) < dmax
                        band[block_slice] = band_block
        
        return band
    
    def _refine_blocks(self, sdf: MemoryAwareArray, band: MemoryAwareArray) -> MemoryAwareArray:
        """Raffine par blocs avec blending."""
        logger.info("Refining blocks with blending...")
        
        # Taille du résultat raffiné
        new_shape = tuple(s * self.factor for s in sdf.shape)
        refined = self.mm.create_array(new_shape, dtype=np.bool_, name="refined_mask")
        weight_sum = self.mm.create_array(new_shape, dtype=np.float32, name="weight_sum")
        value_sum = self.mm.create_array(new_shape, dtype=np.float32, name="value_sum")
        
        # Taille des blocs adaptée
        block_size = optimal_block_size(sdf.shape, self.mm.config.limit_gb)
        
        for i in range(0, sdf.shape[0], block_size):
            for j in range(0, sdf.shape[1], block_size):
                for k in range(0, sdf.shape[2], block_size):
                    block_slice = (slice(i, min(i + block_size, sdf.shape[0])),
                                  slice(j, min(j + block_size, sdf.shape[1])),
                                  slice(k, min(k + block_size, sdf.shape[2])))
                    
                    # Vérifier si le bloc est dans la narrow band
                    if np.any(band[block_slice]):
                        # Extraire et raffiner
                        sdf_block = sdf[block_slice]
                        refined_block = zoom(sdf_block < 0, self.factor, order=1) > 0.5
                        
                        # Coordonnées dans le volume raffiné
                        tgt_slice = (slice(i * self.factor, i * self.factor + refined_block.shape[0]),
                                    slice(j * self.factor, j * self.factor + refined_block.shape[1]),
                                    slice(k * self.factor, k * self.factor + refined_block.shape[2]))
                        
                        # Poids de blending
                        weight = self._compute_blending_weight(refined_block.shape)
                        
                        # Accumuler
                        value_sum[tgt_slice] += refined_block.astype(np.float32) * weight
                        weight_sum[tgt_slice] += weight
        
        # Fusionner avec vote majoritaire
        logger.info("Merging blocks...")
        weight_data = weight_sum.to_numpy()
        value_data = value_sum.to_numpy()
        
        # Éviter la division par zéro
        weight_data[weight_data == 0] = 1
        refined_data = (value_data / weight_data) > 0.5
        
        refined[:] = refined_data
        
        # Nettoyer
        weight_sum.free_memory()
        value_sum.free_memory()
        
        return refined
    
    def _compute_blending_weight(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Calcule les poids de blending pour un bloc."""
        w = np.ones(shape, dtype=np.float32)
        h = self.halo * self.factor
        
        for d in range(3):
            if shape[d] > 2 * h:
                # Fenêtre sigmoïde
                x = np.linspace(-3, 3, h)
                window = 1 / (1 + np.exp(-x))
                
                # Appliquer au début
                slices_start = [slice(None)] * 3
                slices_start[d] = slice(0, h)
                w[tuple(slices_start)] *= window.reshape(
                    [-1 if i == d else 1 for i in range(3)]
                )
                
                # Appliquer à la fin
                slices_end = [slice(None)] * 3
                slices_end[d] = slice(-h, None)
                w[tuple(slices_end)] *= window[::-1].reshape(
                    [-1 if i == d else 1 for i in range(3)]
                )
        
        # Normaliser
        w_max = np.max(w)
        if w_max > 0:
            w = w / w_max
        
        return w
    
    def get_sdf(self) -> Optional[MemoryAwareArray]:
        """Retourne la SDF courante."""
        return self.current_sdf
    
    def cleanup(self):
        """Nettoie toutes les ressources."""
        if self.current_sdf:
            self.current_sdf.free_memory()
        if self.current_mask:
            self.current_mask.free_memory()
        
        force_garbage_collection()

# ============================================================================
# EXTRACTION DE SURFACE STREAMING
# ============================================================================

class StreamingSurfaceExtractor:
    """Extrait la surface depuis de gros volumes par streaming."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.mm = memory_manager
    
    def extract_from_array(self, sdf: MemoryAwareArray, spacing: Tuple[float, ...], 
                          level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Extrait la surface depuis un tableau."""
        logger.info(f"Extracting surface from array {sdf.shape}...")
        
        # Si petit, extraction directe
        if sdf.size_gb < 4.0:
            return self._extract_direct(sdf, spacing, level)
        else:
            return self._extract_streaming(sdf, spacing, level)
    
    def _extract_direct(self, sdf: MemoryAwareArray, spacing: Tuple[float, ...], 
                       level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extraction directe en mémoire."""
        sdf_data = sdf.to_numpy()
        
        try:
            verts, faces, normals, values = marching_cubes(
                sdf_data, 
                level=level, 
                spacing=spacing,
                gradient_direction='descent',
                step_size=1,
                allow_degenerate=True
            )
            
            logger.info(f"Extracted {len(verts)} vertices, {len(faces)} faces")
            return verts, faces
            
        except ValueError as e:
            logger.error(f"Marching cubes failed: {e}")
            raise
    
    def _extract_streaming(self, sdf: MemoryAwareArray, spacing: Tuple[float, ...], 
                          level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extraction par streaming pour gros volumes."""
        chunk_size = 128  # Taille des chunks en Z
        all_verts = []
        all_faces = []
        vertex_offset = 0
        
        logger.info(f"Streaming extraction with chunk size {chunk_size}...")
        
        for z_start in range(0, sdf.shape[2], chunk_size):
            z_end = min(z_start + chunk_size + 1, sdf.shape[2])
            
            logger.info(f"  Processing chunk Z={z_start}-{z_end}")
            
            # Extraire le chunk
            chunk_slice = (slice(None), slice(None), slice(z_start, z_end))
            chunk = sdf[chunk_slice]
            
            try:
                # Extraire la surface dans ce chunk
                verts, faces, _, _ = marching_cubes(
                    chunk, 
                    level=level, 
                    spacing=spacing,
                    step_size=1
                )
                
                if len(verts) > 0:
                    # Ajuster les coordonnées Z
                    verts[:, 2] += z_start * spacing[2]
                    
                    # Ajuster les indices des faces
                    faces += vertex_offset
                    vertex_offset += len(verts)
                    
                    all_verts.append(verts)
                    all_faces.append(faces)
                    
                    logger.info(f"    Found {len(verts)} vertices in chunk")
                
            except ValueError:
                # Pas de surface dans ce chunk
                pass
            
            # Libérer le chunk
            del chunk
            force_garbage_collection()
        
        if not all_verts:
            raise ValueError("No surface found in the volume")
        
        # Concaténer tous les résultats
        final_verts = np.vstack(all_verts)
        final_faces = np.vstack(all_faces)
        
        logger.info(f"Total extracted: {len(final_verts)} vertices, {len(final_faces)} faces")
        
        return final_verts, final_faces

# ============================================================================
# POST-PROCESSING DU MAILLAGE
# ============================================================================

class MeshPostProcessor:
    """Post-processeur de maillage pour CFD."""
    
    @staticmethod
    def process(mesh: trimesh.Trimesh, target_faces: int = None) -> trimesh.Trimesh:
        """Post-process un maillage pour CFD."""
        logger.info("Post-processing mesh...")
        
        # 1. Nettoyage de base
        with memory_context("mesh_cleaning"):
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
        
        # 2. Vérification de l'orientation
        if not mesh.is_winding_consistent:
            logger.info("Fixing normals...")
            mesh.fix_normals()
        
        # 3. Fermeture des trous
        if not mesh.is_watertight:
            logger.info("Filling holes...")
            try:
                mesh.fill_holes()
            except Exception as e:
                logger.warning(f"Failed to fill holes: {e}")
        
        # 4. Simplification si nécessaire
        if target_faces and len(mesh.faces) > target_faces * 1.5:
            logger.info(f"Simplifying mesh: {len(mesh.faces):,} -> {target_faces:,} faces")
            mesh = mesh.simplify_quadric_decimation(target_faces)
        
        # 5. Lissage léger
        if len(mesh.vertices) < 500000:  # Seulement si pas trop gros
            logger.info("Light smoothing...")
            try:
                mesh = mesh.smoothed()
            except:
                pass
        
        logger.info(f"Post-processed mesh: {len(mesh.vertices):,} vertices, "
                   f"{len(mesh.faces):,} faces, watertight: {mesh.is_watertight}")
        
        return mesh
    
    @staticmethod
    def compute_quality_metrics(mesh: trimesh.Trimesh) -> Dict[str, float]:
        """Calcule les métriques de qualité du maillage."""
        metrics = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'volume': float(mesh.volume),
            'area': float(mesh.area),
            'watertight': mesh.is_watertight,
            'euler_characteristic': mesh.euler_number
        }
        
        # Qualité des triangles si le maillage n'est pas trop gros
        if len(mesh.faces) < 100000:
            triangles = mesh.triangles
            a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
            b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
            c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
            
            s = (a + b + c) / 2
            area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
            quality = 4 * np.sqrt(3) * area / (a**2 + b**2 + c**2 + 1e-8)
            
            metrics.update({
                'triangle_quality_min': float(np.min(quality)),
                'triangle_quality_mean': float(np.mean(quality)),
                'triangle_quality_std': float(np.std(quality))
            })
        
        return metrics

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

class UltraMemoryEfficientCFDPipeline:
    """
    Pipeline CFD complet ultra-optimisé pour la mémoire.
    """
    
    def __init__(self, memory_limit_gb: float = None, n_jobs: int = -1):
        self.memory_config = MemoryConfig(limit_gb=memory_limit_gb)
        self.memory_manager = MemoryManager(self.memory_config)
        self.n_jobs = n_jobs
        
        logger.info(f"Initialized CFD Pipeline with memory limit: "
                   f"{self.memory_config.limit_gb:.1f} GB")
    
    def process(self, input_path: str, output_path: str, 
                factor: int = 2, hu_threshold: Optional[float] = None,
                target_resolution_mm: Optional[float] = None) -> Dict[str, Any]:
        """
        Traite un fichier NIfTI complet.
        
        Args:
            input_path: Chemin vers le fichier NIfTI
            output_path: Chemin de sortie pour le fichier STL
            factor: Facteur de raffinement
            hu_threshold: Seuil HU (None pour auto-détection)
            target_resolution_mm: Résolution cible en mm
            
        Returns:
            Dictionnaire avec statistiques et métriques
        """
        start_time = time.time()
        stats = {
            'input_file': input_path,
            'output_file': output_path,
            'factor': factor,
            'success': False
        }
        
        try:
            # 1. CHARGEMENT
            logger.info("\n" + "="*70)
            logger.info("STEP 1: LOADING AND PREPROCESSING")
            logger.info("="*70)
            
            with memory_context("loading"):
                # Charger l'image
                data, metadata = load_nifti_intelligent(
                    input_path, 
                    max_memory_gb=self.memory_config.limit_gb * 0.3
                )
                
                # Mettre à jour les stats
                stats.update({
                    'original_shape': metadata['original_shape'],
                    'processed_shape': data.shape,
                    'spacing': metadata['spacing'],
                    'data_type': str(metadata['dtype'])
                })
            
            # 2. SEGMENTATION
            logger.info("\n" + "="*70)
            logger.info("STEP 2: SEGMENTATION")
            logger.info("="*70)
            
            with memory_context("segmentation"):
                # Détecter le type d'image et le seuil
                is_ct, auto_threshold = detect_image_type(data)
                
                if hu_threshold is None:
                    hu_threshold = auto_threshold
                
                logger.info(f"Image type: {'CT' if is_ct else 'Binary/Segmentation'}")
                logger.info(f"Threshold: {hu_threshold}")
                
                # Appliquer le seuil
                if is_ct:
                    # Lissage léger pour CT
                    sigma = min(1.0, np.mean(metadata['spacing']))
                    data_smoothed = gaussian_filter(data, sigma=sigma)
                    mask = data_smoothed >= hu_threshold
                else:
                    mask = data > 0.5
                
                stats['mask_voxels'] = np.sum(mask)
                stats['mask_percentage'] = np.sum(mask) / mask.size * 100
                
                # Libérer les données originales
                del data
                if is_ct:
                    del data_smoothed
                force_garbage_collection()
            
            # 3. RAFFINEMENT SDF
            logger.info("\n" + "="*70)
            logger.info("STEP 3: SDF REFINEMENT")
            logger.info("="*70)
            
            with memory_context("sdf_refinement"):
                # Initialiser le raffineur
                refiner = UltraMemoryEfficientSDFRefiner(
                    self.memory_manager, 
                    refine_factor=factor
                )
                
                # Traiter le masque
                sdf = refiner.process_mask(mask)
                
                stats['sdf_shape'] = sdf.shape
                stats['sdf_memory_gb'] = sdf.size_gb
            
            # 4. EXTRACTION DE SURFACE
            logger.info("\n" + "="*70)
            logger.info("STEP 4: SURFACE EXTRACTION")
            logger.info("="*70)
            
            with memory_context("surface_extraction"):
                # Calculer le nouvel espacement
                new_spacing = [s / factor for s in metadata['spacing']]
                
                # Extraire la surface
                extractor = StreamingSurfaceExtractor(self.memory_manager)
                verts, faces = extractor.extract_from_array(sdf, new_spacing, level=0.0)
                
                stats['extracted_vertices'] = len(verts)
                stats['extracted_faces'] = len(faces)
            
            # 5. POST-PROCESSING
            logger.info("\n" + "="*70)
            logger.info("STEP 5: MESH POST-PROCESSING")
            logger.info("="*70)
            
            with memory_context("post_processing"):
                # Créer le maillage
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                
                # Déterminer le nombre cible de faces
                if target_resolution_mm:
                    # Basé sur la résolution cible
                    current_res = np.mean(new_spacing)
                    if current_res > target_resolution_mm * 1.5:
                        # Besoin de plus de détails
                        target_faces = int(len(faces) * (current_res / target_resolution_mm) ** 2)
                    else:
                        target_faces = None
                else:
                    # Limiter à 500k faces pour CFD
                    target_faces = min(500000, len(faces))
                
                # Post-processer
                post_processor = MeshPostProcessor()
                mesh = post_processor.process(mesh, target_faces)
                
                # Calculer les métriques
                quality_metrics = post_processor.compute_quality_metrics(mesh)
                stats.update(quality_metrics)
            
            # 6. EXPORT
            logger.info("\n" + "="*70)
            logger.info("STEP 6: EXPORT")
            logger.info("="*70)
            
            with memory_context("export"):
                # Exporter le maillage
                mesh.export(output_path)
                
                # Exporter également en format lisible (OBJ)
                obj_path = output_path.replace('.stl', '.obj')
                mesh.export(obj_path)
                
                logger.info(f"Exported STL: {output_path}")
                logger.info(f"Exported OBJ: {obj_path}")
            
            # 7. RAPPORT FINAL
            total_time = time.time() - start_time
            memory_usage = self.memory_manager.get_memory_usage()
            
            stats.update({
                'success': True,
                'total_time_seconds': total_time,
                'peak_memory_gb': memory_usage['peak_gb'],
                'final_memory_gb': memory_usage['process_rss_gb']
            })
            
            self._print_final_report(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            stats['error'] = str(e)
            stats['success'] = False
            return stats
            
        finally:
            # Nettoyage final
            self.memory_manager.cleanup()
    
    def _print_final_report(self, stats: Dict[str, Any]):
        """Affiche le rapport final."""
        logger.info("\n" + "="*70)
        logger.info("FINAL REPORT")
        logger.info("="*70)
        
        logger.info(f"📊 Processing Statistics:")
        logger.info(f"   Input file: {stats.get('input_file', 'N/A')}")
        logger.info(f"   Output file: {stats.get('output_file', 'N/A')}")
        logger.info(f"   Success: {stats.get('success', False)}")
        
        if stats.get('success'):
            logger.info(f"\n📐 Geometry:")
            logger.info(f"   Original shape: {stats.get('original_shape', 'N/A')}")
            logger.info(f"   Final vertices: {stats.get('vertices', 0):,}")
            logger.info(f"   Final faces: {stats.get('faces', 0):,}")
            logger.info(f"   Volume: {stats.get('volume', 0):.1f} mm³")
            logger.info(f"   Surface area: {stats.get('area', 0):.1f} mm²")
            logger.info(f"   Watertight: {stats.get('watertight', False)}")
            
            if 'triangle_quality_mean' in stats:
                logger.info(f"   Triangle quality: {stats['triangle_quality_mean']:.3f} "
                           f"(min: {stats.get('triangle_quality_min', 0):.3f})")
        
        logger.info(f"\n⚡ Performance:")
        logger.info(f"   Total time: {stats.get('total_time_seconds', 0):.1f} s")
        logger.info(f"   Peak memory: {stats.get('peak_memory_gb', 0):.2f} GB")
        logger.info(f"   Final memory: {stats.get('final_memory_gb', 0):.2f} GB")
        
        logger.info("="*70)
    
    def batch_process(self, input_files: List[str], output_dir: str, 
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Traite plusieurs fichiers en batch.
        
        Args:
            input_files: Liste des fichiers d'entrée
            output_dir: Répertoire de sortie
            **kwargs: Arguments supplémentaires pour process()
            
        Returns:
            Liste des statistiques pour chaque fichier
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, input_file in enumerate(input_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {i}/{len(input_files)}")
            logger.info(f"File: {input_file}")
            logger.info(f"{'='*60}")
            
            try:
                # Générer le nom de sortie
                input_name = Path(input_file).stem
                if input_name.endswith('.nii'):
                    input_name = input_name[:-4]
                
                output_file = str(Path(output_dir) / f"{input_name}_cfd.stl")
                
                # Traiter le fichier
                stats = self.process(input_file, output_file, **kwargs)
                results.append(stats)
                
                if stats.get('success'):
                    logger.info(f"✅ File {i} processed successfully")
                else:
                    logger.error(f"❌ File {i} failed: {stats.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ File {i} failed with exception: {e}")
                results.append({
                    'input_file': input_file,
                    'success': False,
                    'error': str(e)
                })
        
        # Résumé du batch
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[Dict[str, Any]]):
        """Affiche le résumé du batch."""
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        logger.info("\n" + "="*70)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*70)
        
        logger.info(f"Total files: {len(results)}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        
        if failed > 0:
            logger.info("\nFailed files:")
            for result in results:
                if not result.get('success', False):
                    logger.info(f"  • {result.get('input_file', 'Unknown')}: "
                               f"{result.get('error', 'Unknown error')}")
        
        logger.info("="*70)

# ============================================================================
# FONCTIONS D'INTERFACE SIMPLIFIÉE
# ============================================================================

def run_cfd_pipeline(
    input_path: str,
    output_path: str,
    factor: int = 2,
    hu_threshold: Optional[float] = None,
    memory_limit_gb: Optional[float] = None,
    target_resolution_mm: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fonction simplifiée pour exécuter le pipeline CFD.
    
    Args:
        input_path: Chemin du fichier NIfTI d'entrée
        output_path: Chemin du fichier STL de sortie
        factor: Facteur de raffinement (1, 2, 3, 4)
        hu_threshold: Seuil HU (None pour auto-détection)
        memory_limit_gb: Limite mémoire en GB (None pour auto)
        target_resolution_mm: Résolution cible en mm
        verbose: Afficher les logs détaillés
    
    Returns:
        Dictionnaire avec les résultats
    """
    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info(f"Starting CFD Pipeline")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Factor: {factor}")
    
    # Valider les paramètres
    if factor not in [1, 2, 3, 4]:
        logger.warning(f"Factor {factor} not recommended, using 2 instead")
        factor = 2
    
    # Créer et exécuter le pipeline
    pipeline = UltraMemoryEfficientCFDPipeline(memory_limit_gb=memory_limit_gb)
    
    try:
        results = pipeline.process(
            input_path=input_path,
            output_path=output_path,
            factor=factor,
            hu_threshold=hu_threshold,
            target_resolution_mm=target_resolution_mm
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_file': input_path,
            'output_file': output_path
        }

def batch_cfd_pipeline(
    input_files: List[str],
    output_dir: str,
    factor: int = 2,
    hu_threshold: Optional[float] = None,
    memory_limit_gb: Optional[float] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Exécute le pipeline CFD en batch.
    
    Args:
        input_files: Liste des fichiers d'entrée
        output_dir: Répertoire de sortie
        factor: Facteur de raffinement
        hu_threshold: Seuil HU
        memory_limit_gb: Limite mémoire
        **kwargs: Arguments supplémentaires
    
    Returns:
        Liste des résultats
    """
    logger.info(f"Starting Batch CFD Pipeline")
    logger.info(f"  Input files: {len(input_files)}")
    logger.info(f"  Output directory: {output_dir}")
    
    pipeline = UltraMemoryEfficientCFDPipeline(memory_limit_gb=memory_limit_gb)
    
    return pipeline.batch_process(
        input_files=input_files,
        output_dir=output_dir,
        factor=factor,
        hu_threshold=hu_threshold,
        **kwargs
    )

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Exemples d'utilisation du pipeline CFD.
    """
    
    # Exemple 1: Traitement simple
    print("Example 1: Simple processing")
    results = run_cfd_pipeline(
        input_path="example_scan.nii.gz",
        output_path="output_mesh.stl",
        factor=2,
        hu_threshold=250,
        memory_limit_gb=16.0,
        verbose=True
    )
    
    if results.get('success'):
        print(f"✅ Success! Mesh saved to {results['output_file']}")
        print(f"   Vertices: {results.get('vertices', 0):,}")
        print(f"   Faces: {results.get('faces', 0):,}")
        print(f"   Volume: {results.get('volume', 0):.1f} mm³")
    else:
        print(f"❌ Failed: {results.get('error', 'Unknown error')}")
    
    # Exemple 2: Traitement batch
    # print("\nExample 2: Batch processing")
    # input_files = [
    #     "data/patient1.nii.gz",
    #     "data/patient2.nii.gz",
    #     "data/patient3.nii.gz"
    # ]
    # 
    # batch_results = batch_cfd_pipeline(
    #     input_files=input_files,
    #     output_dir="output_meshes",
    #     factor=2,
    #     hu_threshold=None,  # Auto-détection
    #     memory_limit_gb=32.0
    # )
    # 
    # print(f"Batch completed: {len([r for r in batch_results if r['success']])}/{len(batch_results)} successful")
    
    # Exemple 3: Avec résolution cible
    # print("\nExample 3: Target resolution")
    # results = run_cfd_pipeline(
    #     input_path="high_res_scan.nii.gz",
    #     output_path="high_res_mesh.stl",
    #     factor=4,
    #     target_resolution_mm=0.2,  # Cible 0.2 mm
    #     memory_limit_gb=32.0
    # )

# ============================================================================
# CONFIGURATION AUTOMATIQUE
# ============================================================================

def auto_configure_pipeline() -> Dict[str, Any]:
    """
    Configure automatiquement le pipeline basé sur le système.
    
    Returns:
        Configuration optimale
    """
    import multiprocessing
    
    # Analyser le système
    system_memory_gb = psutil.virtual_memory().total / 1024**3
    available_memory_gb = psutil.virtual_memory().available / 1024**3
    cpu_count = multiprocessing.cpu_count()
    
    # Déterminer la configuration
    config = {
        'memory_limit_gb': min(system_memory_gb * 0.7, available_memory_gb * 0.9),
        'n_jobs': min(cpu_count - 1, 8) if cpu_count > 1 else 1,
        'recommended_factor': 2 if system_memory_gb >= 16 else 1,
        'chunk_size': 128 if system_memory_gb >= 32 else 64,
        'compression_level': 3 if system_memory_gb >= 16 else 1
    }
    
    logger.info("Auto-configuration:")
    logger.info(f"  System memory: {system_memory_gb:.1f} GB")
    logger.info(f"  Available memory: {available_memory_gb:.1f} GB")
    logger.info(f"  CPU cores: {cpu_count}")
    logger.info(f"  Recommended memory limit: {config['memory_limit_gb']:.1f} GB")
    logger.info(f"  Recommended factor: {config['recommended_factor']}")
    
    return config

# Exécuter la configuration automatique
if __name__ == "__main__":
    config = auto_configure_pipeline()
    print(f"\nAuto-configuration complete:")
    for key, value in config.items():
        print(f"  {key}: {value}")