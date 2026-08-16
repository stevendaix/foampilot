#!/usr/bin/env python3
"""
Batch TBAD extraction from imageTBAD/
Extracts TL, FL and wall STL for multiple patients.
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_patient(patient_id: int, data_dir: Path, output_dir: Path):
    from data_preproc.extract_tbad_full import extract_lumens, extract_aorta_wall
    
    image_path = data_dir / f"{patient_id}_image.nii.gz"
    label_path = data_dir / f"{patient_id}_label.nii.gz"
    
    if not image_path.exists() or not label_path.exists():
        logger.warning(f"Skipping patient {patient_id}: missing files")
        return False
    
    patient_out = output_dir / f"patient{patient_id}"
    patient_out.mkdir(parents=True, exist_ok=True)
    
    try:
        lumens = extract_lumens(label_path, patient_out, target_tl=30000, target_fl=20000, verbose=False)
        wall = extract_aorta_wall(image_path, label_path, patient_out / f"patient{patient_id}_wall.stl", verbose=False)
        logger.info(f"Patient {patient_id}: OK")
        return True
    except Exception as e:
        logger.error(f"Patient {patient_id}: FAILED - {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="imageTBAD", help="Input NIfTI directory")
    parser.add_argument("--output-dir", default="data_preproc/batch_output", help="Output STL directory")
    parser.add_argument("--patients", nargs="*", type=int, help="Specific patient IDs")
    parser.add_argument("--max-patients", type=int, default=5, help="Max patients to process")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.patients:
        patient_ids = args.patients[:args.max_patients]
    else:
        ids = sorted([int(f.name.split("_")[0]) for f in data_dir.glob("*_image.nii.gz") if f.name.split("_")[0].isdigit()])
        patient_ids = ids[:args.max_patients]
    
    logger.info(f"Processing {len(patient_ids)} patients: {patient_ids}")
    
    results = []
    for pid in patient_ids:
        ok = extract_patient(pid, data_dir, output_dir)
        results.append((pid, ok))
    
    ok_count = sum(1 for _, ok in results if ok)
    logger.info(f"Done: {ok_count}/{len(results)} succeeded")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
