import os
import pydicom

def ingest_local_dicom_folder(folder: str) -> dict:
    """
    Local ingestion for quick testing.
    - Scans folder
    - Reads DICOM headers
    - Returns basic metadata (no PHI handling in v1)
    """
    if not os.path.isdir(folder):
        return {"error": "folder_not_found", "folder": folder}

    files = []
    for root, _, fns in os.walk(folder):
        for fn in fns:
            files.append(os.path.join(root, fn))

    meta = []
    for p in files[:300]:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            meta.append({
                "path": p,
                "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
                "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
                "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
                "Modality": getattr(ds, "Modality", None),
                "PatientID": getattr(ds, "PatientID", None),
                "StudyDate": getattr(ds, "StudyDate", None),
            })
        except Exception:
            continue

    return {"folder": folder, "files_scanned": len(files), "items": meta[:60]}
