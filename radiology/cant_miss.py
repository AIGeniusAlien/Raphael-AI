CANT_MISS_BY_STUDY = {
    "CT_HEAD": [
        "Intracranial hemorrhage",
        "Large vessel occlusion / acute ischemic stroke",
        "Mass effect / herniation",
        "Subdural hematoma",
    ],
    "CXR": [
        "Tension pneumothorax",
        "Pneumothorax",
        "Pulmonary edema",
        "Free air (if visible)",
        "Mediastinal widening (aortic catastrophe context)",
    ],
    "CTPE": [
        "Massive pulmonary embolism (right heart strain)",
        "Aortic dissection (if covered)",
        "Pneumothorax",
    ],
}

def get_checklist(study: str) -> list[str]:
    return CANT_MISS_BY_STUDY.get(study.upper(), [])
