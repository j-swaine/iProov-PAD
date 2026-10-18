"""
Project-wide constants: class labels and folder names.
"""

# Attack is the positive class: every score this package outputs is P(attack).
BONAFIDE = 0
ATTACK = 1
LABEL_NAMES = {BONAFIDE: "bonafide", ATTACK: "attack"}

# Folder names in the supplied dataset (DATA/test/data/<folder>/<id>.png).
CLASS_FOLDERS = {"Bonafide": BONAFIDE, "Screens": ATTACK}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

SPLITS = ("train", "val", "dev", "test")
