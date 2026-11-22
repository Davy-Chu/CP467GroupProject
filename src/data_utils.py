"""Load and validate the Kaggle GTSRB layout without changing raw files.

Required: Train.csv, Test.csv, Train/, Test/. CSVs need Path and ClassId.
Meta.csv describes sign classes; it cannot replace image-level labels.
Images returned by load_image are RGB uint8, not OpenCV's default BGR.
"""

import hashlib
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    EXPECTED_TEST_IMAGES, EXPECTED_TRAIN_IMAGES, IMAGE_SIZE, NUM_CLASSES,
    RANDOM_SEED, VALIDATION_FRACTION,
)


class DatasetValidationError(ValueError):
    """The selected dataset cannot safely be used for this project."""


def resolve_image_path(data_root, relative_path):
    """Resolve CSV paths relative to raw/, including Windows CSV separators."""
    root = Path(data_root).expanduser().resolve()
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise DatasetValidationError(f"Expected a relative image path inside raw/: {relative_path!r}")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise DatasetValidationError(f"Image path leaves the dataset folder: {relative_path!r}")
    return path


def load_metadata(data_root, split="Train"):
    """Read labeled metadata in stable path order (Train, Test, or Meta)."""
    if split not in ("Train", "Test", "Meta"):
        raise ValueError("split must be Train, Test, or Meta")
    csv_path = Path(data_root) / f"{split}.csv"
    if not csv_path.is_file():
        raise DatasetValidationError(f"Missing label file: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DatasetValidationError(f"Cannot read {csv_path}: {exc}") from exc
    missing = {"Path", "ClassId"} - set(frame.columns)
    if missing:
        raise DatasetValidationError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    if frame.empty or frame[["Path", "ClassId"]].isna().any().any():
        raise DatasetValidationError(f"{csv_path.name} has empty or missing paths/labels.")
    labels = pd.to_numeric(frame["ClassId"], errors="coerce")
    if not (labels.notna() & labels.between(0, NUM_CLASSES - 1) & (labels % 1 == 0)).all():
        raise DatasetValidationError(f"{csv_path.name}: ClassId must be integers 0..{NUM_CLASSES - 1}.")
    frame["ClassId"] = labels.astype(int)
    frame["Path"] = frame["Path"].astype(str).str.replace("\\", "/", regex=False)
    for value in frame["Path"]:
        path = resolve_image_path(data_root, value)
        canonical = path.relative_to(Path(data_root).expanduser().resolve()).as_posix()
        if canonical != value or not value.startswith(f"{split}/"):
            raise DatasetValidationError(f"{csv_path.name}: expected {split}/... path, got {value!r}.")
    if frame["Path"].duplicated().any():
        raise DatasetValidationError(f"{csv_path.name} contains duplicate image paths.")
    return frame.sort_values("Path").reset_index(drop=True)


def load_image(data_root, relative_path, image_size=IMAGE_SIZE):
    """Load an RGB uint8 image; image_size=None preserves the original size."""
    path = resolve_image_path(data_root, relative_path)
    # imdecode also handles non-ASCII Windows/Drive paths reliably.
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise DatasetValidationError(f"Cannot load image: {path}") from exc
    if image is None:
        raise DatasetValidationError(f"Unreadable image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return image


def validate_dataset(data_root, check_images="sample", require_full_counts=True):
    """Fail clearly on missing labels/images/classes or incomplete uploads.

    Always checks every CSV image path exists and both splits cover 43 classes.
    check_images='sample' decodes one image per class per split; 'all' decodes
    every referenced image (slower on Drive). Full Kaggle counts are required
    by default. Disable counts only for small development fixtures, never for
    the project's readiness check. Returns validated Train and Test frames.
    """
    if check_images not in ("sample", "all"):
        raise ValueError("check_images must be 'sample' or 'all'")
    root = Path(data_root).expanduser().resolve()
    print(f"Dataset root: {root}")
    if not root.is_dir():
        raise DatasetValidationError(
            f"Dataset directory does not exist: {root}\n"
            "Mount Drive and set DRIVE_DATA_ROOT to the folder containing Train.csv and Test.csv."
        )
    errors = []
    frames = {}
    for split, expected in (("Train", EXPECTED_TRAIN_IMAGES), ("Test", EXPECTED_TEST_IMAGES)):
        directory = root / split
        if not directory.is_dir():
            errors.append(f"Missing image directory: {split}/")
        else:
            count = sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
            print(f"{split}/: {count:,} PNG images present (expected {expected:,}).")
            if require_full_counts and count != expected:
                errors.append(f"{split}/ has {count:,} PNG images; expected {expected:,}.")
        try:
            frame = load_metadata(root, split)
        except DatasetValidationError as exc:
            errors.append(str(exc))
            continue
        frames[split] = frame
        if require_full_counts and len(frame) != expected:
            errors.append(f"{split}.csv has {len(frame):,} rows; expected {expected:,}.")
        missing_classes = sorted(set(range(NUM_CLASSES)) - set(frame["ClassId"]))
        if missing_classes:
            errors.append(f"{split}.csv missing class IDs: {missing_classes}")
        missing_paths = [p for p in frame["Path"] if not resolve_image_path(root, p).is_file()]
        if missing_paths:
            errors.append(f"{split}: {len(missing_paths):,} referenced images missing; examples: {missing_paths[:5]}")
            continue
        sample = frame if check_images == "all" else frame.groupby("ClassId", sort=True).head(1)
        unreadable = []
        print(f"Decoding {len(sample):,} {split} images ({check_images} check)...")
        for path in sample["Path"]:
            try:
                load_image(root, path, image_size=None)
            except DatasetValidationError:
                unreadable.append(path)
        if unreadable:
            errors.append(f"{split}: {len(unreadable):,} unreadable images; examples: {unreadable[:5]}")
        print(f"{split}.csv: {len(frame):,} rows, {frame['ClassId'].nunique()} classes.")
    if errors:
        raise DatasetValidationError(
            "Dataset is NOT ready:\n- " + "\n- ".join(errors)
            + "\nRestore the original missing GTSRB files or choose a complete Drive copy. "
            "Meta.csv is class metadata, not training/test labels."
        )
    print(f"Dataset validation passed (image decoding: {check_images}).")
    return frames["Train"], frames["Test"]


def make_train_validation_split(training_metadata):
    """One shared stratified 80/20 split, sorted by Path before using seed 42.

    Accepts provided training data only. Both notebooks call this function;
    never pass the official Test.csv or split it for tuning. Sorting makes the
    split independent of CSV row order and the local/Drive absolute root.
    """
    frame = training_metadata.sort_values("Path").reset_index(drop=True).copy()
    if frame.empty or not frame["Path"].str.startswith("Train/").all():
        raise DatasetValidationError("Validation must be split from Train.csv only.")
    if frame["Path"].duplicated().any():
        raise DatasetValidationError("Duplicate training image paths would leak across splits.")
    if set(frame["ClassId"]) != set(range(NUM_CLASSES)):
        raise DatasetValidationError(f"Training split must contain all {NUM_CLASSES} classes.")
    try:
        train, validation = train_test_split(
            frame, test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED,
            stratify=frame["ClassId"],
        )
    except ValueError as exc:
        raise DatasetValidationError(f"Cannot make the shared stratified split: {exc}") from exc
    if any(part["ClassId"].nunique() != NUM_CLASSES for part in (train, validation)):
        raise DatasetValidationError("Too few examples to retain every class in both subsets.")
    return tuple(part.sort_values("Path").reset_index(drop=True) for part in (train, validation))


def split_fingerprint(train, validation):
    """Portable identifier: teammates should see the same ID for their split."""
    digest = hashlib.sha256()
    for name, frame in (("train", train), ("validation", validation)):
        digest.update(name.encode())
        digest.update(frame.sort_values("Path")[["Path", "ClassId"]].to_csv(index=False).encode())
    return digest.hexdigest()[:16]
