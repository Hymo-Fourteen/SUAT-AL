"""Load SYKE plankton datasets using their official image splits.

The official ``splits.json`` controls the train/validation/test assignment of
images.  JODA's semantic roles (initial InD, discoverable and irrelevant OOD)
can be configured independently with explicit original class IDs.  The older
official open-set-trial mode remains available for existing configurations.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter


class PlanktonImageListDataset(Dataset):
    """Image-list dataset exposing the target/count contract used by JODA."""

    def __init__(self, samples: Sequence[Tuple[Path, int]]):
        self.samples = [(str(path), int(target)) for path, target in samples]
        self.targets = [target for _, target in self.samples]
        self.transform = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def get_class_counts(self, indices: List[int]):
        values, counts = np.unique(np.asarray(self.targets)[indices], return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_plankton_pool(dataset: Dict, config: Dict, order: int = 1):
    """Build a JODA pool directly from SYKE split manifests without copying."""

    del dataset, order  # Paths and the official trial live in data_config.
    data_config = config.get("data_config", {})
    image_root = Path(data_config["image_root"])
    split_root = Path(data_config["split_root"])
    discovery_original_ids = [int(value) for value in data_config["discovery_original_class_ids"]]
    initial_per_class = int(data_config["initial_labeled_per_class"])

    split_data = _load_json(split_root / "splits.json")
    categories = {int(key): value for key, value in split_data["categories"].items()}
    image_splits = {int(key): value for key, value in split_data["images"].items()}
    explicit_ind_ids = data_config.get("ind_original_class_ids")
    if explicit_ind_ids is None:
        trial = int(data_config.get("osr_trial", 0))
        trial_data = _load_json(split_root / f"{trial}.json")
        known_original_ids = [int(value) for value in trial_data["Known"]]
        legacy_far_original_ids = [int(value) for value in trial_data["Unknown"]]
        discovery_set = set(discovery_original_ids)
        if not discovery_set or not discovery_set.issubset(known_original_ids):
            raise ValueError(
                "Discovery classes must be a non-empty subset of the official Known classes"
            )
        ind_original_ids = [
            value for value in known_original_ids if value not in discovery_set
        ]
        role_assignment = "official-open-set-trial"
    else:
        ind_original_ids = [int(value) for value in explicit_ind_ids]
        legacy_far_original_ids = []
        role_assignment = "explicit-semantic-classes"

    if not ind_original_ids or not discovery_original_ids:
        raise ValueError("Both initial InD and discovery class lists must be non-empty")
    if len(ind_original_ids) != len(set(ind_original_ids)):
        raise ValueError("ind_original_class_ids contains duplicates")
    if len(discovery_original_ids) != len(set(discovery_original_ids)):
        raise ValueError("discovery_original_class_ids contains duplicates")
    if set(ind_original_ids) & set(discovery_original_ids):
        raise ValueError("Initial InD and discovery classes must be disjoint")
    unknown_target_ids = (
        set(ind_original_ids) | set(discovery_original_ids)
    ) - set(categories)
    if unknown_target_ids:
        raise ValueError(f"Unknown target original class IDs: {sorted(unknown_target_ids)}")
    target_original_ids = ind_original_ids + discovery_original_ids
    target_map = {original_id: new_id for new_id, original_id in enumerate(target_original_ids)}
    expected_ind_ids = list(range(len(ind_original_ids)))
    if list(config["ind_classes"]) != expected_ind_ids:
        raise ValueError(
            f"ind_classes must be {expected_ind_ids} for the configured plankton split, "
            f"got {config['ind_classes']}"
        )

    def split_paths(root, split_categories, split_images, original_id: int, split: str):
        class_name = split_categories[original_id]
        result = [root / class_name / filename for filename in split_images[original_id][split]]
        missing = [str(path) for path in result if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} split images; first: {missing[0]}")
        return result

    def paths(original_id: int, split: str):
        return split_paths(image_root, categories, image_splits, original_id, split)

    initial_samples = []
    remaining_target_samples = []
    for original_id in ind_original_ids:
        class_paths = paths(original_id, "train")
        if len(class_paths) < initial_per_class:
            raise ValueError(
                f"Class {original_id} ({categories[original_id]}) has only {len(class_paths)} "
                f"official training samples; requested {initial_per_class} initial labels"
            )
        initial_samples.extend((path, target_map[original_id]) for path in class_paths[:initial_per_class])
        remaining_target_samples.extend(
            (path, target_map[original_id]) for path in class_paths[initial_per_class:]
        )
    for original_id in discovery_original_ids:
        remaining_target_samples.extend(
            (path, target_map[original_id]) for path in paths(original_id, "train")
        )

    far_data_config = data_config.get("far_data_config")
    if far_data_config:
        far_image_root = Path(far_data_config["image_root"])
        far_split_root = Path(far_data_config["split_root"])
        far_split_data = _load_json(far_split_root / "splits.json")
        far_categories = {
            int(key): value for key, value in far_split_data["categories"].items()
        }
        far_image_splits = {
            int(key): value for key, value in far_split_data["images"].items()
        }
        configured_far_ids = far_data_config.get("original_class_ids")
        if configured_far_ids is None:
            far_trial = int(far_data_config.get("osr_trial", 0))
            far_trial_data = _load_json(far_split_root / f"{far_trial}.json")
            configured_far_ids = far_trial_data["Unknown"]
        far_original_ids = [int(value) for value in configured_far_ids]

        def far_paths(original_id: int, split: str):
            return split_paths(
                far_image_root,
                far_categories,
                far_image_splits,
                original_id,
                split,
            )
    else:
        far_split_root = split_root
        far_categories = categories
        far_paths = paths
        far_original_ids = legacy_far_original_ids

    if not far_original_ids:
        raise ValueError("At least one irrelevant OOD class must be configured")
    if len(far_original_ids) != len(set(far_original_ids)):
        raise ValueError("far original class IDs contain duplicates")
    unknown_far_ids = set(far_original_ids) - set(far_categories)
    if unknown_far_ids:
        raise ValueError(f"Unknown far-OOD original class IDs: {sorted(unknown_far_ids)}")
    if far_split_root.resolve() == split_root.resolve():
        overlap = set(far_original_ids) & set(target_original_ids)
        if overlap:
            raise ValueError(
                "Target and far-OOD roles overlap in the same source: "
                f"{sorted(overlap)}"
            )

    far_splits = tuple(
        far_data_config.get("splits", ("train", "valid", "test"))
        if far_data_config else ("train", "valid", "test")
    )
    invalid_far_splits = set(far_splits) - {"train", "valid", "test"}
    if not far_splits or invalid_far_splits:
        raise ValueError(
            f"far_data_config.splits must use train/valid/test; got {far_splits}"
        )
    max_far_per_class = (
        far_data_config.get("max_samples_per_class") if far_data_config else None
    )
    far_sampling_seed = int(
        far_data_config.get("sampling_seed", 0) if far_data_config else 0
    )
    if max_far_per_class is not None:
        max_far_per_class = int(max_far_per_class)
        if max_far_per_class <= 0:
            raise ValueError("max_samples_per_class must be positive")

    far_samples = []
    for original_id in far_original_ids:
        class_far_paths = []
        for split in far_splits:
            class_far_paths.extend(far_paths(original_id, split))
        if max_far_per_class is not None and len(class_far_paths) > max_far_per_class:
            class_far_paths = random.Random(
                far_sampling_seed + original_id
            ).sample(class_far_paths, max_far_per_class)
        if not class_far_paths:
            raise ValueError(
                f"Far-OOD class {original_id} ({far_categories[original_id]}) has no "
                f"images in splits {far_splits}"
            )
        far_samples.extend((path, -1) for path in class_far_paths)

    pool_samples = initial_samples + remaining_target_samples + far_samples
    training_pool = PlanktonImageListDataset(pool_samples)
    label_idx = list(range(len(initial_samples)))
    unlabeled_idx = list(range(len(initial_samples), len(pool_samples)))

    def target_evaluation_dataset(split: str):
        samples = []
        for original_id in target_original_ids:
            samples.extend((path, target_map[original_id]) for path in paths(original_id, split))
        return PlanktonImageListDataset(samples)

    validation_set = target_evaluation_dataset("valid")
    test_set = target_evaluation_dataset("test")
    sample_size = int(data_config.get("image_size", 64))
    if sample_size != 64:
        raise ValueError(
            "The registered SYKE plankton preprocessing currently produces 64x64 images; "
            f"got image_size={sample_size}"
        )
    feature_sizes = {
        model: [sample_size, sample_size // 2, sample_size // 4, sample_size // 8]
        for model in ("Resnet18", "Resnet34", "Resnet50")
    }
    dataset_config = {
        "num_classes": len(target_original_ids),
        "all_classes": len(target_original_ids),
        "feature_sizes": feature_sizes,
        "encoding_dimension": 240,
        "sample_size": (sample_size, sample_size),
        "class_names": [categories[value] for value in target_original_ids],
        "ind_original_class_ids": ind_original_ids,
        "discovery_original_class_ids": discovery_original_ids,
        "far_original_class_ids": far_original_ids,
        "far_class_names": [far_categories[value] for value in far_original_ids],
        "far_source": (
            "target"
            if far_split_root.resolve() == split_root.resolve()
            else "external"
        ),
        "far_splits": list(far_splits),
        "far_max_samples_per_class": max_far_per_class,
        "far_sampling_seed": far_sampling_seed,
        "role_assignment": role_assignment,
    }
    return (
        training_pool,
        label_idx,
        [unlabeled_idx],
        validation_set,
        test_set,
        dataset_config,
        DatasetScenarioConverter(),
    )
