import json
import random
import shutil
from pathlib import Path

RANDOM_SEED = 42
NUM_IMAGES = 400

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AI2D_ROOT = (
    PROJECT_ROOT
    / "dataset_workspace"
    / "downloads"
    / "ai2d"
    / "ai2d"
)

IMAGE_DIR = AI2D_ROOT / "images"
ANNOTATION_DIR = AI2D_ROOT / "annotations"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "images"
    / "ai2d"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "metadata"
)

METADATA_FILE = METADATA_DIR / "ai2d_selected_ids.json"

GRAPHIC_FIELDS = [
    "blobs",
    "arrows",
    "arrowHeads",
    "containers",
    "meaningfulSpaces",
]


def count_items(value):
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_valid_images():
    valid = []

    annotation_files = sorted(
        ANNOTATION_DIR.glob("*.json")
    )

    print(f"Annotation files: {len(annotation_files)}")

    for ann_path in annotation_files:
        image_name = ann_path.stem

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            continue

        annotation = load_json(ann_path)

        text_count = count_items(
            annotation.get("text", {})
        )

        graphic_counts = {
            field: count_items(annotation.get(field, {}))
            for field in GRAPHIC_FIELDS
        }

        graphic_count = sum(graphic_counts.values())

        if text_count > 0 and graphic_count > 0:
            valid.append(
                {
                    "image_name": image_name,
                    "annotation_file": ann_path.name,
                    "text_count": text_count,
                    "graphic_count": graphic_count,
                    "graphic_counts": graphic_counts,
                }
            )

    return valid


def select_images(valid_images):
    if len(valid_images) < NUM_IMAGES:
        raise RuntimeError(
            f"Not enough valid images. "
            f"Found {len(valid_images)}, need {NUM_IMAGES}."
        )

    valid_images = sorted(
        valid_images,
        key=lambda x: x["image_name"]
    )

    random.seed(RANDOM_SEED)

    return random.sample(
        valid_images,
        NUM_IMAGES
    )


def copy_selected(selected):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Clean old files so reruns stay deterministic
    for old_file in OUTPUT_DIR.iterdir():
        if old_file.is_file():
            old_file.unlink()

    for item in selected:
        src = IMAGE_DIR / item["image_name"]
        dst = OUTPUT_DIR / item["image_name"]

        shutil.copy2(src, dst)


def save_metadata(selected, valid_count):
    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "source": "ai2d",
        "random_seed": RANDOM_SEED,
        "selection_rule": (
            "at least one text element and "
            "at least one graphical element"
        ),
        "graphic_fields": GRAPHIC_FIELDS,
        "num_valid_candidates": valid_count,
        "num_selected": len(selected),
        "images": selected,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )


def main():
    print("=" * 60)
    print("DecisionMoE - AI2D Selection")
    print("=" * 60)

    print("\n[1/4] Finding valid AI2D images...")
    valid = get_valid_images()

    print(f"Valid candidates: {len(valid)}")

    print(f"\n[2/4] Selecting {NUM_IMAGES} images...")
    selected = select_images(valid)

    print("\n[3/4] Copying selected images...")
    copy_selected(selected)

    print("\n[4/4] Saving metadata...")
    save_metadata(
        selected,
        len(valid)
    )

    final_count = len(
        [
            p for p in OUTPUT_DIR.iterdir()
            if p.is_file()
        ]
    )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"Selected: {len(selected)}")
    print(f"Final image count: {final_count}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
