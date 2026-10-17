import json
import random
import shutil
from pathlib import Path


RANDOM_SEED = 42
NUM_IMAGES = 800
MIN_UNIQUE_CATEGORIES = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COCO_ROOT = (
    PROJECT_ROOT
    / "dataset_workspace"
    / "downloads"
    / "coco"
)

IMAGE_DIR = COCO_ROOT / "val2017"

ANNOTATION_FILE = (
    COCO_ROOT
    / "annotations"
    / "instances_val2017.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "images"
    / "coco"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "metadata"
)

SELECTED_IDS_FILE = (
    METADATA_DIR
    / "coco_selected_ids.json"
)


def load_annotations():
    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_image_category_map(data):
    image_categories = {}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if image_id not in image_categories:
            image_categories[image_id] = set()

        image_categories[image_id].add(
            category_id
        )

    return image_categories


def get_valid_images(
    data,
    image_categories,
):
    valid_images = []

    for image in data["images"]:
        image_id = image["id"]

        categories = image_categories.get(
            image_id,
            set(),
        )

        if len(categories) >= MIN_UNIQUE_CATEGORIES:
            valid_images.append(
                {
                    "id": image_id,
                    "file_name": image["file_name"],
                    "unique_category_count": len(
                        categories
                    ),
                }
            )

    return valid_images


def select_images(valid_images):
    if len(valid_images) < NUM_IMAGES:
        raise RuntimeError(
            f"Only {len(valid_images)} valid images found."
        )

    random.seed(RANDOM_SEED)

    valid_images = sorted(
        valid_images,
        key=lambda x: x["id"],
    )

    return random.sample(
        valid_images,
        NUM_IMAGES,
    )


def copy_images(selected_images):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0

    for item in selected_images:
        src = IMAGE_DIR / item["file_name"]
        dst = OUTPUT_DIR / item["file_name"]

        if not src.exists():
            raise FileNotFoundError(
                f"Missing image: {src}"
            )

        shutil.copy2(
            src,
            dst,
        )

        copied += 1

    return copied


def save_metadata(selected_images):
    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "source": "coco",
        "split": "val2017",
        "random_seed": RANDOM_SEED,
        "selection_rule": (
            f"at least {MIN_UNIQUE_CATEGORIES} "
            f"unique COCO categories"
        ),
        "num_selected": len(
            selected_images
        ),
        "images": selected_images,
    }

    with open(
        SELECTED_IDS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )


def validate_output():
    files = list(
        OUTPUT_DIR.glob("*.jpg")
    )

    if len(files) != NUM_IMAGES:
        raise RuntimeError(
            f"Expected {NUM_IMAGES} images, "
            f"found {len(files)}"
        )

    return len(files)


def main():
    print("=" * 60)
    print("DecisionMoE - COCO Image Selection")
    print("=" * 60)

    print("\n[1/5] Loading annotations...")
    data = load_annotations()

    print(
        f"Images in annotations: "
        f"{len(data['images'])}"
    )

    print(
        f"Object annotations: "
        f"{len(data['annotations'])}"
    )

    print("\n[2/5] Building category mapping...")
    image_categories = (
        build_image_category_map(data)
    )

    print("\n[3/5] Filtering images...")
    valid_images = get_valid_images(
        data,
        image_categories,
    )

    print(
        f"Valid images with >= "
        f"{MIN_UNIQUE_CATEGORIES} categories: "
        f"{len(valid_images)}"
    )

    print(
        f"\n[4/5] Selecting "
        f"{NUM_IMAGES} images "
        f"with seed={RANDOM_SEED}..."
    )

    selected_images = select_images(
        valid_images
    )

    print("\n[5/5] Copying images...")
    copied = copy_images(
        selected_images
    )

    save_metadata(
        selected_images
    )

    final_count = validate_output()

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(
        f"Copied images: "
        f"{copied}"
    )

    print(
        f"Final image count: "
        f"{final_count}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nMetadata:\n"
        f"{SELECTED_IDS_FILE}"
    )


if __name__ == "__main__":
    main()
