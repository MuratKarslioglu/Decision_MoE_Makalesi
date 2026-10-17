import json
import random
import time
from pathlib import Path

import requests


RANDOM_SEED = 42
NUM_IMAGES = 300

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXTVQA_JSON = (
    PROJECT_ROOT
    / "dataset_workspace"
    / "downloads"
    / "textvqa"
    / "TextVQA_0.5.1_val.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "images"
    / "textvqa"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "metadata"
    / "textvqa_selected_ids.json"
)

FAILURE_FILE = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "metadata"
    / "textvqa_download_failures.json"
)

TIMEOUT = 30
MAX_RETRIES = 3


def load_dataset():
    with open(
        TEXTVQA_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_unique_image_pool(data):
    image_map = {}

    for item in data["data"]:
        image_id = str(item["image_id"])

        if image_id in image_map:
            continue

        urls = []

        original_url = item.get(
            "flickr_original_url"
        )

        small_url = item.get(
            "flickr_300k_url"
        )

        if original_url:
            urls.append(original_url)

        if (
            small_url
            and small_url not in urls
        ):
            urls.append(small_url)

        if not urls:
            continue

        image_map[image_id] = {
            "image_id": image_id,
            "urls": urls,
            "width": item.get(
                "image_width"
            ),
            "height": item.get(
                "image_height"
            ),
        }

    return list(
        image_map.values()
    )


def download_image(
    session,
    candidate,
):
    image_id = candidate["image_id"]

    output_path = (
        OUTPUT_DIR
        / f"{image_id}.jpg"
    )

    if (
        output_path.exists()
        and output_path.stat().st_size > 1000
    ):
        return True, "exists"

    last_error = "unknown error"

    for url in candidate["urls"]:
        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):
            try:
                response = session.get(
                    url,
                    timeout=TIMEOUT,
                    allow_redirects=True,
                )

                if response.status_code == 404:
                    last_error = (
                        f"404: {url}"
                    )
                    break

                response.raise_for_status()

                content = response.content

                if len(content) < 1000:
                    raise RuntimeError(
                        "Downloaded file too small"
                    )

                content_type = (
                    response.headers
                    .get(
                        "content-type",
                        ""
                    )
                    .lower()
                )

                if (
                    "image" not in content_type
                    and len(content) < 5000
                ):
                    raise RuntimeError(
                        f"Unexpected content type: "
                        f"{content_type}"
                    )

                output_path.write_bytes(
                    content
                )

                return True, url

            except Exception as exc:
                last_error = str(exc)

                if attempt < MAX_RETRIES:
                    time.sleep(
                        attempt * 2
                    )

    return False, last_error


def save_metadata(
    selected,
    pool_size,
):
    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "source": "textvqa",
        "split": "val",
        "random_seed": RANDOM_SEED,
        "selection_rule": (
            "unique TextVQA validation images "
            "downloaded from Flickr URLs"
        ),
        "unique_candidate_images": (
            pool_size
        ),
        "num_selected": len(
            selected
        ),
        "images": selected,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    print("=" * 60)
    print(
        "DecisionMoE - TextVQA "
        "Image Selection & Download"
    )
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n[1/4] Loading TextVQA..."
    )

    data = load_dataset()

    print(
        f"Questions: "
        f"{len(data['data'])}"
    )

    print(
        "\n[2/4] Building unique "
        "image pool..."
    )

    candidates = (
        build_unique_image_pool(
            data
        )
    )

    print(
        f"Unique candidate images: "
        f"{len(candidates)}"
    )

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        candidates
    )

    print(
        f"\n[3/4] Downloading "
        f"{NUM_IMAGES} images..."
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "DecisionMoE-Dataset/1.0"
            )
        }
    )

    selected = []
    failures = []

    for candidate in candidates:
        if len(selected) >= NUM_IMAGES:
            break

        image_id = (
            candidate["image_id"]
        )

        ok, info = download_image(
            session,
            candidate,
        )

        if ok:
            selected.append(
                {
                    "image_id": image_id,
                    "file_name": (
                        f"{image_id}.jpg"
                    ),
                    "source_url": info,
                    "width": (
                        candidate["width"]
                    ),
                    "height": (
                        candidate["height"]
                    ),
                }
            )

            print(
                f"[{len(selected):03d}/"
                f"{NUM_IMAGES}] "
                f"OK {image_id}"
            )

        else:
            failures.append(
                {
                    "image_id": image_id,
                    "error": info,
                }
            )

            print(
                f"FAIL {image_id}: "
                f"{info}"
            )

        time.sleep(0.05)

    print(
        "\n[4/4] Saving metadata..."
    )

    save_metadata(
        selected,
        len(candidates),
    )

    with open(
        FAILURE_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            failures,
            f,
            indent=2,
            ensure_ascii=False,
        )

    final_count = len(
        list(
            OUTPUT_DIR.glob(
                "*.jpg"
            )
        )
    )

    print(
        "\n" + "=" * 60
    )

    if len(selected) == NUM_IMAGES:
        print("SUCCESS")
    else:
        print("INCOMPLETE")

    print("=" * 60)

    print(
        f"Selected: "
        f"{len(selected)}"
    )

    print(
        f"Failed attempts: "
        f"{len(failures)}"
    )

    print(
        f"Final image count: "
        f"{final_count}"
    )

    print(
        f"Metadata: "
        f"{METADATA_FILE}"
    )

    print(
        f"Failures: "
        f"{FAILURE_FILE}"
    )


if __name__ == "__main__":
    main()
