import json
import time
import requests
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "metadata"
    / "gqa_selected_ids.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "images"
    / "gqa"
)

BASE_URLS = [
    "https://cs.stanford.edu/people/rak248/VG_100K/{id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{id}.jpg",
]

TIMEOUT = 30
MAX_RETRIES = 3


def load_selected_ids():
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return [
        str(item["image_id"])
        for item in data["images"]
    ]


def download_one(session, image_id):
    output_path = OUTPUT_DIR / f"{image_id}.jpg"

    if output_path.exists() and output_path.stat().st_size > 1000:
        return True, "exists"

    for url_template in BASE_URLS:
        url = url_template.format(id=image_id)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = session.get(
                    url,
                    timeout=TIMEOUT,
                    allow_redirects=True,
                )

                if response.status_code == 404:
                    break

                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    ""
                ).lower()

                if (
                    "image" not in content_type
                    and len(response.content) < 1000
                ):
                    raise RuntimeError(
                        f"Unexpected response from {url}"
                    )

                output_path.write_bytes(
                    response.content
                )

                return True, url

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    last_error = str(exc)
                else:
                    time.sleep(attempt * 2)

    return False, last_error


def main():
    print("=" * 60)
    print("DecisionMoE - Download Selected GQA Images")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_ids = load_selected_ids()

    print(f"Selected IDs: {len(selected_ids)}")

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 DecisionMoE-Dataset/1.0"
            )
        }
    )

    success = 0
    failed = []

    for index, image_id in enumerate(
        selected_ids,
        start=1
    ):
        ok, info = download_one(
            session,
            image_id
        )

        if ok:
            success += 1

            print(
                f"[{index:03d}/{len(selected_ids)}] "
                f"OK   {image_id}"
            )

        else:
            failed.append(
                {
                    "image_id": image_id,
                    "error": info,
                }
            )

            print(
                f"[{index:03d}/{len(selected_ids)}] "
                f"FAIL {image_id}: {info}"
            )

        time.sleep(0.05)

    failure_file = (
        PROJECT_ROOT
        / "decisionmoe_dataset"
        / "metadata"
        / "gqa_download_failures.json"
    )

    with open(
        failure_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            failed,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Successful: {success}")
    print(f"Failed:     {len(failed)}")
    print(f"Output:     {OUTPUT_DIR}")

    if failed:
        print(
            f"Failures:   {failure_file}"
        )


if __name__ == "__main__":
    main()
