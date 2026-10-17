import json
import random
from pathlib import Path

RANDOM_SEED = 42
NUM_IMAGES = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GQA_ROOT = (
    PROJECT_ROOT
    / "dataset_workspace"
    / "downloads"
    / "gqa"
)

SCENE_GRAPHS = GQA_ROOT / "val_sceneGraphs.json"
QUESTIONS = GQA_ROOT / "val_balanced_questions.json"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "decisionmoe_dataset"
    / "metadata"
    / "gqa_selected_ids.json"
)

SPATIAL_KEYWORDS = [
    "left",
    "right",
    "above",
    "below",
    "under",
    "over",
    "behind",
    "front",
    "near",
    "next to",
    "beside",
    "between",
    "inside",
    "outside",
    "on top",
    "in front",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_spatial_question(question_data):
    question_text = (
        question_data.get("question", "")
        .lower()
        .strip()
    )

    # Direct keyword check
    if any(
        keyword in question_text
        for keyword in SPATIAL_KEYWORDS
    ):
        return True

    # GQA semantic/type metadata check
    types = question_data.get("types", {})

    type_text = json.dumps(
        types,
        ensure_ascii=False
    ).lower()

    spatial_terms = [
        "position",
        "spatial",
        "location",
        "relation",
    ]

    return any(
        term in type_text
        for term in spatial_terms
    )


def count_spatial_relations(scene_graph):
    count = 0

    objects = scene_graph.get(
        "objects",
        {}
    )

    for obj in objects.values():
        relations = obj.get(
            "relations",
            []
        )

        for relation in relations:
            name = str(
                relation.get("name", "")
            ).lower()

            if any(
                keyword in name
                for keyword in SPATIAL_KEYWORDS
            ):
                count += 1

    return count


def build_question_scores(questions):
    scores = {}

    for question in questions.values():
        image_id = str(
            question.get("imageId", "")
        )

        if not image_id:
            continue

        if image_id not in scores:
            scores[image_id] = 0

        if is_spatial_question(question):
            scores[image_id] += 1

    return scores


def build_candidates(
    scene_graphs,
    question_scores,
):
    candidates = []

    for image_id, scene_graph in scene_graphs.items():
        image_id = str(image_id)

        spatial_questions = (
            question_scores.get(
                image_id,
                0
            )
        )

        spatial_relations = (
            count_spatial_relations(
                scene_graph
            )
        )

        score = (
            2 * spatial_questions
            + spatial_relations
        )

        candidates.append(
            {
                "image_id": image_id,
                "spatial_question_count": spatial_questions,
                "spatial_relation_count": spatial_relations,
                "score": score,
            }
        )

    return candidates


def select_images(candidates):
    candidates = sorted(
        candidates,
        key=lambda x: (
            -x["score"],
            x["image_id"],
        ),
    )

    # First create a high-quality candidate pool.
    # This keeps selection focused on spatially rich images,
    # while seed=42 still gives deterministic diversity.
    pool_size = min(
        len(candidates),
        max(NUM_IMAGES * 3, NUM_IMAGES),
    )

    high_quality_pool = (
        candidates[:pool_size]
    )

    random.seed(RANDOM_SEED)

    selected = random.sample(
        high_quality_pool,
        NUM_IMAGES,
    )

    selected = sorted(
        selected,
        key=lambda x: x["image_id"],
    )

    return selected, pool_size


def save_metadata(
    selected,
    total_candidates,
    pool_size,
):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "source": "gqa",
        "split": "val",
        "random_seed": RANDOM_SEED,
        "selection_rule": (
            "spatial-rich images ranked by "
            "2 * spatial_question_count + spatial_relation_count"
        ),
        "total_candidates": total_candidates,
        "high_quality_pool_size": pool_size,
        "num_selected": len(selected),
        "images": selected,
    }

    with open(
        OUTPUT_FILE,
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
    print("DecisionMoE - GQA Spatial Image Selection")
    print("=" * 60)

    print("\n[1/5] Loading GQA questions...")
    questions = load_json(
        QUESTIONS
    )

    print(
        f"Questions: "
        f"{len(questions)}"
    )

    print("\n[2/5] Loading scene graphs...")
    scene_graphs = load_json(
        SCENE_GRAPHS
    )

    print(
        f"Scene graphs: "
        f"{len(scene_graphs)}"
    )

    print("\n[3/5] Scoring spatial questions...")
    question_scores = (
        build_question_scores(
            questions
        )
    )

    print("\n[4/5] Building candidate scores...")
    candidates = build_candidates(
        scene_graphs,
        question_scores,
    )

    candidates_with_signal = [
        x
        for x in candidates
        if x["score"] > 0
    ]

    print(
        f"Total candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Candidates with spatial signal: "
        f"{len(candidates_with_signal)}"
    )

    if len(candidates_with_signal) < NUM_IMAGES:
        raise RuntimeError(
            "Not enough spatial candidates."
        )

    print(
        f"\n[5/5] Selecting "
        f"{NUM_IMAGES} images..."
    )

    selected, pool_size = (
        select_images(
            candidates_with_signal
        )
    )

    save_metadata(
        selected,
        len(candidates_with_signal),
        pool_size,
    )

    scores = [
        item["score"]
        for item in selected
    ]

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(
        f"Selected images: "
        f"{len(selected)}"
    )

    print(
        f"Score range: "
        f"{min(scores)} - {max(scores)}"
    )

    print(
        f"Metadata: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
