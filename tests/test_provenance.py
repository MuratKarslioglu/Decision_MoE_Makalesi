"""provenance.collect_meta testleri."""
from src.utils.provenance import collect_meta, file_hash


def test_meta_has_required_keys():
    m = collect_meta(seed=7, extra={"quantization": "nf4"})
    for k in ("timestamp", "git_commit", "python", "packages", "config_hashes", "seed", "gpu", "quantization"):
        assert k in m
    assert m["seed"] == 7 and m["quantization"] == "nf4"
    assert "model.yaml" in m["config_hashes"] and "ppo.yaml" in m["config_hashes"]


def test_file_hash_stable_and_missing(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("x")
    assert file_hash(f) == file_hash(f) and len(file_hash(f)) == 12
    assert file_hash(tmp_path / "yok") == "missing"
