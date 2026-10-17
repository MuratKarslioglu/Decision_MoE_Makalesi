"""dataset.py ve collator padding testleri (model gerektirmez)."""
import pytest
import torch

from src.data.collator import pad_input_ids
from src.data.dataset import DecisionMoEDataset


def test_pad_input_ids():
    ids, mask = pad_input_ids([torch.tensor([1, 2, 3]), torch.tensor([4])], pad_id=0)
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_dataset_loads_and_filters():
    ds = DecisionMoEDataset("val", question_types=("T4",), limit=5)
    assert 0 < len(ds) <= 5 and all(it["question_type"] == "T4" for it in ds)


def test_invalid_split():
    with pytest.raises(ValueError):
        DecisionMoEDataset("dev")
