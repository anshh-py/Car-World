"""Pytest wrapper for the evaluation-dataset schema validation."""

from __future__ import annotations

import pytest

from evaluation.validate_eval_dataset import (
    DATASET_PATH,
    EXPECTED_TOTAL,
    load_records,
    validate,
)


def test_dataset_file_exists():
    assert DATASET_PATH.is_file()


def test_exactly_40_records():
    assert len(load_records()) == EXPECTED_TOTAL


def test_dataset_schema_is_valid():
    errors = validate(load_records())
    assert errors == [], "\n".join(errors)


def test_ids_are_unique():
    records = load_records()
    ids = [record["id"] for record in records]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("record_index", list(range(40)))
def test_every_record_has_required_fields(record_index):
    records = load_records()
    required = {
        "id",
        "category",
        "user_input",
        "objective",
        "expected_behavior",
        "expected_tool",
        "reference_facts",
        "scoring_criteria",
        "severity",
        "provider_applicability",
    }
    assert required <= set(records[record_index].keys())
