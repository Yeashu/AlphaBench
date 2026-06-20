"""Tests for alphabench.prompts."""

import hashlib

import pytest
from alphabench.prompts import PROMPT_REGISTRY, format_prompt, get_prompt


def test_get_prompt_returns_text_and_version():
    text, pv = get_prompt("v1.0.0")
    assert isinstance(text, str)
    assert len(text) > 100
    assert pv.version == "v1.0.0"
    assert pv.prompt_id == "system_prompt"


def test_sha256_matches_text():
    text, pv = get_prompt("v1.0.0")
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert pv.sha256 == expected


def test_unknown_version_raises():
    with pytest.raises(KeyError, match="v99.0.0"):
        get_prompt("v99.0.0")


def test_format_prompt_fills_asset_id():
    text, pv = format_prompt("v1.0.0", asset_id="BTC-USDT")
    assert "BTC-USDT" in text


def test_different_prompts_different_hashes():
    """Any change to prompt text must change the SHA-256."""
    text, pv = get_prompt("v1.0.0")
    modified = text + " "
    modified_sha = hashlib.sha256(modified.encode()).hexdigest()
    assert modified_sha != pv.sha256


def test_all_registry_versions_parseable():
    for version in PROMPT_REGISTRY:
        text, pv = get_prompt(version)
        assert len(text) > 0
        assert pv.sha256
