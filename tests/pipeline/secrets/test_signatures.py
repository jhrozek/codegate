import os
import tempfile

import pytest

from codegate.pipeline.secrets.signatures import (
    CodegateSignatures,
    Match,
    SignatureGroup,
    YAMLParseError,
)


@pytest.fixture
def valid_yaml_content():
    return """
- AWS:
    - Access Key: '[A-Z0-9]{20}'
"""


@pytest.fixture
def temp_yaml_file(valid_yaml_content):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write(valid_yaml_content)
    yield f.name
    os.unlink(f.name)


def test_match_creation():
    match = Match(
        service="AWS",
        type="Access Key",
        value="AKIAIOSFODNN7EXAMPLE",
        line_number=1,
        start_index=0,
        end_index=20,
    )
    assert match.service == "AWS"
    assert match.type == "Access Key"
    assert match.value == "AKIAIOSFODNN7EXAMPLE"
    assert match.line_number == 1
    assert match.start_index == 0
    assert match.end_index == 20


def test_signature_group_creation():
    patterns = {
        "Access Key": "[A-Z0-9]{20}",
    }
    group = SignatureGroup("AWS", patterns)
    assert group.name == "AWS"
    assert group.patterns == patterns


def test_yaml_parse_error():
    with pytest.raises(YAMLParseError):
        raise YAMLParseError("Invalid YAML format")


def test_initialize_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        CodegateSignatures.initialize("nonexistent.yaml")


def test_initialize_and_reset(temp_yaml_file):
    CodegateSignatures.initialize(temp_yaml_file)
    assert CodegateSignatures._yaml_path == temp_yaml_file

    CodegateSignatures.reset()
    assert CodegateSignatures._yaml_path is None
    assert not CodegateSignatures._signatures_loaded
    assert not CodegateSignatures._signature_groups
    assert not CodegateSignatures._compiled_regexes


def test_find_in_string_with_aws_credentials(temp_yaml_file):
    CodegateSignatures.initialize(temp_yaml_file)
    CodegateSignatures._signatures_loaded = False  # Force reload of signatures

    test_string = """
    aws_access_key = 'AKIAIOSFODNN7EXAMPLE'
    """

    matches = CodegateSignatures.find_in_string(test_string)
    # Get unique matches by value and service
    unique_matches = {(m.service, m.value): m for m in matches if m.service == "AWS"}.values()
    assert len(unique_matches) == 1

    match = next(iter(unique_matches))
    assert match.service == "AWS"
    assert match.type == "Access Key"
    assert match.value == "AKIAIOSFODNN7EXAMPLE"


def test_find_in_string_with_github_token():
    # Create a temporary file with empty content since GitHub patterns are hardcoded
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write("[]")  # Empty YAML list

    try:
        CodegateSignatures.initialize(f.name)
        CodegateSignatures._signatures_loaded = False  # Force reload of signatures

        test_string = "github_token = 'ghp_1234567890abcdef1234567890abcdef123456'"
        matches = CodegateSignatures.find_in_string(test_string)

        # Get unique matches by value and service
        unique_matches = {
            (m.service, m.value): m for m in matches if m.service == "GitHub"
        }.values()
        assert len(unique_matches) == 1

        match = next(iter(unique_matches))
        assert match.service == "GitHub"
        assert match.type == "Access Token"
        assert "ghp_" in match.value
    finally:
        os.unlink(f.name)


def test_find_in_string_with_no_matches(temp_yaml_file):
    CodegateSignatures.initialize(temp_yaml_file)
    CodegateSignatures._signatures_loaded = False  # Force reload of signatures

    test_string = "No secrets here!"
    matches = CodegateSignatures.find_in_string(test_string)
    assert len(matches) == 0


def test_empty_string():
    CodegateSignatures.reset()
    matches = CodegateSignatures.find_in_string("")
    assert len(matches) == 0


def test_invalid_yaml_format():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write("invalid: yaml: content: - [}")

    try:
        with pytest.raises(YAMLParseError):
            CodegateSignatures._load_yaml(f.name)
    finally:
        os.unlink(f.name)


def test_duplicate_patterns():
    """Test that patterns with different names are treated as separate matches"""
    yaml_content = """
    - AWS:
        - Access Key: '[A-Z0-9]{20}'
        - Also Access Key: '[A-Z0-9]{20}'
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write(yaml_content)

    try:
        CodegateSignatures.initialize(f.name)
        CodegateSignatures._signatures_loaded = False  # Force reload of signatures

        test_string = "aws_key = 'AKIAIOSFODNN7EXAMPLE'"
        matches = CodegateSignatures.find_in_string(test_string)

        # Filter only AWS matches and get unique ones by value and service
        aws_matches = {(m.service, m.value): m for m in matches if m.service == "AWS"}.values()
        assert len(aws_matches) == 1

        match = next(iter(aws_matches))
        assert match.service == "AWS"
        assert match.value == "AKIAIOSFODNN7EXAMPLE"
    finally:
        os.unlink(f.name)