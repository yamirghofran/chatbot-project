"""Unit tests for the private validation helpers in bookdb.db.crud.

These are pure-Python tests — no database session needed.
"""

import math
import pytest

from bookdb.db.crud import (
    _parse_optional_bigint,
    _require_non_empty,
    _validate_email,
    _validate_year,
)


class TestParseOptionalBigint:
    def test_none_returns_none(self):
        assert _parse_optional_bigint(None) is None

    def test_int_passthrough(self):
        assert _parse_optional_bigint(42) == 42
        assert _parse_optional_bigint(0) == 0

    def test_float_converts(self):
        assert _parse_optional_bigint(42.0) == 42
        assert _parse_optional_bigint(99.9) == 99  # truncates

    def test_float_nan_returns_none(self):
        assert _parse_optional_bigint(float("nan")) is None

    def test_float_inf_returns_none(self):
        assert _parse_optional_bigint(float("inf")) is None
        assert _parse_optional_bigint(float("-inf")) is None

    def test_string_int(self):
        assert _parse_optional_bigint("123") == 123
        assert _parse_optional_bigint("  456  ") == 456

    def test_empty_string_returns_none(self):
        assert _parse_optional_bigint("") is None
        assert _parse_optional_bigint("   ") is None

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _parse_optional_bigint("not_a_number")


class TestRequireNonEmpty:
    def test_valid_string(self):
        assert _require_non_empty("hello", "field") == "hello"

    def test_strips_whitespace(self):
        assert _require_non_empty("  hello  ", "field") == "hello"

    def test_none_raises(self):
        with pytest.raises(ValueError, match="field is required"):
            _require_non_empty(None, "field")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _require_non_empty("", "field")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _require_non_empty("   ", "field")


class TestValidateEmail:
    def test_valid_email(self):
        assert _validate_email("user@example.com") == "user@example.com"
        assert _validate_email("  alice@test.org  ") == "alice@test.org"

    def test_missing_at_raises(self):
        with pytest.raises(ValueError, match="Invalid email"):
            _validate_email("userexample.com")

    def test_missing_domain_raises(self):
        with pytest.raises(ValueError, match="Invalid email"):
            _validate_email("user@")

    def test_missing_tld_raises(self):
        with pytest.raises(ValueError, match="Invalid email"):
            _validate_email("user@example")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="email"):
            _validate_email("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="email is required"):
            _validate_email(None)


class TestValidateYear:
    def test_none_passthrough(self):
        assert _validate_year(None) is None

    def test_valid_year(self):
        assert _validate_year(1984) == 1984
        assert _validate_year(1000) == 1000
        assert _validate_year(9999) == 9999

    def test_too_low_raises(self):
        with pytest.raises(ValueError, match="between 1000 and 9999"):
            _validate_year(999)

    def test_too_high_raises(self):
        with pytest.raises(ValueError, match="between 1000 and 9999"):
            _validate_year(10000)

    def test_non_int_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_year("2024")

    def test_float_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_year(2024.0)
