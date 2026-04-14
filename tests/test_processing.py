"""Tests for bookdb.processing utility functions."""

import matplotlib
matplotlib.use("Agg")

import polars as pl
import pytest

from bookdb.processing.text import is_non_informative, is_short_low_variety
from bookdb.processing.interactions import (
    calculate_weight_bpr,
    calculate_weight_sar,
    convert_to_unix_timestamp,
    parse_goodreads_timestamps,
)
from bookdb.processing.book_ids import map_similar_books, drop_unmatched_book_ids


# bookdb.processing.text

class TestIsNonInformative:
    def test_empty_string(self):
        assert is_non_informative("") is False

    def test_whitespace_only(self):
        assert is_non_informative("   ") is False

    def test_single_character(self):
        assert is_non_informative("a") is True

    def test_punctuation_only(self):
        assert is_non_informative("!!!") is True

    def test_digits_only(self):
        assert is_non_informative("123") is True

    def test_repeated_characters(self):
        assert is_non_informative("aaaa") is True

    def test_normal_text(self):
        assert is_non_informative("Great book!") is False

    def test_mixed_alphanumeric(self):
        assert is_non_informative("abc123") is False


class TestIsShortLowVariety:
    def test_empty_string(self):
        assert is_short_low_variety("") is False

    def test_whitespace_only(self):
        assert is_short_low_variety("   ") is False

    def test_single_repeated_char(self):
        assert is_short_low_variety("aaaa") is True

    def test_two_unique_chars(self):
        assert is_short_low_variety("abab") is True

    def test_three_unique_chars(self):
        assert is_short_low_variety("abcab") is False

    def test_long_low_variety(self):
        # length > 5, so should be False even with low variety
        assert is_short_low_variety("aaaaaa") is False

    def test_normal_short_text(self):
        assert is_short_low_variety("good") is False


# bookdb.processing.interactions

class TestCalculateWeightBpr:
    def _make_df(self, rating, is_read, is_reviewed):
        return pl.DataFrame({
            "rating": [rating],
            "is_read": [is_read],
            "is_reviewed": [is_reviewed],
        })

    def test_base_interaction_only(self):
        # rating=0, not read, not reviewed → weight = 1.0
        df = self._make_df(0, False, False)
        result = calculate_weight_bpr(df)
        assert result["weight"][0] == pytest.approx(1.0)

    def test_read_adds_two(self):
        # rating=0, is_read=True, not reviewed → weight = 1 + 2 = 3.0
        df = self._make_df(0, True, False)
        result = calculate_weight_bpr(df)
        assert result["weight"][0] == pytest.approx(3.0)

    def test_reviewed_adds_three(self):
        # rating=0, not read, is_reviewed=True → weight = 1 + 3 = 4.0
        df = self._make_df(0, False, True)
        result = calculate_weight_bpr(df)
        assert result["weight"][0] == pytest.approx(4.0)

    def test_rating_contribution(self):
        # rating=5 → rating_contrib = 4; not read, not reviewed → weight = 1 + 4 = 5.0
        df = self._make_df(5, False, False)
        result = calculate_weight_bpr(df)
        assert result["weight"][0] == pytest.approx(5.0)

    def test_full_engagement(self):
        # rating=5, is_read=True, is_reviewed=True → 1 + 2 + 4 + 3 = 10.0
        df = self._make_df(5, True, True)
        result = calculate_weight_bpr(df)
        assert result["weight"][0] == pytest.approx(10.0)

    def test_output_column_is_float32(self):
        df = self._make_df(3, True, False)
        result = calculate_weight_bpr(df)
        assert result["weight"].dtype == pl.Float32

    def test_rating_contrib_column_dropped(self):
        df = self._make_df(3, True, True)
        result = calculate_weight_bpr(df)
        assert "rating_contrib" not in result.columns


class TestCalculateWeightSar:
    def _make_df(self, rating, is_read, review_text):
        return pl.DataFrame({
            "rating": [rating],
            "is_read": [is_read],
            "review_text_incomplete": [review_text],
        })

    def test_no_engagement(self):
        # rating=0, not read, no review → base_weight only = 0.1
        df = self._make_df(0, False, "")
        result = calculate_weight_sar(df)
        assert result["weight"][0] == pytest.approx(0.1, abs=1e-4)

    def test_full_rating(self):
        # rating=5 → rating_norm=1.0, not read, no review → 0.1 + 0.4 = 0.5
        df = self._make_df(5, False, "")
        result = calculate_weight_sar(df)
        assert result["weight"][0] == pytest.approx(0.5, abs=1e-4)

    def test_read_contribution(self):
        # rating=0, is_read=True, no review → 0.1 + 0.2 = 0.3
        df = self._make_df(0, True, "")
        result = calculate_weight_sar(df)
        assert result["weight"][0] == pytest.approx(0.3, abs=1e-4)

    def test_review_contribution(self):
        # rating=0, not read, has review → 0.1 + 0.3 = 0.4
        df = self._make_df(0, False, "Great read!")
        result = calculate_weight_sar(df)
        assert result["weight"][0] == pytest.approx(0.4, abs=1e-4)

    def test_full_engagement(self):
        # rating=5, is_read=True, has review → 0.1 + 0.4 + 0.2 + 0.3 = 1.0
        df = self._make_df(5, True, "Loved it")
        result = calculate_weight_sar(df)
        assert result["weight"][0] == pytest.approx(1.0, abs=1e-4)

    def test_intermediate_columns_dropped(self):
        df = self._make_df(3, True, "ok")
        result = calculate_weight_sar(df)
        for col in ("rating_norm", "read_val", "has_review_val"):
            assert col not in result.columns

    def test_custom_weights(self):
        df = self._make_df(5, True, "review")
        result = calculate_weight_sar(
            df,
            base_weight=0.0,
            rating_weight=1.0,
            is_read_weight=0.0,
            review_weight=0.0,
        )
        assert result["weight"][0] == pytest.approx(1.0, abs=1e-4)


class TestConvertToUnixTimestamp:
    def test_known_date(self):
        df = pl.DataFrame({"date_updated": ["Thu Mar 22 15:43:00 -0700 2012"]})
        result = convert_to_unix_timestamp(df)
        assert "timestamp" in result.columns
        assert result["timestamp"][0] is not None

    def test_original_column_preserved(self):
        df = pl.DataFrame({"date_updated": ["Thu Mar 22 15:43:00 -0700 2012"]})
        result = convert_to_unix_timestamp(df)
        assert "date_updated" in result.columns

    def test_custom_column_name(self):
        df = pl.DataFrame({"read_at": ["Thu Mar 22 15:43:00 -0700 2012"]})
        result = convert_to_unix_timestamp(df, timestamp_col="read_at")
        assert "timestamp" in result.columns

    def test_timestamp_is_int64(self):
        df = pl.DataFrame({"date_updated": ["Thu Mar 22 15:43:00 -0700 2012"]})
        result = convert_to_unix_timestamp(df)
        assert result["timestamp"].dtype == pl.Int64


class TestParseGoodreadsTimestamps:
    SAMPLE_DATE = "Thu Mar 22 15:43:00 +0000 2012"

    def test_single_column(self):
        lf = pl.LazyFrame({"started_at": [self.SAMPLE_DATE]})
        result = parse_goodreads_timestamps(lf, ["started_at"]).collect()
        assert "ts_started_at" in result.columns
        assert "started_at" not in result.columns

    def test_multiple_columns(self):
        lf = pl.LazyFrame({
            "started_at": [self.SAMPLE_DATE],
            "read_at": [self.SAMPLE_DATE],
        })
        result = parse_goodreads_timestamps(lf, ["started_at", "read_at"]).collect()
        assert "ts_started_at" in result.columns
        assert "ts_read_at" in result.columns
        assert "started_at" not in result.columns
        assert "read_at" not in result.columns

    def test_output_dtype_is_int64(self):
        lf = pl.LazyFrame({"started_at": [self.SAMPLE_DATE]})
        result = parse_goodreads_timestamps(lf, ["started_at"]).collect()
        assert result["ts_started_at"].dtype == pl.Int64

    def test_invalid_date_non_strict(self):
        lf = pl.LazyFrame({"started_at": ["not a date"]})
        result = parse_goodreads_timestamps(lf, ["started_at"], strict=False).collect()
        assert result["ts_started_at"][0] is None


# bookdb.processing.book_ids

class TestMapSimilarBooks:
    def test_all_mapped(self):
        lookup = {"10": 1, "20": 2, "30": 3}
        assert map_similar_books(["10", "20", "30"], lookup) == [1, 2, 3]

    def test_unmapped_ids_dropped(self):
        lookup = {"10": 1, "30": 3}
        assert map_similar_books(["10", "20", "30"], lookup) == [1, 3]

    def test_empty_list(self):
        assert map_similar_books([], {"10": 1}) == []

    def test_empty_lookup(self):
        assert map_similar_books(["10", "20"], {}) == []

    def test_preserves_order(self):
        lookup = {"a": 3, "b": 1, "c": 2}
        assert map_similar_books(["c", "a", "b"], lookup) == [2, 3, 1]


class TestDropUnmatchedBookIds:
    def test_filters_unmatched_rows(self):
        lf = pl.LazyFrame({"book_id": [1, 2, 3, 4]})
        books_ref = pl.DataFrame({"book_id": [2, 4]})
        result = drop_unmatched_book_ids(lf, books_ref).collect()
        assert sorted(result["book_id"].to_list()) == [2, 4]

    def test_all_matched(self):
        lf = pl.LazyFrame({"book_id": [1, 2, 3]})
        books_ref = pl.DataFrame({"book_id": [1, 2, 3]})
        result = drop_unmatched_book_ids(lf, books_ref).collect()
        assert len(result) == 3

    def test_none_matched(self):
        lf = pl.LazyFrame({"book_id": [5, 6]})
        books_ref = pl.DataFrame({"book_id": [1, 2]})
        result = drop_unmatched_book_ids(lf, books_ref).collect()
        assert len(result) == 0

    def test_custom_book_id_col(self):
        lf = pl.LazyFrame({"item_id": [1, 2, 3]})
        books_ref = pl.DataFrame({"item_id": [1, 3]})
        result = drop_unmatched_book_ids(lf, books_ref, book_id_col="item_id").collect()
        assert sorted(result["item_id"].to_list()) == [1, 3]
