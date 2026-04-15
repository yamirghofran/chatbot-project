"""Data quality checks for book dataset.

Validates schema, completeness, and data integrity of book data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from bookdb.validation import ValidationResult


@dataclass
class DataQualityReport:
    """Complete data quality report."""

    total_books: int
    valid_books: int
    issues: list[ValidationResult] = field(default_factory=list)
    schema_valid: bool = True
    completeness_score: float = 0.0

    @property
    def quality_score(self) -> float:
        """Overall quality score 0-100."""
        if self.total_books == 0:
            return 0.0

        base_score = (self.valid_books / self.total_books) * 100

        # Deduct for errors
        errors = len([i for i in self.issues if i.severity == "error"])
        warnings = len([i for i in self.issues if i.severity == "warning"])

        deduction = (errors * 10) + (warnings * 2)
        return max(0.0, base_score - deduction)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_books": self.total_books,
            "valid_books": self.valid_books,
            "schema_valid": self.schema_valid,
            "completeness_score": round(self.completeness_score, 2),
            "quality_score": round(self.quality_score, 2),
            "issues": [
                {"valid": i.is_valid, "message": i.message, "severity": i.severity}
                for i in self.issues
            ],
        }


class BookDataValidator:
    """Validates book dataset quality."""

    REQUIRED_COLUMNS = {
        "book_id",
        "title",
        "authors",
        "description",
        "similar_books",
    }

    OPTIONAL_COLUMNS = {
        "publisher",
        "publication_year",
        "num_pages",
        "isbn13",
        "language_code",
        "popular_shelves",
        "num_interactions",
    }

    @classmethod
    def validate_schema(cls, df: pl.DataFrame) -> list[ValidationResult]:
        """Validate dataframe schema."""
        results = []

        columns = set(df.columns)

        # Check required columns
        missing_required = cls.REQUIRED_COLUMNS - columns
        if missing_required:
            results.append(
                ValidationResult(
                    False, f"Missing required columns: {missing_required}", "error"
                )
            )

        # Check for unexpected columns (warnings)
        all_expected = cls.REQUIRED_COLUMNS | cls.OPTIONAL_COLUMNS
        unexpected = columns - all_expected
        if unexpected:
            results.append(
                ValidationResult(True, f"Unexpected columns: {unexpected}", "warning")
            )

        return results

    @classmethod
    def validate_data_types(cls, df: pl.DataFrame) -> list[ValidationResult]:
        """Validate column data types."""
        results = []

        # Check book_id is numeric or string
        if "book_id" in df.columns:
            book_id_type = str(df.schema["book_id"])
            if "Int" not in book_id_type and "String" not in book_id_type:
                results.append(
                    ValidationResult(
                        False,
                        f"book_id should be numeric or string, got {book_id_type}",
                        "error",
                    )
                )

        # Check title is string
        if "title" in df.columns:
            title_type = str(df.schema["title"])
            if "String" not in title_type:
                results.append(
                    ValidationResult(
                        False, f"title should be string, got {title_type}", "error"
                    )
                )

        # Check similar_books is list
        if "similar_books" in df.columns:
            similar_type = str(df.schema.get("similar_books", ""))
            if "List" not in similar_type and "Array" not in similar_type:
                results.append(
                    ValidationResult(
                        False,
                        f"similar_books should be a list, got {similar_type}",
                        "error",
                    )
                )

        return results

    @classmethod
    def validate_completeness(
        cls, df: pl.DataFrame
    ) -> tuple[float, list[ValidationResult]]:
        """Validate data completeness.

        Returns completeness score (0-1) and list of issues.
        """
        results = []
        total_rows = df.height

        if total_rows == 0:
            return 0.0, [ValidationResult(False, "Dataset is empty", "error")]

        # Check required fields have values
        completeness_scores = []

        for col in cls.REQUIRED_COLUMNS:
            if col not in df.columns:
                continue

            # Count non-null values
            if df.schema[col] == pl.List:
                # For list columns, check non-empty lists
                non_null = df.filter(
                    pl.col(col).is_not_null() & (pl.col(col).list.len() > 0)
                ).height
            else:
                non_null = df.filter(pl.col(col).is_not_null()).height

            completeness = non_null / total_rows
            completeness_scores.append(completeness)

            if completeness < 0.5:
                results.append(
                    ValidationResult(
                        False,
                        f"Column '{col}' only {completeness:.1%} complete",
                        "error",
                    )
                )
            elif completeness < 0.8:
                results.append(
                    ValidationResult(
                        True,
                        f"Column '{col}' only {completeness:.1%} complete",
                        "warning",
                    )
                )

        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 0.0
        )
        return avg_completeness, results

    @classmethod
    def validate_integrity(cls, df: pl.DataFrame) -> list[ValidationResult]:
        """Validate data integrity constraints."""
        results = []

        # Check for duplicate book_ids
        if "book_id" in df.columns:
            duplicates = (
                df.group_by("book_id")
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") > 1)
            )

            if duplicates.height > 0:
                results.append(
                    ValidationResult(
                        False, f"Found {duplicates.height} duplicate book_ids", "error"
                    )
                )

        # Check similar_books reference valid book_ids
        if "similar_books" in df.columns and "book_id" in df.columns:
            all_book_ids = set(df["book_id"].to_list())

            # Sample a few books to check
            sample = df.sample(min(1000, df.height))
            invalid_refs = 0

            for book in sample.iter_rows(named=True):
                similar = book.get("similar_books", [])
                if similar:
                    for ref_id in similar[:10]:  # Check first 10
                        if ref_id not in all_book_ids:
                            invalid_refs += 1

            if invalid_refs > 0:
                # This is expected (some similar_books may be from different datasets)
                results.append(
                    ValidationResult(
                        True,
                        f"{invalid_refs} similar_books references don't exist in dataset (may be external)",
                        "warning",
                    )
                )

        # Check for suspicious values
        if "publication_year" in df.columns:
            future_books = df.filter(
                pl.col("publication_year").is_not_null()
                & (pl.col("publication_year") > 2025)
            ).height

            if future_books > 0:
                results.append(
                    ValidationResult(
                        True,
                        f"{future_books} books have future publication years",
                        "warning",
                    )
                )

            ancient_books = df.filter(
                pl.col("publication_year").is_not_null()
                & (pl.col("publication_year") < 1000)
            ).height

            if ancient_books > 0:
                results.append(
                    ValidationResult(
                        True,
                        f"{ancient_books} books have suspiciously old publication years",
                        "warning",
                    )
                )

        return results

    @classmethod
    def validate_dataset(cls, df: pl.DataFrame) -> DataQualityReport:
        """Run full validation on dataset."""
        all_issues = []

        # Schema validation
        schema_issues = cls.validate_schema(df)
        all_issues.extend(schema_issues)

        # Data type validation
        type_issues = cls.validate_data_types(df)
        all_issues.extend(type_issues)

        # Completeness
        completeness, completeness_issues = cls.validate_completeness(df)
        all_issues.extend(completeness_issues)

        # Integrity
        integrity_issues = cls.validate_integrity(df)
        all_issues.extend(integrity_issues)

        # Count valid books (books without errors)
        errors = [i for i in all_issues if not i.is_valid and i.severity == "error"]
        schema_valid = len([e for e in errors if "column" in e.message.lower()]) == 0

        return DataQualityReport(
            total_books=df.height,
            valid_books=max(0, df.height - len(errors)),
            issues=all_issues,
            schema_valid=schema_valid,
            completeness_score=completeness,
        )

    @classmethod
    def validate_from_parquet(cls, path: Path | str) -> DataQualityReport:
        """Validate dataset from parquet file."""
        path = Path(path)

        if not path.exists():
            return DataQualityReport(
                total_books=0,
                valid_books=0,
                issues=[ValidationResult(False, f"File not found: {path}", "error")],
                schema_valid=False,
            )

        try:
            df = pl.read_parquet(path)
            return cls.validate_dataset(df)
        except Exception as e:
            return DataQualityReport(
                total_books=0,
                valid_books=0,
                issues=[
                    ValidationResult(
                        False, f"Failed to read parquet: {str(e)}", "error"
                    )
                ],
                schema_valid=False,
            )


def check_data_quality(books_path: Path | str = None) -> DataQualityReport:
    """Quick function to check book data quality.

    Usage:
        report = check_data_quality("data/books.parquet")
        print(f"Quality score: {report.quality_score}/100")
    """
    if books_path is None:
        books_path = Path("data/3_goodreads_books_with_metrics.parquet")

    return BookDataValidator.validate_from_parquet(books_path)


if __name__ == "__main__":
    # Run validation
    print("Running data quality check...")
    report = check_data_quality()

    print(f"\nTotal books: {report.total_books}")
    print(f"Valid books: {report.valid_books}")
    print(f"Schema valid: {report.schema_valid}")
    print(f"Completeness: {report.completeness_score:.1%}")
    print(f"Quality score: {report.quality_score:.1f}/100")

    if report.issues:
        print("\nIssues found:")
        for issue in report.issues:
            icon = (
                "❌"
                if issue.severity == "error"
                else "⚠️"
                if issue.severity == "warning"
                else "ℹ️"
            )
            print(f"  {icon} {issue.message}")
    else:
        print("\n✅ No issues found!")
