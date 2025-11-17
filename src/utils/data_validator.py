"""
Data Validation and Quality Checks Module

Provides comprehensive data validation before processing to prevent runtime errors
and ensure data quality throughout the pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

    def __str__(self):
        status = "PASSED" if self.is_valid else "FAILED"
        return f"Validation {status}: {len(self.errors)} errors, {len(self.warnings)} warnings"


class DataValidator:
    """
    Comprehensive data validation for customer segmentation platform.

    Validates:
    - Schema compliance
    - Data types
    - Missing values
    - Data ranges
    - Business logic
    - Data quality metrics
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize data validator.

        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
        self.metrics = {}

    def validate_transactions(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate transaction data.

        Args:
            df: Transaction DataFrame

        Returns:
            ValidationResult with validation status
        """
        self.errors = []
        self.warnings = []
        self.metrics = {}

        logger.info(f"Validating transaction data: {len(df)} rows")

        # Required columns
        required_columns = ["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]
        self._check_required_columns(df, required_columns, "transactions")

        # Data types
        self._validate_data_types(
            df, {"Quantity": (int, float), "UnitPrice": (int, float)}
        )

        # Check for missing values
        self._check_missing_values(
            df, critical_columns=["CustomerID", "Quantity", "UnitPrice"]
        )

        # Business logic validation
        self._check_positive_values(df, "Quantity", allow_zero=False)
        self._check_positive_values(df, "UnitPrice", allow_zero=True)

        # Date validation
        if "InvoiceDate" in df.columns:
            self._validate_dates(df, "InvoiceDate")

        # Check for duplicates
        self._check_duplicates(df, subset=["CustomerID", "InvoiceDate"])

        # Data quality metrics
        self.metrics["total_rows"] = len(df)
        self.metrics["unique_customers"] = (
            df["CustomerID"].nunique() if "CustomerID" in df.columns else 0
        )
        self.metrics["date_range"] = (
            self._get_date_range(df, "InvoiceDate")
            if "InvoiceDate" in df.columns
            else None
        )
        self.metrics["total_revenue"] = (
            (df["Quantity"] * df["UnitPrice"]).sum()
            if all(col in df.columns for col in ["Quantity", "UnitPrice"])
            else 0
        )

        # Outlier detection
        if "Quantity" in df.columns:
            self._detect_outliers(df, "Quantity", "transactions")

        is_valid = len(self.errors) == 0 and (
            not self.strict_mode or len(self.warnings) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
            metrics=self.metrics.copy(),
        )

    def validate_customers(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate customer data.

        Args:
            df: Customer DataFrame

        Returns:
            ValidationResult with validation status
        """
        self.errors = []
        self.warnings = []
        self.metrics = {}

        logger.info(f"Validating customer data: {len(df)} rows")

        # Required columns
        required_columns = ["CustomerID"]
        self._check_required_columns(df, required_columns, "customers")

        # Check for duplicates
        if "CustomerID" in df.columns:
            duplicates = df["CustomerID"].duplicated().sum()
            if duplicates > 0:
                self.errors.append(f"Found {duplicates} duplicate CustomerIDs")

        # Check for missing values
        self._check_missing_values(df, critical_columns=["CustomerID"])

        # Data quality metrics
        self.metrics["total_customers"] = len(df)
        self.metrics["unique_customers"] = (
            df["CustomerID"].nunique() if "CustomerID" in df.columns else 0
        )

        is_valid = len(self.errors) == 0 and (
            not self.strict_mode or len(self.warnings) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
            metrics=self.metrics.copy(),
        )

    def validate_products(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate product data.

        Args:
            df: Product DataFrame

        Returns:
            ValidationResult with validation status
        """
        self.errors = []
        self.warnings = []
        self.metrics = {}

        logger.info(f"Validating product data: {len(df)} rows")

        # Check for missing values
        self._check_missing_values(
            df,
            critical_columns=["StockCode", "Description"]
            if "StockCode" in df.columns
            else [],
        )

        # Data quality metrics
        self.metrics["total_products"] = len(df)
        self.metrics["unique_products"] = (
            df["StockCode"].nunique() if "StockCode" in df.columns else 0
        )

        is_valid = len(self.errors) == 0 and (
            not self.strict_mode or len(self.warnings) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
            metrics=self.metrics.copy(),
        )

    def _check_required_columns(
        self, df: pd.DataFrame, required: List[str], dataset_name: str
    ):
        """Check if required columns exist"""
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.errors.append(f"Missing required columns in {dataset_name}: {missing}")

    def _validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, tuple]):
        """Validate data types of columns"""
        for col, expected in expected_types.items():
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col], errors="raise")
                    except (ValueError, TypeError):
                        self.errors.append(
                            f"Column '{col}' has invalid data type, expected numeric"
                        )

    def _check_missing_values(self, df: pd.DataFrame, critical_columns: List[str]):
        """Check for missing values in critical columns"""
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100

                if missing_count > 0:
                    if missing_pct > 5:
                        self.errors.append(
                            f"Column '{col}' has {missing_pct:.1f}% missing values ({missing_count} rows)"
                        )
                    else:
                        self.warnings.append(
                            f"Column '{col}' has {missing_count} missing values ({missing_pct:.2f}%)"
                        )

    def _check_positive_values(
        self, df: pd.DataFrame, column: str, allow_zero: bool = False
    ):
        """Check if numeric column has positive values"""
        if column in df.columns:
            if allow_zero:
                invalid = df[column] < 0
            else:
                invalid = df[column] <= 0

            invalid_count = invalid.sum()
            if invalid_count > 0:
                self.errors.append(
                    f"Column '{column}' has {invalid_count} non-positive values"
                )

    def _validate_dates(self, df: pd.DataFrame, column: str):
        """Validate date column"""
        if column in df.columns:
            try:
                dates = pd.to_datetime(df[column], errors="coerce")
                invalid_dates = dates.isna().sum() - df[column].isna().sum()

                if invalid_dates > 0:
                    self.errors.append(
                        f"Column '{column}' has {invalid_dates} invalid dates"
                    )

                # Check for future dates
                if not dates.isna().all():
                    max_date = dates.max()
                    if max_date > pd.Timestamp.now():
                        self.warnings.append(
                            f"Column '{column}' contains future dates (max: {max_date})"
                        )
            except Exception as e:
                self.errors.append(
                    f"Failed to parse dates in column '{column}': {str(e)}"
                )

    def _check_duplicates(self, df: pd.DataFrame, subset: List[str]):
        """Check for duplicate rows"""
        valid_subset = [col for col in subset if col in df.columns]
        if valid_subset:
            duplicates = df.duplicated(subset=valid_subset).sum()
            if duplicates > 0:
                dup_pct = (duplicates / len(df)) * 100
                if dup_pct > 1:
                    self.warnings.append(
                        f"Found {duplicates} duplicate rows ({dup_pct:.1f}%)"
                    )

    def _detect_outliers(self, df: pd.DataFrame, column: str, dataset_name: str):
        """Detect outliers using IQR method"""
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            if outliers > 0:
                outlier_pct = (outliers / len(df)) * 100
                self.warnings.append(
                    f"Column '{column}' in {dataset_name} has {outliers} outliers ({outlier_pct:.1f}%)"
                )

    def _get_date_range(
        self, df: pd.DataFrame, column: str
    ) -> Optional[Dict[str, str]]:
        """Get date range from column"""
        if column in df.columns:
            try:
                dates = pd.to_datetime(df[column], errors="coerce")
                return {
                    "min": str(dates.min()),
                    "max": str(dates.max()),
                    "span_days": (dates.max() - dates.min()).days,
                }
            except:
                return None
        return None


def validate_data_files(data_dir: str = "data") -> Dict[str, ValidationResult]:
    """
    Validate all data files in the data directory.

    Args:
        data_dir: Directory containing data files

    Returns:
        Dictionary mapping filename to ValidationResult
    """
    validator = DataValidator(strict_mode=False)
    results = {}

    files_to_validate = {
        "transactions_real.csv": validator.validate_transactions,
        "customers_real.csv": validator.validate_customers,
        "products_real.csv": validator.validate_products,
    }

    for filename, validation_func in files_to_validate.items():
        filepath = f"{data_dir}/{filename}"
        try:
            logger.info(f"Validating {filepath}")
            df = pd.read_csv(filepath)
            result = validation_func(df)
            results[filename] = result

            if result.is_valid:
                logger.info(f"Validation passed for {filename}")
            else:
                logger.warning(
                    f"Validation failed for {filename}: {len(result.errors)} errors"
                )
                for error in result.errors:
                    logger.error(f"  - {error}")

        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            results[filename] = ValidationResult(
                is_valid=False,
                errors=[f"File not found: {filepath}"],
                warnings=[],
                metrics={},
            )
        except Exception as e:
            logger.error(f"Failed to validate {filepath}: {e}", exc_info=True)
            results[filename] = ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metrics={},
            )

    return results


def generate_validation_report(results: Dict[str, ValidationResult]) -> str:
    """
    Generate a validation report.

    Args:
        results: Dictionary of validation results

    Returns:
        Formatted validation report
    """
    report_lines = ["=" * 70, "DATA VALIDATION REPORT", "=" * 70, ""]

    total_errors = sum(len(r.errors) for r in results.values())
    total_warnings = sum(len(r.warnings) for r in results.values())

    report_lines.append(
        f"Overall Status: {'PASSED' if total_errors == 0 else 'FAILED'}"
    )
    report_lines.append(f"Total Errors: {total_errors}")
    report_lines.append(f"Total Warnings: {total_warnings}")
    report_lines.append("")

    for filename, result in results.items():
        report_lines.append(f"\n{filename}:")
        report_lines.append(f"  Status: {'PASSED' if result.is_valid else 'FAILED'}")

        if result.errors:
            report_lines.append(f"  Errors ({len(result.errors)}):")
            for error in result.errors:
                report_lines.append(f"    - {error}")

        if result.warnings:
            report_lines.append(f"  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                report_lines.append(f"    - {warning}")

        if result.metrics:
            report_lines.append("  Metrics:")
            for key, value in result.metrics.items():
                report_lines.append(f"    {key}: {value}")

    report_lines.append("\n" + "=" * 70)

    return "\n".join(report_lines)
