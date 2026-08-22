"""Frozen yearly expanding walk-forward split semantics for ML V1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_FIRST_VALIDATION_YEAR = 2018
DEFAULT_BOUNDARY_GAP = 20


@dataclass(frozen=True)
class WalkForwardFold:
    """Integer row positions for one train/validation/test fold."""

    fold_id: str
    validation_year: int
    test_year: int
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    raw_train_positions: tuple[int, ...]
    raw_validation_positions: tuple[int, ...]

    def audit_record(self, dates: pd.DatetimeIndex) -> dict[str, object]:
        """Return explicit raw/used boundaries for the fold manifest table."""

        return {
            "fold_id": self.fold_id,
            "validation_year": self.validation_year,
            "test_year": self.test_year,
            "raw_train_rows": len(self.raw_train_positions),
            "train_rows": len(self.train_positions),
            "raw_validation_rows": len(self.raw_validation_positions),
            "validation_rows": len(self.validation_positions),
            "test_rows": len(self.test_positions),
            "train_start_date": _boundary_date(dates, self.train_positions, "min"),
            "train_end_date": _boundary_date(dates, self.train_positions, "max"),
            "raw_train_end_date": _boundary_date(
                dates, self.raw_train_positions, "max"
            ),
            "validation_start_date": _boundary_date(
                dates, self.validation_positions, "min"
            ),
            "validation_end_date": _boundary_date(
                dates, self.validation_positions, "max"
            ),
            "raw_validation_end_date": _boundary_date(
                dates, self.raw_validation_positions, "max"
            ),
            "test_start_date": _boundary_date(dates, self.test_positions, "min"),
            "test_end_date": _boundary_date(dates, self.test_positions, "max"),
        }


def build_yearly_expanding_folds(
    dates: pd.Series | pd.DatetimeIndex,
    *,
    first_validation_year: int = DEFAULT_FIRST_VALIDATION_YEAR,
    boundary_gap: int = DEFAULT_BOUNDARY_GAP,
) -> list[WalkForwardFold]:
    """Build yearly folds with a trailing gap before validation and test.

    The last ``boundary_gap`` calendar rows of raw training and validation are
    omitted. This prevents a future-entry target of the same maximum horizon
    from crossing either split boundary.
    """

    if isinstance(boundary_gap, bool) or not isinstance(boundary_gap, int):
        raise ValueError("boundary_gap must be a non-negative integer")
    if boundary_gap < 0:
        raise ValueError("boundary_gap must be a non-negative integer")
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce"))
    if len(index) == 0 or index.isna().any():
        raise ValueError("dates must be non-empty and valid")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")

    years = index.year.to_numpy()
    max_year = int(years.max())
    folds: list[WalkForwardFold] = []
    for validation_year in range(first_validation_year, max_year):
        test_year = validation_year + 1
        raw_train = np.flatnonzero(years < validation_year)
        raw_validation = np.flatnonzero(years == validation_year)
        test = np.flatnonzero(years == test_year)
        if not len(raw_validation) or not len(test):
            continue
        if len(raw_train) <= boundary_gap or len(raw_validation) <= boundary_gap:
            raise ValueError(
                f"fold {test_year} has too few rows for boundary_gap={boundary_gap}"
            )
        train = raw_train[:-boundary_gap] if boundary_gap else raw_train
        validation = (
            raw_validation[:-boundary_gap] if boundary_gap else raw_validation
        )
        folds.append(
            WalkForwardFold(
                fold_id=f"wf_{test_year}",
                validation_year=validation_year,
                test_year=test_year,
                train_positions=tuple(int(value) for value in train),
                validation_positions=tuple(int(value) for value in validation),
                test_positions=tuple(int(value) for value in test),
                raw_train_positions=tuple(int(value) for value in raw_train),
                raw_validation_positions=tuple(
                    int(value) for value in raw_validation
                ),
            )
        )
    if not folds:
        raise ValueError("no yearly walk-forward folds can be built")
    return folds


def _boundary_date(
    dates: pd.DatetimeIndex,
    positions: tuple[int, ...],
    operation: str,
) -> pd.Timestamp:
    values = dates[list(positions)]
    return pd.Timestamp(values.min() if operation == "min" else values.max())
