"""Region-location and forward-outcome evaluation modules."""

from .region_matching import (
    MATCH_COLUMNS,
    METRIC_COLUMNS,
    match_signal_regions,
    summarize_region_matches,
)
from .post_event import (
    build_forward_event_outcomes,
    summarize_forward_event_outcomes,
)
from .region_metrics import (
    add_diagnostic_region_slices,
    summarize_region_slices,
)

__all__ = [
    "MATCH_COLUMNS",
    "METRIC_COLUMNS",
    "add_diagnostic_region_slices",
    "build_forward_event_outcomes",
    "match_signal_regions",
    "summarize_forward_event_outcomes",
    "summarize_region_matches",
    "summarize_region_slices",
]
