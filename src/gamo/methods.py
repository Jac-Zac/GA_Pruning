"""Stable identifiers shared by experiments, artifacts, and reports."""

RANDOM = "Random"
MAGNITUDE = "Magnitude"
MAGNITUDE_PER_LAYER = "Magnitude (per-layer)"
RANDOM_SEARCH = "Random search (equal budget)"
HILL_CLIMBING = "Hill climbing (equal budget)"

STATIC_METHODS = (RANDOM, MAGNITUDE, MAGNITUDE_PER_LAYER)
STRUCTURED_METHODS = (
    "GA-uniform",
    "GA-two_point",
    HILL_CLIMBING,
    RANDOM_SEARCH,
    MAGNITUDE,
    MAGNITUDE_PER_LAYER,
    RANDOM,
)


def ga_label(crossover: str) -> str:
    return f"GA-{crossover}"
