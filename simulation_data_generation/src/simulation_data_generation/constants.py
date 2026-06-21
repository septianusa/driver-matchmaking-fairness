"""Shared constants for the Surabaya dispatch simulation generator."""

from __future__ import annotations

POI_CATEGORIES: tuple[str, ...] = (
    "school",
    "university",
    "office",
    "market",
    "shopping_mall",
    "hospital",
    "transport_hub",
    "residential",
    "restaurant_or_food_area",
    "recreation",
)

SCORE_SEGMENTS: tuple[str, ...] = ("high", "medium", "low")
DEFAULT_TIMEZONE = "Asia/Jakarta"
MINUTES_PER_DAY = 1440
MAX_DRIVER_ONLINE_MINUTES = 480

TIME_PERIOD_ORDER: tuple[str, ...] = (
    "graveyard",
    "morning_peak",
    "daytime_off_peak",
    "evening_peak",
    "evening_off_peak",
    "late_evening",
)
