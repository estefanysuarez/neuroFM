"""Objectives: classify, regress, masked prediction."""

from enum import Enum


class Objective(str, Enum):
    """Available training objectives."""
    CLASSIFY = "classify"
    REGRESS = "regress"
    MASKED_PREDICTION = "masked_prediction"
