"""OpenCesium tasks package."""

from tasks.easy import TASK_EASY, grade_easy
from tasks.medium import TASK_MEDIUM, grade_medium
from tasks.hard import TASK_HARD, grade_hard
from tasks.registry import TASK_REGISTRY

__all__ = [
    "TASK_EASY",
    "TASK_MEDIUM",
    "TASK_HARD",
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "TASK_REGISTRY",
]
