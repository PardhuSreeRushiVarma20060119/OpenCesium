"""Task registry — maps task IDs to (TaskDefinition, grader) pairs."""

from tasks.easy import TASK_EASY, grade_easy
from tasks.medium import TASK_MEDIUM, grade_medium
from tasks.hard import TASK_HARD, grade_hard

# Maps task_id -> (TaskDefinition, grader callable)
TASK_REGISTRY: dict = {
    TASK_EASY.task_id: (TASK_EASY, grade_easy),
    TASK_MEDIUM.task_id: (TASK_MEDIUM, grade_medium),
    TASK_HARD.task_id: (TASK_HARD, grade_hard),
}

__all__ = ["TASK_REGISTRY"]
