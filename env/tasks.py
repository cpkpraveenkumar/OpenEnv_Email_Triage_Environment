from __future__ import annotations
from typing import List, Dict
from enum import Enum

from .schemas import EmailItem, Priority


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


def grade_priority_classification(emails: List[EmailItem], classified: Dict[int, Priority]) -> float:
    if not emails:
        return 0.0
    correct = 0
    for e in emails:
        if classified.get(e.id) == e.true_priority:
            correct += 1
    return correct / len(emails)


def score_reply_quality(emails: List[EmailItem], replies: Dict[int, str]) -> float:
    if not emails:
        return 0.0
    total = 0.0
    for e in emails:
        r = replies.get(e.id, "")
        if not r:
            continue
        # simple quality heuristic: length, problem-specific keyword, personalization
        length_score = min(max(len(r) / 120, 0.0), 1.0)
        keyword = "thank" if e.true_priority != Priority.high else "priority"
        keyword_score = 0.2 if keyword in r.lower() else 0.0
        personalization = 0.2 if e.sender.split("@")[0] in r else 0.0
        total += min(1.0, length_score + keyword_score + personalization)
    return total / len(emails)


def score_workflow_completion(state: Dict) -> float:
    emails = state.get("inbox", [])
    if not emails:
        return 0.0
    completed = 0
    for e in emails:
        status = e.status if hasattr(e, "status") else e.get("status")
        if status in ["archived", "replied", "scheduled"]:
            completed += 1
    return completed / len(emails)


def get_task_list() -> List[Dict]:
    return [
        {
            "name": Difficulty.easy.value,
            "description": "Classify each incoming email into low/medium/high priority before time runs out.",
            "difficulty": 1,
        },
        {
            "name": Difficulty.medium.value,
            "description": "Draft a helpful reply for each unanswered email, with relevant phrasing and references.",
            "difficulty": 2,
        },
        {
            "name": Difficulty.hard.value,
            "description": "Complete triage: classify, reply or archive, and schedule follow-ups where appropriate.",
            "difficulty": 3,
        },
    ]
