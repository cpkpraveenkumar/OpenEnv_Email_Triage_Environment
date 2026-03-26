from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ActionType(str, Enum):
    classify = "classify"
    reply = "reply"
    schedule = "schedule"
    archive = "archive"
    noop = "noop"


class EmailItem(BaseModel):
    id: int
    sender: str
    subject: str
    body: str
    created_at: str
    true_priority: Priority
    status: str = Field(default="unprocessed")
    reply: Optional[str] = None
    scheduled_slot: Optional[str] = None


class Observation(BaseModel):
    current_email: Optional[EmailItem]
    pending: int
    processed: int
    high_priority_remaining: int
    medium_priority_remaining: int
    low_priority_remaining: int
    task_name: str
    turn: int


class Action(BaseModel):
    type: ActionType
    email_id: Optional[int] = None
    priority: Optional[Priority] = None
    reply_body: Optional[str] = None
    schedule_slot: Optional[str] = None


class Reward(BaseModel):
    value: float
    task_progress: float
    message: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]
