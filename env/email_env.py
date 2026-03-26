from __future__ import annotations
import random
from typing import List, Dict, Optional

from .schemas import Action, ActionType, Observation, Reward, StepResult, EmailItem, Priority
from .tasks import grade_priority_classification, score_reply_quality, score_workflow_completion, Difficulty


class EmailTriageEnv:
    def __init__(self, task_name: str = Difficulty.easy.value, seed: Optional[int] = None, max_steps: int = 30):
        self.task_name = task_name
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.step_count = 0
        self.inbox: List[EmailItem] = []
        self.is_done = False

    def _generate_inbox(self) -> List[EmailItem]:
        candidate = [
            {
                "sender": "alice@example.com",
                "subject": "Q1 report deadline",
                "body": "Can you share the Q1 sales report by EOD?",
                "true_priority": Priority.high,
            },
            {
                "sender": "marketing@example.com",
                "subject": "New branding guidelines",
                "body": "Please review the new branding guidelines and confirm.",
                "true_priority": Priority.medium,
            },
            {
                "sender": "newsletter@example.com",
                "subject": "Weekly community newsletter",
                "body": "Enjoy our weekly updates and industry news.",
                "true_priority": Priority.low,
            },
            {
                "sender": "bob@example.com",
                "subject": "Customer complaint: order #335",
                "body": "Customer is unhappy with delivery time; need urgent fix.",
                "true_priority": Priority.high,
            },
            {
                "sender": "events@example.com",
                "subject": "Invite: team building exercise",
                "body": "RSVP for event by Friday.",
                "true_priority": Priority.medium,
            },
        ]
        self.rng.shuffle(candidate)
        inbox = []
        for idx, c in enumerate(candidate, start=1):
            inbox.append(
                EmailItem(
                    id=idx,
                    sender=c["sender"],
                    subject=c["subject"],
                    body=c["body"],
                    created_at="2026-03-26T10:{:02d}:00Z".format(idx),
                    true_priority=c["true_priority"],
                )
            )
        return inbox

    def reset(self) -> Observation:
        self.step_count = 0
        self.is_done = False
        self.inbox = self._generate_inbox()
        return self._make_observation()

    def _make_observation(self) -> Observation:
        pending = len([e for e in self.inbox if e.status == "unprocessed"])
        processed = len(self.inbox) - pending
        high_remaining = len([e for e in self.inbox if e.status == "unprocessed" and e.true_priority == Priority.high])
        medium_remaining = len([e for e in self.inbox if e.status == "unprocessed" and e.true_priority == Priority.medium])
        low_remaining = len([e for e in self.inbox if e.status == "unprocessed" and e.true_priority == Priority.low])
        current_email = next((e for e in self.inbox if e.status == "unprocessed"), None)

        return Observation(
            current_email=current_email,
            pending=pending,
            processed=processed,
            high_priority_remaining=high_remaining,
            medium_priority_remaining=medium_remaining,
            low_priority_remaining=low_remaining,
            task_name=self.task_name,
            turn=self.step_count,
        )

    def _final_score(self) -> float:
        if self.task_name == Difficulty.easy.value:
            classified = {e.id: e.true_priority for e in self.inbox if e.status == "classified"}
            return grade_priority_classification(self.inbox, classified)
        if self.task_name == Difficulty.medium.value:
            replies = {e.id: e.reply for e in self.inbox if e.reply}
            return score_reply_quality(self.inbox, replies)
        return score_workflow_completion({"inbox": self.inbox})

    def step(self, action: Action) -> StepResult:
        if self.is_done:
            raise RuntimeError("Episode is done; call reset() before step().")
        if self.step_count >= self.max_steps:
            self.is_done = True
            return StepResult(
                observation=self._make_observation(),
                reward=Reward(value=0.0, task_progress=self._final_score(), message="max steps reached"),
                done=True,
                info={"reason": "max_steps"},
            )

        self.step_count += 1
        info: Dict[str, any] = {}
        reward_value = 0.0

        current_email = next((e for e in self.inbox if e.status == "unprocessed"), None)
        if action.type == ActionType.noop:
            reward_value -= 0.01
            info["action"] = "noop"
        elif action.type == ActionType.classify:
            email = next((e for e in self.inbox if e.id == action.email_id), None)
            if not email or action.priority is None:
                reward_value -= 0.2
                info["error"] = "invalid classify action"
            else:
                if email.status != "unprocessed":
                    reward_value -= 0.1
                else:
                    if email.true_priority == action.priority:
                        reward_value += 0.6
                    else:
                        reward_value -= 0.2
                    email.status = "classified"
                info["action"] = "classify"

        elif action.type == ActionType.reply:
            email = next((e for e in self.inbox if e.id == action.email_id), None)
            if not email or not action.reply_body or email.status == "archived":
                reward_value -= 0.3
                info["error"] = "invalid reply action"
            else:
                quality = min(max(len(action.reply_body) / 100.0, 0.0), 1.0)
                bonus = 0.2 if "thank" in action.reply_body.lower() else 0.0
                reward_value += min(1.0, quality + bonus) * 0.7
                email.reply = action.reply_body
                email.status = "replied"
                info["action"] = "reply"

        elif action.type == ActionType.schedule:
            email = next((e for e in self.inbox if e.id == action.email_id), None)
            if not email or not action.schedule_slot:
                reward_value -= 0.2
                info["error"] = "invalid schedule action"
            else:
                email.scheduled_slot = action.schedule_slot
                email.status = "scheduled"
                reward_value += 0.5
                info["action"] = "schedule"

        elif action.type == ActionType.archive:
            email = next((e for e in self.inbox if e.id == action.email_id), None)
            if not email:
                reward_value -= 0.1
                info["error"] = "invalid archive action"
            else:
                email.status = "archived"
                reward_value += 0.3
                info["action"] = "archive"

        done = False
        if self.step_count >= self.max_steps:
            done = True
            self.is_done = True
            info["reason"] = "max_steps"

        # episode feels complete when no unprocessed left in easy/medium/hard
        if all(e.status != "unprocessed" for e in self.inbox):
            done = True
            self.is_done = True
            info["reason"] = "all_processed"

        progress = self._final_score()
        reward = Reward(value=reward_value, task_progress=progress)
        obs = self._make_observation()
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Dict:
        return {
            "task_name": self.task_name,
            "step": self.step_count,
            "inbox": [e.dict() for e in self.inbox],
            "done": self.is_done,
        }
