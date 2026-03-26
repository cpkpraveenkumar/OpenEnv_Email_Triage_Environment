import pytest

from env.email_env import EmailTriageEnv
from env.schemas import Action, ActionType, Priority
from env.tasks import Difficulty


def test_reset_and_state():
    env = EmailTriageEnv(task_name=Difficulty.easy.value, seed=123)
    obs = env.reset()
    assert obs.pending == 5
    state = env.state()
    assert state["done"] is False
    assert state["step"] == 0


def test_easy_classification():
    env = EmailTriageEnv(task_name=Difficulty.easy.value, seed=1, max_steps=10)
    env.reset()
    for e in env.inbox:
        action = Action(type=ActionType.classify, email_id=e.id, priority=e.true_priority)
        result = env.step(action)
    grade = env._final_score()
    assert grade == 1.0


def test_invalid_action_penalty():
    env = EmailTriageEnv(task_name=Difficulty.easy.value, seed=1, max_steps=10)
    env.reset()
    result = env.step(Action(type=ActionType.archive, email_id=999))
    assert result.reward.value < 0


def test_medium_reply_quality():
    env = EmailTriageEnv(task_name=Difficulty.medium.value, seed=1, max_steps=10)
    env.reset()
    for e in env.inbox:
        action = Action(type=ActionType.reply, email_id=e.id, reply_body=f"Thanks {e.sender.split('@')[0]}, got it.")
        env.step(action)
    score = env._final_score()
    assert 0.0 <= score <= 1.0
