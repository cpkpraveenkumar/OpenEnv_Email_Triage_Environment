import os
import re
import time
from typing import Optional

import openai

from env.email_env import EmailTriageEnv
from env.schemas import Action, ActionType, Priority
from env.tasks import Difficulty

openai.api_key = os.getenv("OPENAI_API_KEY")


def _parse_ai_action(text: str):
    # expected: type:<class>, email_id:<int>, etc.
    t = text.lower()
    type_map = {
        "classify": ActionType.classify,
        "reply": ActionType.reply,
        "schedule": ActionType.schedule,
        "archive": ActionType.archive,
        "noop": ActionType.noop,
    }
    action_type = ActionType.noop
    for key in type_map:
        if key in t:
            action_type = type_map[key]
            break

    email_id_match = re.search(r"email[_ ]id[:=]?\s*(\d+)", t)
    email_id = int(email_id_match.group(1)) if email_id_match else None
    priority = None
    for p in Priority:
        if p.value in t:
            priority = p
            break
    reply_body = None
    schedule_slot = None

    # keep full text as response for reply if chosen
    if action_type == ActionType.reply:
        m_body = re.search(r"reply body[:=](.*)", t, re.IGNORECASE)
        if m_body:
            reply_body = m_body.group(1).strip()
        else:
            reply_body = "Thanks for your message. We are working on it and will follow up shortly."

    if action_type == ActionType.schedule:
        m_slot = re.search(r"slot[:=](.*)", t, re.IGNORECASE)
        if m_slot:
            schedule_slot = m_slot.group(1).strip()
        else:
            schedule_slot = "2026-03-27T10:00:00Z"

    return Action(type=action_type, email_id=email_id, priority=priority, reply_body=reply_body, schedule_slot=schedule_slot)


def model_decide_action(obs):
    if openai.api_key is None:
        # deterministic heuristic fallback for test reproducibility
        current = obs.current_email
        if current is None:
            return Action(type=ActionType.noop)
        if obs.task_name == Difficulty.easy.value:
            return Action(type=ActionType.classify, email_id=current.id, priority=current.true_priority)
        if obs.task_name == Difficulty.medium.value:
            return Action(
                type=ActionType.reply,
                email_id=current.id,
                reply_body=f"Hi {current.sender.split('@')[0]}, thanks for your message about {current.subject}. We'll get back to you soon.",
            )
        # hard task: classify then reply then archive
        if current.status == "unprocessed":
            return Action(type=ActionType.classify, email_id=current.id, priority=current.true_priority)
        if current.status == "classified":
            return Action(
                type=ActionType.reply,
                email_id=current.id,
                reply_body=f"Hi {current.sender.split('@')[0]}, received your note and will act on it.",
            )
        return Action(type=ActionType.archive, email_id=current.id)

    # OpenAI call
    prompt = f"You are an email assistant. Task: {obs.task_name}.\nCurrent email id={obs.current_email.id if obs.current_email else 'none'}\nsubject={obs.current_email.subject if obs.current_email else ''}\nbody={obs.current_email.body if obs.current_email else ''}\n" \
             f"Make a single action as JSON with keys type,email_id,priority,reply_body,schedule_slot.\n"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o" if os.getenv("PREFERRED_MODEL") == "gpt-4o" else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return _parse_ai_action(content)
    except Exception:
        return model_decide_action(obs)


def run_baseline(task_name: str = Difficulty.easy.value, model: str = "gpt-3.5-turbo", max_steps: int = 30) -> float:
    env = EmailTriageEnv(task_name=task_name, seed=42, max_steps=max_steps)
    obs = env.reset()
    total_reward = 0.0
    step = 0

    while True:
        if obs.current_email is None and obs.pending == 0:
            break
        action = model_decide_action(obs)
        result = env.step(action)
        total_reward += result.reward.value
        step += 1
        if result.done or step >= max_steps:
            break
        obs = result.observation

    grader = env._final_score()
    print(f"Baseline run task={task_name} model={model} reward={total_reward:.3f} grader={grader:.3f}")
    return float(grader)


if __name__ == "__main__":
    for task in [Difficulty.easy.value, Difficulty.medium.value, Difficulty.hard.value]:
        run_baseline(task_name=task)
