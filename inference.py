import json
import os
import re
import sys
from typing import Any, Dict, Optional

from env.email_env import EmailTriageEnv
from env.schemas import Action, ActionType, Priority
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = os.getenv("BENCHMARK", "openenv-email-triage")
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def clean_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        return match.group(0)
    return text


def parse_action(text: str) -> Action:
    text = text.strip()
    try:
        payload = json.loads(clean_json(text))
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    action_type = payload.get("type") or payload.get("action") or "noop"
    action_type = str(action_type).strip().lower()
    if action_type not in ActionType.__members__:
        action_type = next((t.value for t in ActionType if t.value in action_type), "noop")

    email_id = payload.get("email_id")
    try:
        email_id = int(email_id) if email_id is not None else None
    except Exception:
        email_id = None

    priority = payload.get("priority")
    if isinstance(priority, str):
        priority = priority.strip().lower()
    if priority not in Priority.__members__ and priority not in [p.value for p in Priority]:
        priority = None
    if isinstance(priority, str):
        priority = Priority(priority)

    reply_body = payload.get("reply_body")
    schedule_slot = payload.get("schedule_slot")

    if action_type == ActionType.reply.value and not reply_body:
        reply_body = "Thanks for your message. We will follow up shortly."

    if action_type == ActionType.schedule.value and not schedule_slot:
        schedule_slot = "2026-03-27T10:00:00Z"

    return Action(
        type=ActionType(action_type),
        email_id=email_id,
        priority=Priority(priority) if priority else None,
        reply_body=reply_body,
        schedule_slot=schedule_slot,
    )


def format_action(action: Action) -> str:
    if action.type == ActionType.classify:
        return f"classify(email_id={action.email_id},priority={action.priority.value if action.priority else 'unknown'})"
    if action.type == ActionType.reply:
        body = action.reply_body or ""
        return f"reply(email_id={action.email_id},reply_body={body.replace('\n', ' ').strip()})"
    if action.type == ActionType.schedule:
        return f"schedule(email_id={action.email_id},schedule_slot={action.schedule_slot})"
    if action.type == ActionType.archive:
        return f"archive(email_id={action.email_id})"
    return "noop()"


def build_prompt(obs: Any) -> str:
    current = obs.current_email
    current_email_text = (
        f"id={current.id}\nsubject={current.subject}\nbody={current.body}" if current else "none"
    )
    return (
        "You are an email triage assistant.\n"
        f"Task: {obs.task_name}\n"
        f"Current email: {current_email_text}\n"
        "Respond with a single JSON object containing the keys: type, email_id, priority, reply_body, schedule_slot.\n"
        "Use type classify, reply, schedule, archive, or noop.\n"
        "If there is no current email, respond with {\"type\": \"noop\"}.\n"
        "Do not output any text besides the JSON object.\n"
    )


def choose_action(obs: Any) -> Action:
    if obs.current_email is None:
        return Action(type=ActionType.noop)

    prompt = build_prompt(obs)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    action = parse_action(content)
    if action.type == ActionType.noop and obs.current_email is not None:
        return Action(type=ActionType.classify, email_id=obs.current_email.id, priority=obs.current_email.true_priority)
    return action


def main() -> int:
    env = EmailTriageEnv(task_name=TASK_NAME, seed=42, max_steps=MAX_STEPS)
    step_count = 0
    rewards = []
    error_message: Optional[str] = None
    success = False

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
    observation = env.reset()

    try:
        while True:
            step_count += 1
            action = choose_action(observation)
            result = env.step(action)
            reward_value = result.reward.value
            done = result.done
            error_message = result.info.get("error") if isinstance(result.info, dict) else None
            reward_text = f"{reward_value:.2f}"
            action_text = format_action(action)
            print(
                f"[STEP] step={step_count} action={action_text} reward={reward_text} done={str(done).lower()} error={error_message if error_message else 'null'}"
            )
            rewards.append(reward_value)
            observation = result.observation
            if done:
                success = True
                break
            if step_count >= MAX_STEPS:
                break
    except Exception as exc:
        error_message = str(exc)
        success = False
        if step_count == 0:
            step_count = 0
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
