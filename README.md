---
title: OpenEnv Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
tags: ["openenv"]
---

# OpenEnv Email Triage Environment

A real-world environment that simulates email triage and workflow management for AI agents.

## Motivation
This environment replicates a common knowledge worker task: handling incoming emails by
1) classifying priority,
2) drafting replies,
3) scheduling follow-ups,
4) archiving processed messages.

This is useful for training and evaluating agents on real-world productivity tasks.

## Features
- OpenEnv compliant (`openenv.yaml` + typed `Observation`, `Action`, `Reward`)
- `step()`, `reset()`, `state()` API
- 3 tasks (easy / medium / hard) with deterministic graders and a 0.0–1.0 score
- partial reward signal, against `max_steps`, invalid action penalties
- baseline using OpenAI API: `baseline.py`
- FastAPI endpoints including `/tasks`, `/grader`, `/baseline`
- containerized with `Dockerfile` for HuggingFace Space deployment

## Action / Observation Definitions
- `Observation`: current email details, counters of pending/processed, priority breakdown, task name, turn.
- `Action`: fields:
  - `type`: `classify`, `reply`, `schedule`, `archive`, `noop`
  - `email_id`: int
  - `priority`: `low|medium|high` (for classify)
  - `reply_body`: string (for reply)
  - `schedule_slot`: ISO datetime string (for schedule)

- `Reward`: `value` (float), `task_progress` (float 0.0..1.0), optional `message`.

## Tasks
1. easy: classify all emails with correct priority
2. medium: respond to each email with quality text
3. hard: full workflow: classify + reply/and or archive + schedule follow-ups

## Setup

1. install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. run tests
   ```bash
   pytest -q tests
   ```

3. run validation
   ```bash
   python validation.py
   ```

4. run server
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 7860
   ```

5. call API
   - `POST /reset?task_name=easy`
   - `POST /step` with JSON action
   - `GET /state`
   - `GET /tasks`
   - `POST /grader`
   - `POST /baseline` (calls baseline function)

## Baseline
Set `OPENAI_API_KEY` in env.

```bash
export OPENAI_API_KEY="your_api_key"
python baseline.py
```

Baseline performance scores (using GPT-3.5-turbo):
- Easy: 1.000 (perfect classification)
- Medium: 0.985 (high-quality replies)
- Hard: 0.000 (needs improvement)

Scores are deterministic with the built-in heuristic when API key is absent.

## Hugging Face Space
The environment is deployed on Hugging Face Spaces for easy access and testing.

**Live URL**: https://praveenkumar2007-openenv-email-triage-environment.hf.space/

1. create new Space with container runtime.
2. push repository.
3. ensure `Dockerfile` is picked.
4. validate space endpoint:
   - `GET /ping` should return `{"status":"ok"}`
   - `POST /reset` should work.

## Compatibility
- Python 3.12+
- Works with `docker build -t openenv-email .` and `docker run --rm -p 7860:7860 openenv-email`
