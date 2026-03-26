from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from env.email_env import EmailTriageEnv
from env.schemas import Action, ActionType, Reward, Observation, StepResult
from env.tasks import get_task_list
from env.tasks import grade_priority_classification, score_reply_quality, score_workflow_completion, Difficulty

app = FastAPI(title="OpenEnv Email Triage Environment")

# Default environment instance for API interactions.
ENV = EmailTriageEnv(task_name=Difficulty.easy.value, seed=42)
ENV.reset()


class BaselineRequest(BaseModel):
    task_name: str = Difficulty.easy.value
    model: str = "gpt-3.5-turbo"


@app.get("/ping")
def ping() -> Dict[str, str]:
    return {"status": "ok", "env": "email-triage"}


@app.post("/reset")
def api_reset(task_name: Optional[str] = Difficulty.easy.value) -> Observation:
    if task_name not in [Difficulty.easy.value, Difficulty.medium.value, Difficulty.hard.value]:
        raise HTTPException(status_code=400, detail="Unknown task_name")
    global ENV
    ENV = EmailTriageEnv(task_name=task_name, seed=42)
    obs = ENV.reset()
    return obs


@app.post("/step")
def api_step(action: Action) -> StepResult:
    global ENV
    try:
        result = ENV.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
def api_state() -> Dict[str, Any]:
    return ENV.state()


@app.post("/grader")
def api_grader() -> Dict[str, Any]:
    state = ENV.state()
    task_name = state["task_name"]
    if task_name == Difficulty.easy.value:
        classified = {e["id"]: e.get("true_priority") for e in state["inbox"] if e["status"] == "classified"}
        score = grade_priority_classification([n for n in state["inbox"]], classified)
    elif task_name == Difficulty.medium.value:
        replies = {e["id"]: e.get("reply") for e in state["inbox"] if e.get("reply")}
        score = score_reply_quality([n for n in state["inbox"]], replies)
    else:
        score = score_workflow_completion(state)
    return {"task": task_name, "score": score}


@app.get("/tasks")
def api_tasks() -> List[Dict[str, Any]]:
    return get_task_list()


@app.post("/baseline")
def api_baseline(request: BaselineRequest) -> Dict[str, Any]:
    from baseline import run_baseline

    score = run_baseline(task_name=request.task_name, model=request.model)
    return {"task": request.task_name, "baseline_score": score}
