from env.tasks import grade_priority_classification, score_reply_quality, score_workflow_completion
from env.schemas import EmailItem, Priority


def test_grade_priority_classification():
    emails = [EmailItem(id=1, sender="a@x.com", subject="", body="", created_at="", true_priority=Priority.high)]
    score = grade_priority_classification(emails, {1: Priority.high})
    assert score == 1.0


def test_score_reply_quality():
    emails = [EmailItem(id=1, sender="a@x.com", subject="Request", body="", created_at="", true_priority=Priority.medium)]
    score = score_reply_quality(emails, {1: "Thanks a, we got your request. Thank you."})
    assert 0.0 <= score <= 1.0


def test_score_workflow_completion():
    state = {"inbox": [{"status": "replied"}, {"status": "archived"}, {"status": "scheduled"}]}
    score = score_workflow_completion(state)
    assert score == 1.0
