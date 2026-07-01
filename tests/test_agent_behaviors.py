import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import Message
from app.services.agent import Agent, AgentConfig
from app.services.catalog import Catalog


def make_offline_agent():
    catalog = Catalog.load()
    agent = Agent(catalog, AgentConfig())
    agent._client = None
    return agent


def test_off_topic_refusal_without_recommendations():
    agent = make_offline_agent()
    result = agent.handle(
        [Message(role="user", content="What interview questions should I ask a backend engineer?")]
    )
    assert "only help with selecting SHL assessments" in result.reply
    assert result.recommendations == []
    assert result.end_of_conversation is False


def test_compare_request_uses_catalog_data():
    agent = make_offline_agent()
    result = agent.handle(
        [Message(role="user", content="What is the difference between OPQ32r and Global Skills Assessment?")]
    )
    assert result.recommendations == []
    assert "Occupational Personality Questionnaire OPQ32r" in result.reply
    assert "Global Skills Assessment" in result.reply
    assert result.end_of_conversation is False


def test_refine_request_updates_shortlist_with_prior_conversation():
    agent = make_offline_agent()
    history = [
        Message(role="user", content="I am hiring a Java developer"),
        Message(role="assistant", content="Here are grounded SHL assessments."),
        Message(role="user", content="Actually add personality tests"),
    ]
    result = agent.handle(history)
    assert "updated the grounded shortlist" in result.reply.lower()
    assert len(result.recommendations) > 0
    assert result.end_of_conversation is False
