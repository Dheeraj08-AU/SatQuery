import os
import pytest
from dotenv import load_dotenv
from modules.agent_controller import AgentController

load_dotenv()

@pytest.fixture
def agent():
    # Requires GEMINI_API_KEY in environment or .env
    return AgentController()

def test_real_llm_messy_query_1(agent):
    query = "I need you to highlight the water body for me please"
    input_config = {"image_count": 1, "input_mode": "single"}
    
    result = agent.classify_intent(query, input_config)
    print(f"\nQUERY: '{query}'")
    print(f"TASK: {result.task}")
    print(f"EXTRACTED NOUNS: {result.search_nouns}")
    
    assert result.task == "grounding"
    assert result.search_nouns is not None
    assert len(result.search_nouns) > 0

def test_real_llm_messy_query_2(agent):
    query = "can you locate the built-up area and buildings?"
    input_config = {"image_count": 1, "input_mode": "single"}
    
    result = agent.classify_intent(query, input_config)
    print(f"\nQUERY: '{query}'")
    print(f"TASK: {result.task}")
    print(f"EXTRACTED NOUNS: {result.search_nouns}")
    
    assert result.task == "grounding"
    assert result.search_nouns is not None
    assert len(result.search_nouns) > 0

def test_real_llm_messy_query_3(agent):
    query = "detect the road in this scene right now"
    input_config = {"image_count": 1, "input_mode": "single"}
    
    result = agent.classify_intent(query, input_config)
    print(f"\nQUERY: '{query}'")
    print(f"TASK: {result.task}")
    print(f"EXTRACTED NOUNS: {result.search_nouns}")
    
    assert result.task == "grounding"
    assert result.search_nouns is not None
    assert len(result.search_nouns) > 0
