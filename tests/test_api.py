"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import OpenAIModel

client = TestClient(app)

@pytest.fixture
def mock_components():
    with patch('src.core.mcts.MCTS') as mock_mcts, \
         patch('src.core.ppm.ProcessPreferenceModel') as mock_ppm, \
         patch('src.models.model_interface.OpenAIModel') as mock_model:
        
        # Configure MCTS mock
        mock_mcts_instance = Mock()
        mock_mcts_instance.search.return_value = ("test_action", [{"state": "test", "value": 1.0}])
        mock_mcts.return_value = mock_mcts_instance
        
        # Configure PPM mock
        mock_ppm_instance = Mock()
        mock_ppm_instance.evaluate_step.return_value = 0.8
        mock_ppm.return_value = mock_ppm_instance
        
        # Configure Model mock
        mock_model_instance = Mock()
        mock_model_instance.generate_response.return_value = "4"
        mock_model_instance.evaluate_reasoning.return_value = 0.9
        mock_model.return_value = mock_model_instance
        
        yield mock_mcts_instance, mock_ppm_instance, mock_model_instance

def test_solve_endpoint(mock_components):
    mcts_mock, ppm_mock, model_mock = mock_components
    
    response = client.post(
        "/solve",
        json={
            "problem_text": "What is 2 + 2?",
            "model_name": "gpt-4",
            "use_rstar": True,
            "mcts_simulations": 100,
            "temperature": 0.7
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "solution_steps" in data
    assert "confidence_score" in data
    assert "reasoning_path" in data
    assert "execution_time" in data

def test_compare_models_endpoint(mock_components):
    mcts_mock, ppm_mock, model_mock = mock_components
    
    response = client.post(
        "/compare-models",
        json={
            "problem_text": "What is 2 + 2?",
            "model_name": "all",
            "use_rstar": True,
            "mcts_simulations": 100,
            "temperature": 0.7
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "openai" in data
    assert "anthropic" in data
    
    for model_results in data.values():
        assert "solution" in model_results
        assert "score" in model_results
        assert "execution_time" in model_results

def test_generate_integration_code_endpoint(mock_components):
    response = client.post(
        "/generate-integration-code",
        json={
            "framework": "rasa",
            "config": {
                "model": "gpt-4",
                "use_rstar": True
            }
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "code" in data
    assert "rstar_math" in data["code"].lower()

def test_invalid_input():
    response = client.post(
        "/solve",
        json={
            "invalid_field": "test"
        }
    )
    assert response.status_code == 422

def test_error_handling(mock_components):
    mcts_mock, ppm_mock, model_mock = mock_components
    model_mock.generate_response.side_effect = Exception("API Error")
    
    response = client.post(
        "/solve",
        json={
            "problem_text": "What is 2 + 2?",
            "model_name": "gpt-4",
            "use_rstar": True
        }
    )
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
