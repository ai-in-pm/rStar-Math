"""
Tests for LLM model implementations
"""
import pytest
from unittest.mock import Mock, patch
from src.models.model_interface import (
    ModelConfig,
    OpenAIModel,
    AnthropicModel,
    ModelFactory
)

@pytest.fixture
def mock_openai_response():
    return Mock(
        choices=[
            Mock(
                message=Mock(
                    content="Test response"
                )
            )
        ],
        data=[
            Mock(
                embedding=[0.1, 0.2, 0.3]
            )
        ]
    )

@pytest.fixture
def mock_anthropic_response():
    return Mock(content="Test response")

def test_model_config():
    config = ModelConfig(model="test-model", temperature=0.5, max_tokens=100)
    assert config.model == "test-model"
    assert config.temperature == 0.5
    assert config.max_tokens == 100

@patch('openai.ChatCompletion.create')
def test_openai_generate_response(mock_create, mock_openai_response):
    mock_create.return_value = mock_openai_response
    model = OpenAIModel("test-key")
    
    response = model.generate_response("test prompt")
    assert response == "Test response"
    
    mock_create.assert_called_once()
    call_args = mock_create.call_args[1]
    assert call_args['messages'][0]['content'] == "test prompt"

@patch('openai.Embedding.create')
def test_openai_embed_text(mock_create, mock_openai_response):
    mock_create.return_value = mock_openai_response
    model = OpenAIModel("test-key")
    
    embedding = model.embed_text("test text")
    assert embedding == [0.1, 0.2, 0.3]
    
    mock_create.assert_called_once_with(
        model="text-embedding-ada-002",
        input="test text"
    )

@patch('anthropic.Client')
def test_anthropic_generate_response(mock_client, mock_anthropic_response):
    mock_instance = Mock()
    mock_instance.messages.create.return_value = mock_anthropic_response
    mock_client.return_value = mock_instance
    
    model = AnthropicModel("test-key")
    response = model.generate_response("test prompt")
    
    assert response == "Test response"
    mock_instance.messages.create.assert_called_once()

def test_model_factory():
    with pytest.raises(ValueError):
        ModelFactory.create_model("unknown-model", "test-key")
        
    model = ModelFactory.create_model("openai", "test-key")
    assert isinstance(model, OpenAIModel)
    
    model = ModelFactory.create_model("anthropic", "test-key")
    assert isinstance(model, AnthropicModel)

def test_evaluate_reasoning():
    test_problem = "What is 2 + 2?"
    test_steps = ["First, we identify that this is an addition problem",
                 "Then, we add 2 and 2 together",
                 "Therefore, 2 + 2 = 4"]
    
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = Mock(
            choices=[Mock(message=Mock(content="0.8"))]
        )
        
        model = OpenAIModel("test-key")
        score = model.evaluate_reasoning(test_problem, test_steps)
        
        assert 0 <= score <= 1
        mock_create.assert_called_once()
