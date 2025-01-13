"""
Tests for new model implementations (Mistral, Groq, Gemini)
"""
import pytest
from unittest.mock import Mock, patch
from src.models.mistral_model import MistralModel
from src.models.groq_model import GroqModel
from src.models.gemini_model import GeminiModel
from src.models.model_interface import ModelConfig

@pytest.fixture
def mock_config():
    return ModelConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100
    )

@pytest.fixture
def mock_api_response():
    return Mock(
        choices=[Mock(message=Mock(content="Test response"))],
        data=[Mock(embedding=[0.1, 0.2, 0.3])]
    )

class TestMistralModel:
    @pytest.fixture
    def model(self):
        return MistralModel("test-key", mock_config())
    
    @patch('mistralai.client.MistralClient')
    def test_generate_response(self, mock_client, model):
        mock_client.return_value.chat.return_value = mock_api_response
        response = model.generate_response("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0
        
    @patch('mistralai.client.MistralClient')
    def test_evaluate_reasoning(self, mock_client, model):
        mock_client.return_value.chat.return_value = mock_api_response
        score = model.evaluate_reasoning(
            "What is 2+2?",
            ["First step", "Second step"]
        )
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
    @patch('mistralai.client.MistralClient')
    def test_embed_text(self, mock_client, model):
        mock_client.return_value.embeddings.return_value = mock_api_response
        embedding = model.embed_text("Test text")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

class TestGroqModel:
    @pytest.fixture
    def model(self):
        return GroqModel("test-key", mock_config())
    
    @patch('groq.Client')
    def test_generate_response(self, mock_client, model):
        mock_client.return_value.chat.completions.create.return_value = mock_api_response
        response = model.generate_response("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0
        
    @patch('groq.Client')
    def test_evaluate_reasoning(self, mock_client, model):
        mock_client.return_value.chat.completions.create.return_value = mock_api_response
        score = model.evaluate_reasoning(
            "What is 2+2?",
            ["First step", "Second step"]
        )
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
    @patch('groq.Client')
    def test_embed_text(self, mock_client, model):
        # Test fallback to OpenAI embeddings
        with patch('src.models.model_interface.OpenAIModel') as mock_openai:
            mock_openai.return_value.embed_text.return_value = [0.1, 0.2, 0.3]
            embedding = model.embed_text("Test text")
            assert isinstance(embedding, list)
            assert all(isinstance(x, float) for x in embedding)

class TestGeminiModel:
    @pytest.fixture
    def model(self):
        return GeminiModel("test-key", mock_config())
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_response(self, mock_model, model):
        mock_model.return_value.generate_content.return_value = Mock(text="Test response")
        response = model.generate_response("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0
        
    @patch('google.generativeai.GenerativeModel')
    def test_evaluate_reasoning(self, mock_model, model):
        mock_model.return_value.generate_content.return_value = Mock(text="0.8")
        score = model.evaluate_reasoning(
            "What is 2+2?",
            ["First step", "Second step"]
        )
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
    @patch('google.generativeai.GenerativeModel')
    def test_embed_text(self, mock_model, model):
        mock_model.return_value.embed_content.return_value = Mock(
            embedding=[0.1, 0.2, 0.3]
        )
        embedding = model.embed_text("Test text")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

# Integration tests
@pytest.mark.integration
class TestModelIntegration:
    @pytest.mark.parametrize("model_class", [
        MistralModel,
        GroqModel,
        GeminiModel
    ])
    def test_model_chain(self, model_class):
        """Test complete chain of operations."""
        model = model_class("test-key", mock_config())
        
        # Generate response
        response = model.generate_response("What is 2+2?")
        assert isinstance(response, str)
        
        # Evaluate reasoning
        score = model.evaluate_reasoning(
            "What is 2+2?",
            ["First, identify this is addition", "Then, add 2 and 2", "Therefore, 2+2=4"]
        )
        assert isinstance(score, float)
        
        # Generate embeddings
        embedding = model.embed_text("Test text")
        assert isinstance(embedding, list)
        
    @pytest.mark.parametrize("model_class", [
        MistralModel,
        GroqModel,
        GeminiModel
    ])
    def test_error_handling(self, model_class):
        """Test error handling in models."""
        model = model_class("invalid-key", mock_config())
        
        # Test invalid API key
        with pytest.raises(Exception):
            model.generate_response("Test prompt")
            
        # Test invalid prompt
        with pytest.raises(Exception):
            model.generate_response("")
            
        # Test invalid reasoning steps
        score = model.evaluate_reasoning("Test", [])
        assert score == 0.0
