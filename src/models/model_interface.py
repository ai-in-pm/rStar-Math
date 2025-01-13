"""
Interface for different LLM implementations
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import openai
import anthropic
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    
class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, 
                         prompt: str, 
                         temperature: float = 0.7,
                         max_tokens: int = 1000) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def evaluate_reasoning(self, 
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate the quality of reasoning steps."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass

class OpenAIModel(LLMInterface):
    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.api_key = api_key
        openai.api_key = api_key
        self.config = config or ModelConfig(model="gpt-4")
        
    @classmethod
    def from_config_file(cls, config_path: str, api_key: str) -> 'OpenAIModel':
        """Create OpenAI model instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['openai'])
        return cls(api_key, config)
        
    def generate_response(self, 
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """Generate a response using OpenAI API."""
        response = openai.ChatCompletion.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        return response.choices[0].message.content
    
    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate reasoning steps using OpenAI."""
        prompt = f"""
        Problem: {problem}
        Solution Steps:
        {chr(10).join(f'{i+1}. {step}' for i, step in enumerate(solution_steps))}
        
        Rate the quality of these solution steps from 0 to 1, where:
        0 = completely incorrect or invalid reasoning
        1 = perfect, clear, and mathematically sound reasoning
        
        Provide only the numerical rating.
        """
        
        response = self.generate_response(prompt)
        try:
            rating = float(response.strip())
            return max(0.0, min(1.0, rating))
        except ValueError:
            return 0.0
            
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API."""
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

class AnthropicModel(LLMInterface):
    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.api_key = api_key
        self.client = anthropic.Client(api_key=api_key)
        self.config = config or ModelConfig(model="claude-2")
        
    @classmethod
    def from_config_file(cls, config_path: str, api_key: str) -> 'AnthropicModel':
        """Create Anthropic model instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['anthropic'])
        return cls(api_key, config)
        
    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """Generate a response using Anthropic API."""
        response = self.client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        return response.content
    
    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate reasoning steps using Anthropic."""
        prompt = f"""
        Problem: {problem}
        Solution Steps:
        {chr(10).join(f'{i+1}. {step}' for i, step in enumerate(solution_steps))}
        
        Rate the quality of these solution steps from 0 to 1, where:
        0 = completely incorrect or invalid reasoning
        1 = perfect, clear, and mathematically sound reasoning
        
        Provide only the numerical rating.
        """
        
        response = self.generate_response(prompt)
        try:
            rating = float(response.strip())
            return max(0.0, min(1.0, rating))
        except ValueError:
            return 0.0
            
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using Anthropic API."""
        # Note: Anthropic doesn't provide embedding API yet
        # For now, we'll use OpenAI's embedding API as a fallback
        openai_model = OpenAIModel(self.api_key)
        return openai_model.embed_text(text)

# Additional model implementations (Mistral, Groq, Gemini, Cohere, Emergence)
# would follow similar pattern

class ModelFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_model(model_type: str,
                    api_key: str,
                    config_path: Optional[str] = None) -> LLMInterface:
        """Create a model instance based on type."""
        if config_path:
            if model_type.lower() == "openai":
                return OpenAIModel.from_config_file(config_path, api_key)
            elif model_type.lower() == "anthropic":
                return AnthropicModel.from_config_file(config_path, api_key)
            # Add other model types here
        else:
            if model_type.lower() == "openai":
                return OpenAIModel(api_key)
            elif model_type.lower() == "anthropic":
                return AnthropicModel(api_key)
            # Add other model types here
            
        raise ValueError(f"Unknown model type: {model_type}")
