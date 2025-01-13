"""
Google Gemini model implementation
"""
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from .model_interface import LLMInterface, ModelConfig

class GeminiModel(LLMInterface):
    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.config = config or ModelConfig(model="gemini-pro")
        self.model = genai.GenerativeModel(self.config.model)
        
    @classmethod
    def from_config_file(cls, config_path: str, api_key: str) -> 'GeminiModel':
        """Create Gemini model instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['gemini'])
        return cls(api_key, config)
        
    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """Generate a response using Gemini API."""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature or self.config.temperature,
                max_output_tokens=max_tokens or self.config.max_tokens
            )
        )
        
        return response.text
    
    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate reasoning steps using Gemini."""
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
        """Generate embeddings using Gemini API."""
        model = genai.GenerativeModel('embedding-001')
        result = model.embed_content(text=text)
        return result.embedding
