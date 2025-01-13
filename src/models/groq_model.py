"""
Groq model implementation
"""
from typing import List, Dict, Any, Optional
import groq
from .model_interface import LLMInterface, ModelConfig

class GroqModel(LLMInterface):
    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.api_key = api_key
        self.client = groq.Client(api_key=api_key)
        self.config = config or ModelConfig(model="mixtral-8x7b-32768")
        
    @classmethod
    def from_config_file(cls, config_path: str, api_key: str) -> 'GroqModel':
        """Create Groq model instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['groq'])
        return cls(api_key, config)
        
    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """Generate a response using Groq API."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        
        return response.choices[0].message.content
    
    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate reasoning steps using Groq."""
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
        """Generate embeddings using Groq API."""
        # Note: Groq doesn't provide embedding API yet
        # For now, we'll use OpenAI's embedding API as a fallback
        from .model_interface import OpenAIModel
        openai_model = OpenAIModel(self.api_key)
        return openai_model.embed_text(text)
