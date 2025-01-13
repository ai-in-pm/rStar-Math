"""
Integration with llama.cpp for local model inference
"""
from typing import List, Dict, Any, Optional
import json
import os
from llama_cpp import Llama
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import LLMInterface, ModelConfig

class LlamaCppModel(LLMInterface):
    def __init__(self, model_path: str, config: Optional[ModelConfig] = None):
        """Initialize Llama model."""
        self.model_path = model_path
        self.config = config or ModelConfig(
            model="llama2",
            temperature=0.7,
            max_tokens=100
        )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count()
        )
        
    @classmethod
    def from_config_file(cls, config_path: str, model_path: str) -> 'LlamaCppModel':
        """Create Llama model instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['llama'])
        return cls(model_path, config)
        
    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """Generate response using local Llama model."""
        response = self.llm(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            echo=False
        )
        return response['choices'][0]['text'].strip()
    
    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate reasoning steps using Llama."""
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
        """Generate embeddings using Llama."""
        embeddings = self.llm.embed(text)
        return embeddings.tolist()

def main():
    """Example usage of Llama integration."""
    # Initialize components
    model_path = "path/to/llama/model.bin"  # Update with actual path
    model = LlamaCppModel(model_path)
    mcts = MCTS.from_config_file('config/default.json')
    ppm = ProcessPreferenceModel.from_config_file('config/default.json')
    
    # Test problem
    problem = "What is the derivative of x^2?"
    
    print(f"Problem: {problem}\n")
    
    # Generate solution with rStar-Math
    action, trajectory = mcts.search(problem)
    
    print("Solution Steps:")
    for step in trajectory:
        confidence = ppm.evaluate_step(step['state'], model)
        print(f"- {step['state']}")
        print(f"  Confidence: {confidence:.2f}\n")
    
    # Direct solution comparison
    print("Direct Solution:")
    direct_solution = model.generate_response(problem)
    direct_confidence = model.evaluate_reasoning(problem, [direct_solution])
    print(direct_solution)
    print(f"Confidence: {direct_confidence:.2f}")

if __name__ == "__main__":
    main()
