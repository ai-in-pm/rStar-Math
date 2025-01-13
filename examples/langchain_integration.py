"""
Example integration with LangChain framework
"""
from typing import Any, Dict, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import ModelFactory

class RStarMathChain(LLMChain):
    def __init__(
        self,
        llm: BaseLLM,
        prompt: PromptTemplate = None,
        api_key: str = "YOUR_API_KEY",
        config_path: str = "config/default.json"
    ):
        if prompt is None:
            prompt = PromptTemplate(
                input_variables=["problem"],
                template="Solve this math problem: {problem}"
            )
            
        super().__init__(llm=llm, prompt=prompt)
        
        # Initialize rStar-Math components
        self.mcts = MCTS.from_config_file(config_path)
        self.ppm = ProcessPreferenceModel.from_config_file(config_path)
        self.model = ModelFactory.create_model(
            "openai",
            api_key,
            config_path
        )
        
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Process the input using rStar-Math enhanced reasoning."""
        problem = inputs["problem"]
        
        # Get direct solution from LLM (System 1)
        direct_solution = super()._call(inputs)
        
        # Get enhanced solution using rStar-Math (System 2)
        action, trajectory = self.mcts.search(problem)
        
        # Evaluate and compare solutions
        direct_score = self.model.evaluate_reasoning(
            problem,
            [direct_solution["text"]]
        )
        
        enhanced_steps = []
        total_score = 0.0
        for step in trajectory:
            step_text = step["state"]
            step_score = self.ppm.evaluate_step(step_text, self.model)
            enhanced_steps.append(f"{step_text} (confidence: {step_score:.2f})")
            total_score += step_score
            
        enhanced_score = total_score / len(trajectory) if trajectory else 0.0
        
        return {
            "direct_solution": direct_solution["text"],
            "direct_score": direct_score,
            "enhanced_solution": "\n".join(enhanced_steps),
            "enhanced_score": enhanced_score,
            "improvement": enhanced_score - direct_score
        }
        
# Example usage:
"""
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
chain = RStarMathChain(llm=llm, api_key="YOUR_API_KEY")

result = chain.run("What is 2 + 2?")
print(f"Direct solution (score: {result['direct_score']:.2f}):")
print(result['direct_solution'])
print("\nEnhanced solution (score: {result['enhanced_score']:.2f}):")
print(result['enhanced_solution'])
print(f"\nImprovement: {result['improvement']:.2f}")
"""
