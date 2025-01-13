"""
Utility functions for rStar-Math Demonstrator
"""
from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def format_solution_steps(steps: List[str]) -> str:
    """Format solution steps for display."""
    return "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

def calculate_metrics(predictions: List[float], 
                     targets: List[float]) -> Dict[str, float]:
    """Calculate performance metrics."""
    if not predictions or not targets:
        return {}
    
    accuracy = sum(1 for p, t in zip(predictions, targets) if abs(p - t) < 1e-6) / len(predictions)
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    
    return {
        "accuracy": accuracy,
        "mse": mse
    }

def generate_framework_template(framework: str,
                              config: Dict[str, Any]) -> str:
    """Generate integration code template for specified framework."""
    templates = {
        "rasa": """
from rasa_sdk import Action
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel

class RStarMathAction(Action):
    def name(self) -> str:
        return "action_solve_math"
        
    def run(self, dispatcher, tracker, domain):
        # Implementation details
        pass
""",
        "azure": """
from botbuilder.core import TurnContext
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel

class RStarMathBot:
    async def on_message_activity(self, turn_context: TurnContext):
        # Implementation details
        pass
""",
        "langchain": """
from langchain import LLMChain
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel

class RStarMathChain(LLMChain):
    def __init__(self):
        # Implementation details
        pass
        
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        # Implementation details
        pass
"""
    }
    
    return templates.get(framework, "Framework template not found")

def log_execution(func_name: str,
                 start_time: float,
                 end_time: float,
                 success: bool,
                 error: Optional[str] = None) -> None:
    """Log execution details for monitoring."""
    execution_time = end_time - start_time
    log_entry = {
        "function": func_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": execution_time,
        "success": success,
        "error": error
    }
    
    # Implementation details for logging to be added
    pass
