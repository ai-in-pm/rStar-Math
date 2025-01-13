"""
Example integration with Rasa chatbot framework
"""
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import ModelFactory

class RStarMathAction(Action):
    def __init__(self):
        super().__init__()
        self.mcts = MCTS.from_config_file("config/default.json")
        self.ppm = ProcessPreferenceModel.from_config_file("config/default.json")
        self.model = ModelFactory.create_model(
            "openai",
            "YOUR_API_KEY",
            "config/default.json"
        )
        
    def name(self) -> Text:
        return "action_solve_math"
        
    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Get the math problem from user message
        problem = tracker.latest_message.get("text")
        
        try:
            # Solve using rStar-Math
            action, trajectory = self.mcts.search(problem)
            
            # Format solution steps
            steps = []
            for step in trajectory:
                step_text = step["state"]
                step_score = self.ppm.evaluate_step(step_text, self.model)
                steps.append(f"{step_text} (confidence: {step_score:.2f})")
            
            # Send response
            dispatcher.utter_message(
                text=f"Here's how I solved it:\n" + "\n".join(steps)
            )
            
        except Exception as e:
            dispatcher.utter_message(
                text=f"I encountered an error while solving the problem: {str(e)}"
            )
            
        return []
