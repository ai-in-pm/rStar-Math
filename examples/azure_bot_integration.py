"""
Example integration with Azure Bot Framework
"""
from botbuilder.core import TurnContext, ActivityHandler
from botbuilder.schema import Activity
import asyncio
from typing import List, Dict, Any

from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import ModelFactory

class RStarMathBot(ActivityHandler):
    def __init__(self, api_key: str = "YOUR_API_KEY", config_path: str = "config/default.json"):
        super().__init__()
        
        # Initialize rStar-Math components
        self.mcts = MCTS.from_config_file(config_path)
        self.ppm = ProcessPreferenceModel.from_config_file(config_path)
        self.model = ModelFactory.create_model(
            "openai",
            api_key,
            config_path
        )
        
    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages."""
        # Get the math problem from user message
        problem = turn_context.activity.text
        
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
            await turn_context.send_activity(
                f"Here's how I solved it:\n" + "\n".join(steps)
            )
            
        except Exception as e:
            await turn_context.send_activity(
                f"I encountered an error while solving the problem: {str(e)}"
            )
            
    async def on_members_added_activity(
        self,
        members_added: List[ChannelAccount],
        turn_context: TurnContext
    ):
        """Welcome new users."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "Welcome! I'm an AI math tutor powered by rStar-Math. "
                    "Send me any math problem, and I'll solve it step by step!"
                )

# Example usage:
"""
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings

# Initialize bot
SETTINGS = BotFrameworkAdapterSettings("APP_ID", "APP_PASSWORD")
ADAPTER = BotFrameworkAdapter(SETTINGS)
BOT = RStarMathBot(api_key="YOUR_API_KEY")

# Error handler
async def on_error(context: TurnContext, error: Exception):
    await context.send_activity("Sorry, I encountered an error!")
    
ADAPTER.on_turn_error = on_error

# Message handler
async def message_handler(req, res):
    async def callback(turn_context):
        await BOT.on_turn(turn_context)
        
    await ADAPTER.process_activity(req, res, callback)
"""
