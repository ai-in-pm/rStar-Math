"""
Gradio integration for rStar-Math
"""
import os
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import ModelFactory

class RStarMathGradio:
    def __init__(self):
        """Initialize rStar-Math components."""
        self.mcts = MCTS.from_config_file('config/default.json')
        self.ppm = ProcessPreferenceModel.from_config_file('config/default.json')
        
        # Initialize models
        self.models = {}
        model_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
            'groq': 'GROQ_API_KEY',
            'gemini': 'GEMINI_API_KEY'
        }
        
        for name, key in model_keys.items():
            api_key = os.getenv(key)
            if api_key:
                self.models[name] = ModelFactory.create_model(
                    name, api_key, 'config/default.json'
                )
    
    def solve_problem(self,
                     problem: str,
                     model_name: str,
                     use_rstar: bool = True,
                     show_visualization: bool = True):
        """Solve math problem and return solution with visualizations."""
        model = self.models[model_name]
        
        if use_rstar:
            action, trajectory = self.mcts.search(problem)
            solution_steps = []
            confidence_scores = []
            
            for step in trajectory:
                confidence = self.ppm.evaluate_step(step['state'], model)
                solution_steps.append(step['state'])
                confidence_scores.append(confidence)
        else:
            solution = model.generate_response(problem)
            confidence = model.evaluate_reasoning(problem, [solution])
            solution_steps = [solution]
            confidence_scores = [confidence]
            
        # Format output
        output = "Solution Steps:\\n"
        for i, (step, conf) in enumerate(zip(solution_steps, confidence_scores), 1):
            output += f"Step {i}: {step}\\nConfidence: {conf:.2f}\\n\\n"
            
        # Create visualization
        if show_visualization and len(confidence_scores) > 1:
            df = pd.DataFrame({
                'Step': range(1, len(confidence_scores) + 1),
                'Confidence': confidence_scores
            })
            fig = px.line(df, x='Step', y='Confidence',
                         title='Solution Confidence Trend')
            
            return output, fig
        
        return output, None

def create_examples():
    """Create example problems for the demo."""
    return [
        ["What is 15 × 27?", "openai", True, True],
        ["Solve for x: 2x + 5 = 13", "openai", True, True],
        ["Find the derivative of f(x) = x² + 3x", "openai", True, True],
        ["Find the area of a circle with radius 5", "openai", True, True]
    ]

def main():
    """Create and launch Gradio interface."""
    rstar = RStarMathGradio()
    
    # Create interface
    with gr.Blocks(title="rStar-Math Demonstrator") as demo:
        gr.Markdown("# rStar-Math Problem Solver")
        
        with gr.Row():
            with gr.Column():
                problem_input = gr.Textbox(
                    label="Enter your math problem",
                    placeholder="e.g., What is 2 + 2?"
                )
                model_select = gr.Dropdown(
                    choices=list(rstar.models.keys()),
                    value=list(rstar.models.keys())[0],
                    label="Select Model"
                )
                use_rstar = gr.Checkbox(
                    label="Use rStar-Math Enhancement",
                    value=True
                )
                show_viz = gr.Checkbox(
                    label="Show Visualization",
                    value=True
                )
                solve_btn = gr.Button("Solve")
            
            with gr.Column():
                solution_output = gr.Textbox(
                    label="Solution",
                    lines=10
                )
                plot_output = gr.Plot(label="Confidence Trend")
        
        # Add examples
        gr.Examples(
            examples=create_examples(),
            inputs=[problem_input, model_select, use_rstar, show_viz]
        )
        
        # Connect components
        solve_btn.click(
            fn=rstar.solve_problem,
            inputs=[problem_input, model_select, use_rstar, show_viz],
            outputs=[solution_output, plot_output]
        )
        
        # Add documentation
        with gr.Accordion("About"):
            gr.Markdown("""
            ## rStar-Math Demonstrator
            
            This interface demonstrates the capabilities of rStar-Math, an AI framework
            that enhances mathematical reasoning using Monte Carlo Tree Search and
            Process Preference Models.
            
            ### Features:
            - Multiple LLM support (OpenAI, Anthropic, Mistral, etc.)
            - Step-by-step solution breakdown
            - Confidence scoring for each step
            - Visual confidence tracking
            
            ### Usage:
            1. Enter your math problem
            2. Select a model
            3. Choose whether to use rStar-Math enhancement
            4. Click "Solve" to see the solution
            """)
    
    # Launch interface
    demo.launch(share=False)

if __name__ == "__main__":
    main()
