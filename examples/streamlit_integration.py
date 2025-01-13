"""
Streamlit integration for rStar-Math
"""
import os
import streamlit as st
import plotly.express as px
import pandas as pd
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import ModelFactory

def initialize_components():
    """Initialize rStar-Math components."""
    st.session_state.mcts = MCTS.from_config_file('config/default.json')
    st.session_state.ppm = ProcessPreferenceModel.from_config_file('config/default.json')
    
    # Initialize models
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'mistral': os.getenv('MISTRAL_API_KEY'),
        'groq': os.getenv('GROQ_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY')
    }
    
    st.session_state.models = {}
    for name, key in api_keys.items():
        if key:
            st.session_state.models[name] = ModelFactory.create_model(
                name, key, 'config/default.json'
            )

def solve_problem(problem: str, model_name: str, use_rstar: bool = True):
    """Solve math problem with selected model."""
    model = st.session_state.models[model_name]
    
    if use_rstar:
        action, trajectory = st.session_state.mcts.search(problem)
        solution_steps = []
        confidence_scores = []
        
        for step in trajectory:
            confidence = st.session_state.ppm.evaluate_step(step['state'], model)
            solution_steps.append(step['state'])
            confidence_scores.append(confidence)
            
        return solution_steps, confidence_scores
    else:
        solution = model.generate_response(problem)
        confidence = model.evaluate_reasoning(problem, [solution])
        return [solution], [confidence]

def main():
    st.title("rStar-Math Demonstrator")
    
    # Initialize components if not done
    if 'mcts' not in st.session_state:
        initialize_components()
    
    # Sidebar settings
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(st.session_state.models.keys())
    )
    
    use_rstar = st.sidebar.checkbox("Use rStar-Math", value=True)
    
    # Main interface
    st.header("Math Problem Solver")
    
    # Input section
    problem = st.text_area("Enter your math problem:")
    
    if st.button("Solve"):
        if problem:
            with st.spinner("Solving..."):
                solution_steps, confidence_scores = solve_problem(
                    problem, selected_model, use_rstar
                )
                
                # Display solution
                st.header("Solution")
                for i, (step, confidence) in enumerate(zip(solution_steps, confidence_scores), 1):
                    st.markdown(f"**Step {i}:** {step}")
                    st.progress(confidence)
                    st.markdown(f"Confidence: {confidence:.2f}")
                
                # Plot confidence trend
                if len(confidence_scores) > 1:
                    df = pd.DataFrame({
                        'Step': range(1, len(confidence_scores) + 1),
                        'Confidence': confidence_scores
                    })
                    fig = px.line(df, x='Step', y='Confidence',
                                title='Solution Confidence Trend')
                    st.plotly_chart(fig)
                
                # Overall statistics
                st.header("Solution Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Steps", len(solution_steps))
                col2.metric("Average Confidence", f"{sum(confidence_scores)/len(confidence_scores):.2f}")
                col3.metric("Min Confidence", f"{min(confidence_scores):.2f}")
        else:
            st.error("Please enter a math problem")
    
    # Example problems
    st.header("Example Problems")
    examples = {
        "Basic Arithmetic": "What is 15 × 27?",
        "Algebra": "Solve for x: 2x + 5 = 13",
        "Calculus": "Find the derivative of f(x) = x² + 3x",
        "Geometry": "Find the area of a circle with radius 5",
        "Word Problem": "If a train travels 60 mph for 2.5 hours, how far does it go?"
    }
    
    if st.button("Try an Example"):
        example = st.selectbox("Select an example:", list(examples.keys()))
        st.text_area("Problem", examples[example], key="example_problem")

if __name__ == "__main__":
    main()
