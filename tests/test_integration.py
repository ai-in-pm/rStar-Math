"""
Integration tests for rStar-Math components
"""
import pytest
from unittest.mock import Mock, patch
import torch
import numpy as np

from src.core.mcts import MCTS, MCTSConfig
from src.core.ppm import ProcessPreferenceModel, PPMConfig
from src.models.model_interface import OpenAIModel

class TestProblemSolver:
    @pytest.fixture
    def setup_components(self):
        mcts_config = MCTSConfig(
            exploration_weight=1.0,
            max_simulations=100,
            max_depth=5
        )
        ppm_config = PPMConfig(
            input_dim=768,
            hidden_dim=256,
            learning_rate=0.001,
            batch_size=16
        )
        
        mcts = MCTS(mcts_config)
        ppm = ProcessPreferenceModel(ppm_config)
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content="Test response"))]
            )
            model = OpenAIModel("test-key")
            
        return mcts, ppm, model
    
    def test_solve_simple_problem(self, setup_components):
        mcts, ppm, model = setup_components
        problem = "What is 2 + 2?"
        
        # Mock necessary methods
        mcts.get_possible_actions = Mock(return_value=["Add the numbers"])
        mcts.apply_action = Mock(return_value="4")
        mcts.evaluate_state = Mock(return_value=1.0)
        
        action, trajectory = mcts.search(problem)
        assert action is not None
        assert isinstance(trajectory, list)
        
    def test_evaluate_solution(self, setup_components):
        mcts, ppm, model = setup_components
        problem = "What is 2 + 2?"
        solution_steps = [
            "First, we identify this is an addition problem",
            "Then, we add 2 and 2 together",
            "Therefore, 2 + 2 = 4"
        ]
        
        # Mock embedding generation
        embeddings = torch.randn(len(solution_steps), 768)
        model.embed_text = Mock(side_effect=[e.numpy().tolist() for e in embeddings])
        
        # Evaluate solution using PPM
        for step in solution_steps:
            score = ppm.evaluate_step(step, model)
            assert isinstance(score, float)
            assert 0 <= score <= 1
            
    def test_self_evolution(self, setup_components):
        mcts, ppm, model = setup_components
        problem = "What is 2 + 2?"
        
        # Generate initial solution
        mcts.get_possible_actions = Mock(return_value=["Add the numbers"])
        mcts.apply_action = Mock(return_value="4")
        mcts.evaluate_state = Mock(return_value=1.0)
        
        initial_action, initial_trajectory = mcts.search(problem)
        
        # Evolve solution
        evolved_action, evolved_trajectory = mcts.search(problem)
        
        assert len(evolved_trajectory) > 0
        
        # Compare trajectories
        if len(initial_trajectory) > 0 and len(evolved_trajectory) > 0:
            initial_value = initial_trajectory[-1]["value"]
            evolved_value = evolved_trajectory[-1]["value"]
            assert evolved_value >= initial_value

class TestModelComparison:
    @pytest.fixture
    def setup_comparison(self):
        problem = "What is 2 + 2?"
        solution_steps = [
            "First, we identify this is an addition problem",
            "Then, we add 2 and 2 together",
            "Therefore, 2 + 2 = 4"
        ]
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content="0.8"))]
            )
            openai_model = OpenAIModel("test-key")
            
        return problem, solution_steps, openai_model
    
    def test_model_comparison(self, setup_comparison):
        problem, solution_steps, model = setup_comparison
        
        # Test direct reasoning (System 1)
        direct_response = model.generate_response(problem)
        assert isinstance(direct_response, str)
        
        # Test rStar-Math enhanced reasoning (System 2)
        mcts = MCTS()
        ppm = ProcessPreferenceModel(input_dim=768)
        
        mcts.get_possible_actions = Mock(return_value=solution_steps)
        mcts.apply_action = Mock(return_value="4")
        mcts.evaluate_state = Mock(return_value=1.0)
        
        enhanced_action, enhanced_trajectory = mcts.search(problem)
        assert enhanced_action is not None
        assert len(enhanced_trajectory) > 0
        
        # Compare solutions
        direct_score = model.evaluate_reasoning(problem, [direct_response])
        enhanced_score = model.evaluate_reasoning(problem, solution_steps)
        
        assert isinstance(direct_score, float)
        assert isinstance(enhanced_score, float)
        assert 0 <= direct_score <= 1
        assert 0 <= enhanced_score <= 1
