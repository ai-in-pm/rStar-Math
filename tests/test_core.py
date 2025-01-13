"""
Tests for core rStar-Math functionality
"""
import pytest
from src.core.mcts import MCTS, MCTSNode
from src.core.ppm import ProcessPreferenceModel

def test_mcts_node_creation():
    node = MCTSNode(state="initial")
    assert node.state == "initial"
    assert node.parent is None
    assert len(node.children) == 0
    assert node.visits == 0
    assert node.value == 0.0

def test_mcts_selection():
    mcts = MCTS(exploration_weight=1.0)
    root = MCTSNode(state="root")
    child = root.add_child(action="test", state="child")
    
    # Update values
    root.visits = 10
    child.visits = 5
    child.value = 2.5
    
    selected = mcts.select_action(root)
    assert selected == child

def test_process_preference_model():
    model = ProcessPreferenceModel(input_dim=10)
    # Add more specific tests based on implementation
    pass

def test_math_problem_solving():
    """Integration test for solving a simple math problem."""
    problem = "What is 2 + 2?"
    mcts = MCTS()
    ppm = ProcessPreferenceModel(input_dim=10)
    
    # Test implementation to be added
    pass

if __name__ == "__main__":
    pytest.main([__file__])
