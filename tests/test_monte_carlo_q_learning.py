
import pytest
import os
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict

from tic_tac_learn.agents.monte_carlo_q_learning import merge_q_tables, _create_nested_q_table

@pytest.fixture
def empty_q_table():
    """Returns an empty Q-table"""
    return defaultdict(_create_nested_q_table)

@pytest.fixture
def sample_q_tables():
    """Returns a pair of Q-tables with sample data"""
    q1 = defaultdict(_create_nested_q_table)
    q2 = defaultdict(_create_nested_q_table)
    
    # Set up first Q-table
    state1 = ("state1",)
    q1[state1][1] = 0.5
    q1[state1][2] = 0.2
    
    # Set up second Q-table
    q2[state1][1] = 0.7
    q2[state1][3] = 0.4
    
    return q1, q2

class TestQTableMerging:
    """Test suite for Q-table merging functionality"""
    
    def test_empty_merge_returns_valid_defaultdict(self, empty_q_table):
        """Test that merging empty tables returns a properly structured defaultdict"""
        merged = merge_q_tables([])
        
        assert isinstance(merged, defaultdict)
        assert isinstance(merged[("test_state",)], defaultdict)
        assert merged[("test_state",)][0] == 0.0

    def test_max_strategy_merging(self, sample_q_tables):
        """Test max strategy merging takes highest Q-values"""
        q1, q2 = sample_q_tables
        state = ("state1",)
        
        merged = merge_q_tables([q1, q2], merge_strategy="max")
        
        assert merged[state][1] == pytest.approx(0.7)  # Should take higher value
        assert merged[state][2] == pytest.approx(0.2)  # Should keep unique value
        assert merged[state][3] == pytest.approx(0.4)  # Should keep unique value

    @pytest.mark.parametrize("strategy", ["avg", "weighted_avg"])
    def test_averaging_strategies(self, empty_q_table, strategy):
        """Test both averaging strategies produce expected results"""
        q1, q2 = empty_q_table, empty_q_table
        state = ("state_avg",)
        
        # Setup test data
        q1[state][1] = 1.0
        q2[state][1] = 3.0
        
        merged = merge_q_tables([q1, q2], merge_strategy=strategy)
        
        assert merged[state][1] == pytest.approx(2.0)

    def test_partial_state_action_coverage(self, empty_q_table):
        """Test merging when state-action pairs aren't present in all tables"""
        q1, q2 = empty_q_table, empty_q_table
        state = ("partial_state",)
        
        # Setup asymmetric test data
        q1[state][1] = 0.5  # Only in q1
        q1[state][2] = 0.2  # In both
        q2[state][2] = 0.4  # In both
        q2[state][3] = 0.8  # Only in q2
        
        merged = merge_q_tables([q1, q2], merge_strategy="avg")
        
        assert merged[state][1] == pytest.approx(0.5)  # Single value
        assert merged[state][2] == pytest.approx(0.3)  # Average of two values
        assert merged[state][3] == pytest.approx(0.8)  # Single value

    @pytest.mark.parametrize("invalid_strategy", ["invalid", "mean", None])
    def test_invalid_merge_strategy(self, sample_q_tables, invalid_strategy):
        """Test handling of invalid merge strategies"""
        q1, q2 = sample_q_tables
        
        with pytest.raises(ValueError):
            merge_q_tables([q1, q2], merge_strategy=invalid_strategy)