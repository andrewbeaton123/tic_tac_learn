import unittest
from collections import defaultdict

from tic_tac_learn.agents.monte_carlo_q_learning import (
    merge_q_tables,
    _create_nested_q_table,
)


class TestQTableMerging(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample Q-tables for testing
        self.q1 = defaultdict(_create_nested_q_table)
        self.q2 = defaultdict(_create_nested_q_table)
        self.state = ("state1",)

    def test_merge_empty_returns_defaultdict(self):
        """Test that merging empty tables returns a properly structured defaultdict."""
        merged = merge_q_tables([])
        self.assertIsInstance(merged, defaultdict)
        self.assertIsInstance(merged[("any_state",)], defaultdict)
        self.assertEqual(merged[("any_state",)][0], 0.0)

    def test_merge_max_strategy_merges_correctly(self):
        """Test max strategy merging takes highest Q-values."""
        # Setup test data
        self.q1[self.state][1] = 0.5
        self.q1[self.state][2] = 0.2
        self.q2[self.state][1] = 0.7
        self.q2[self.state][3] = 0.4

        merged = merge_q_tables([self.q1, self.q2], merge_strategy="max")

        self.assertAlmostEqual(merged[self.state][1], 0.7)
        self.assertAlmostEqual(merged[self.state][2], 0.2)
        self.assertAlmostEqual(merged[self.state][3], 0.4)

    def test_merge_avg_strategy_averages_present_values_only(self):
        """Test averaging strategy only averages values present in tables."""
        state = ("stateA",)
        self.q1[state][10] = 0.5  # present only in q1
        self.q1[state][20] = 0.2  # present in both
        self.q2[state][20] = 0.4
        self.q2[state][30] = 0.8  # present only in q2

        merged = merge_q_tables([self.q1, self.q2], merge_strategy="avg")

        self.assertAlmostEqual(merged[state][10], 0.5)  # only one value
        self.assertAlmostEqual(merged[state][20], 0.3)  # average of 0.2 and 0.4
        self.assertAlmostEqual(merged[state][30], 0.8)  # only one value

    def test_merge_weighted_avg_behaves_like_avg(self):
        """Test that weighted average strategy behaves same as regular average."""
        state = ("S",)
        self.q1[state][0] = 1.0
        self.q2[state][0] = 3.0

        merged_avg = merge_q_tables([self.q1, self.q2], merge_strategy="avg")
        merged_weighted = merge_q_tables([self.q1, self.q2], merge_strategy="weighted_avg")

        self.assertAlmostEqual(merged_avg[state][0], 2.0)
        self.assertAlmostEqual(merged_weighted[state][0], 2.0)

    def test_invalid_merge_strategy(self):
        """Test that invalid merge strategy raises ValueError."""
        with self.assertRaises(ValueError):
            merge_q_tables([self.q1, self.q2], merge_strategy="invalid")


if __name__ == '__main__':
    unittest.main()