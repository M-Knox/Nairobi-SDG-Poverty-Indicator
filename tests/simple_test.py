"""
Simple test for PovertyMapper class
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Import the PovertyMapper class
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from poverty_mapper import PovertyMapper

class TestPovertyMapper(unittest.TestCase):
    """Test cases for PovertyMapper class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample subcounty data for testing
        self.sample_data = pd.DataFrame({
            'Administrative_Unit': ['Westlands', 'Kibera', 'Karen', 'Mathare', 'Kilimani'],
            'Total_Population': [50000, 120000, 30000, 80000, 45000],
            'Total_Households': [15625, 20690, 10345, 13115, 12857],
            'Land_Area_SqKm': [12.5, 5.6, 112.1, 3.1, 18.2],
            'Male_Population': [24500, 58800, 14700, 39200, 22050],
            'Female_Population': [25500, 61200, 15300, 40800, 22950],
            'Group_Quarters': [500, 1200, 200, 800, 450]
        })
        
        # Initialize the PovertyMapper
        self.poverty_mapper = PovertyMapper()
    
    def test_calculate_subcounty_poverty_indicators(self):
        """Test the main poverty indicators calculation."""
        # Calculate poverty indicators
        results = self.poverty_mapper.calculate_subcounty_poverty_indicators(
            self.sample_data
        )
        
        # Check required columns in results
        required_columns = [
            'subcounty_name',
            'subcounty_overall_poverty_rate',
            'subcounty_food_poverty_rate',
            'subcounty_hardcore_poverty_rate',
            'estimated_poor_population',
            'sdg1_progress_category'
        ]
        
        for col in required_columns:
            self.assertIn(col, results.columns, f"Missing required column: {col}")
        
        # Check data types
        self.assertIsInstance(results['subcounty_overall_poverty_rate'][0], (int, float, np.number))
        self.assertIsInstance(results['sdg1_progress_category'][0], str)
        
        # Check that poverty rates are within valid range (0-100%)
        self.assertTrue(
            (results['subcounty_overall_poverty_rate'].between(0, 100)).all(),
            "Poverty rates should be between 0 and 100"
        )
    
    def test_household_size_factor(self):
        """Test the household size factor calculation."""
        # Test small household size (1-3)
        small_hh_factor = self.poverty_mapper._get_household_size_factor(2.5)
        self.assertAlmostEqual(small_hh_factor, 0.5, places=1)
        
        # Test medium household size (4-6)
        med_hh_factor = self.poverty_mapper._get_household_size_factor(5.0)
        self.assertAlmostEqual(med_hh_factor, 1.0, places=1)
        
        # Test large household size (7+)
        large_hh_factor = self.poverty_mapper._get_household_size_factor(8.0)
        self.assertAlmostEqual(large_hh_factor, 1.51, places=2)
    
    def test_sdg1_progress_classification(self):
        """Test the SDG1 progress classification."""
        from config import sdg_progress_thresholds
        
        # Test each threshold
        test_cases = [
            (sdg_progress_thresholds['excellent'] - 5, "Excellent Progress"),
            (sdg_progress_thresholds['excellent'] + 1, "Good Progress"),
            (sdg_progress_thresholds['good'] + 1, "Moderate Progress"),
            (sdg_progress_thresholds['moderate'] + 1, "Slow Progress"),
            (sdg_progress_thresholds['slow'] + 1, "Needs Significant Improvement"),
            (sdg_progress_thresholds['significant'] + 1, "Needs Urgent Attention")
        ]
        
        for rate, expected in test_cases:
            with self.subTest(rate=rate, expected=expected):
                result = self.poverty_mapper._classify_sdg1_progress(rate)
                self.assertEqual(result, expected)

def run_tests():
    """Run the tests and return the test result."""
    print("Running PovertyMapper tests...")
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPovertyMapper)
    test_runner = unittest.TextTestRunner(verbosity=2)
    return test_runner.run(test_suite)

if __name__ == "__main__":
    run_tests()
