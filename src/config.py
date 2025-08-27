"""
Configuration parameters for the Nairobi SDG Indicators project.
Contains constants, file paths, and configuration parameters.
"""

from pathlib import Path

# Project root directory
project_dir = Path(__file__).parent.parent

# Data directories
data_dir = project_dir / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"
clean_data_dir = data_dir / "clean"

# Output directories
outputs_dir = project_dir / "outputs"
reports_dir = outputs_dir / "reports"
figures_dir = outputs_dir / "figures"

# File paths
raw_census_file = raw_data_dir / "2019-population_census.xlsx"
processeed_data_file = processed_data_dir / "nairobi_census_2019.xlsx"
clean_data_file = clean_data_dir / "nairobi_census_2019_cleaned.csv"

# Nairobi County Poverty Rates (as percentages)
# Figures obtained from the Kenya Poverty Report 2019
poverty_rates = {
    "overall_poverty": 10.2,  # Overall poverty rate for Nairobi County
    "food_poverty": 12.7,      # Food poverty rate
    "hardcore_poverty": 8.3   # Hardcore poverty rate
}

# Demographic adjustment factors
# Demographic adjustment factors based on Kenya 2022 KCHS poverty data
# These factors adjust poverty rates based on household characteristics
# Base poverty rate: 39.8% nationally (2022)

demographic_factors = {
    "household_size": {
        "1-3": 0.50,      # 20% poverty rate (from document: "low of 20 per cent")
        "4-6": 1.0,       # Baseline - around national average
        "7+": 1.51        # 60% poverty rate (from document: "60 per cent among households with 7 or more members")
    },
    
    "household_head_gender": {
        "male": 0.82,     # 32.6% poverty rate
        "female": 0.89    # 35.3% poverty rate  
    },
    
    "household_head_age": {
        "15-19": 0.24,    # Estimated from urban youth data (9.6% in urban areas)
        "20-39": 0.85,    # Estimated baseline for working age
        "40-59": 1.0,     # Baseline
        "60-69": 1.1,     # Slightly elevated
        "70+": 1.16       # 46% in rural areas (highest mentioned)
    },
    
    "children_in_household": {
        "no_children": 0.60,   # 24% poverty rate
        "with_children": 0.95  # 38% poverty rate
    },
    
    "marital_status_head": {
        "monogamous": 0.83,    # 33% poverty rate
        "polygamous": 1.21     # 48% poverty rate (1.5x more likely to be poor)
    },
    
}


# Administrative levels for analysis
admin_levels = {
    "subcounty": "Sub-County",
}

# SDG Indicator Configuration
sdg_indicators = {
    "1.1.1": "Proportion of population below the international poverty line",
    "1.2.1": "Proportion of population living below the national poverty line",
    "1.2.2": "Poverty gap ratio"
}

# SDG Progress Thresholds
sdg_progress_thresholds = {
    "excellent": 5,
    "good": 15,
    "moderate": 25,
    "slow": 35,
    "significant": 40
} 

# Validation thresholds
validation_thresholds = {
    "max_household_size": 20,
    "min_population": 100,
    "max_poverty_rate": 100.0,
    "min_poverty_rate": 0.0
}

# Ensure required directories exist
for directory in [raw_data_dir, processed_data_dir, clean_data_dir, 
                 reports_dir, figures_dir]:
    directory.mkdir(parents=True, exist_ok=True)
    