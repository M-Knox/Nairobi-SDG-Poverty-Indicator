import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import demographic_factors, poverty_rates, sdg_progress_thresholds, clean_data_file


class PovertyMapper:
    """
    Map SDG 1 poverty indicators at subcounty level using KCHS 2022 demographic proxies
    and the 2019 population census data
    """
    
    def __init__(self, nairobi_county_poverty_rate=None, 
                 nairobi_food_poverty_rate=None,
                 nairobi_hardcore_poverty_rate=None):
        
        # County-level baseline rates from config
        self.county_overall_poverty = nairobi_county_poverty_rate or poverty_rates['overall_poverty']
        self.county_food_poverty = nairobi_food_poverty_rate or poverty_rates['food_poverty']
        self.county_hardcore_poverty = nairobi_hardcore_poverty_rate or poverty_rates['hardcore_poverty']
        
        # Demographic adjustment factors from config
        self.demographic_factors = {
            'household_size_adjustment': {
                'small_hh_1_3': demographic_factors['household_size']['1-3'],
                'medium_hh_4_6': demographic_factors['household_size']['4-6'],
                'large_hh_7plus': demographic_factors['household_size']['7+']
            },
            'gender_head_adjustment': {
                'male_headed': demographic_factors['household_head_gender']['male'],
                'female_headed': demographic_factors['household_head_gender']['female']
            },
            'children_adjustment': {
                'no_children': demographic_factors['children_in_household']['no_children'],
                'with_children': demographic_factors['children_in_household']['with_children']
            },
            'age_head_adjustment': {
                'young_20_39': demographic_factors['household_head_age']['20-39'],
                'adult_40_59': demographic_factors['household_head_age']['40-59'],
                'elderly_70plus': demographic_factors['household_head_age']['70+']
            }
        }
    
    def calculate_subcounty_poverty_indicators(self, subcounty_demographics_df):
        """
        Calculate subcounty-level poverty indicators using only available census data
        
        Parameters:
        subcounty_demographics_df: DataFrame with columns:
        - Administrative_Unit (or subcounty_name), Total_Population, Total_Households, 
        - Land_Area_SqKm, Male_Population, Female_Population, Group_Quarters
        """
        
        # Input validation - check for census column names
        census_columns = ['Administrative_Unit', 'Total_Population', 'Total_Households', 'Land_Area_SqKm']
        alt_columns = ['subcounty_name', 'Total_Population', 'Total_Households', 'Land_Area_SqKm']
        
        # Check if using census naming or alternative naming
        if all(col in subcounty_demographics_df.columns for col in census_columns):
            name_col = 'Administrative_Unit'
        elif all(col in subcounty_demographics_df.columns for col in alt_columns):
            name_col = 'subcounty_name'
        else:
            missing = [col for col in census_columns if col not in subcounty_demographics_df.columns]
            raise ValueError(f"Missing required columns. Need either {census_columns} or {alt_columns}. Missing: {missing}")
        
        results_df = subcounty_demographics_df.copy()
        
        # Standardize column naming if needed
        if name_col == 'Administrative_Unit':
            results_df['subcounty_name'] = results_df['Administrative_Unit']
        
        # Calculate derived indicators from available census data only
        results_df['avg_household_size'] = results_df['Total_Population'] / results_df['Total_Households']
        results_df['population_density'] = results_df['Total_Population'] / results_df['Land_Area_SqKm']
        
        # Calculate gender ratio if gender columns available
        if 'Male_Population' in results_df.columns and 'Female_Population' in results_df.columns:
            results_df['gender_ratio'] = results_df['Male_Population'] / results_df['Female_Population']
            # Estimate female-headed household proportion from gender imbalance (rough proxy)
            # Areas with more females might have more female-headed households
            results_df['gender_imbalance_factor'] = np.where(
                results_df['gender_ratio'] < 0.9, 1.1,  # More females = slight increase in poverty risk
                np.where(results_df['gender_ratio'] > 1.1, 0.95, 1.0)  # More males = slight decrease
            )
        else:
            results_df['gender_imbalance_factor'] = 1.0
        
        # Calculate demographic adjustments using only available data
        # 1. Household size adjustment (using your empirical factors)
        results_df['size_adjustment'] = results_df['avg_household_size'].apply(
            lambda x: self._get_household_size_factor(x) - 1.0  # Convert to deviation
        )
        
        # 2. Population density adjustment (high density often correlates with poverty in urban areas)
        results_df['density_adjustment'] = self._get_density_adjustment(results_df['population_density'])
        
        # 3. Gender imbalance adjustment (if available)
        results_df['gender_adjustment'] = results_df['gender_imbalance_factor'] - 1.0
        
        # 4. Group quarters adjustment (if available) - institutional population proxy
        if 'Group_Quarters' in results_df.columns:
            # Convert to numeric, replacing errors with NaN
            results_df['Group_Quarters'] = pd.to_numeric(results_df['Group_Quarters'], errors='coerce')
            results_df['group_quarters_ratio'] = results_df['Group_Quarters'] / results_df['Total_Population']

            # High group quarters might indicate different socioeconomic conditions
            results_df['institutional_adjustment'] = np.where(
                results_df['group_quarters_ratio'] > 0.05, 0.1,  # 10% increase if >5% in group quarters
                0.0
            )
        else:
            results_df['institutional_adjustment'] = 0.0
        
        # Sum all available adjustments (they're now deviations from baseline)
        results_df['total_adjustment'] = (
            results_df['size_adjustment'] + 
            results_df['density_adjustment'] +
            results_df['gender_adjustment'] +
            results_df['institutional_adjustment']
        )
        
        # Apply total adjustment to baseline poverty rates
        results_df['subcounty_overall_poverty_rate'] = (
            self.county_overall_poverty * (1 + results_df['total_adjustment'])
        ).clip(0, 100)
        
        results_df['subcounty_food_poverty_rate'] = (
            self.county_food_poverty * (1 + results_df['total_adjustment'])
        ).clip(0, 100)
        
        results_df['subcounty_hardcore_poverty_rate'] = (
            self.county_hardcore_poverty * (1 + results_df['total_adjustment'])
        ).clip(0, 100)
        
        # Calculate absolute numbers
        results_df['estimated_poor_population'] = (
            results_df['Total_Population'] * 
            results_df['subcounty_overall_poverty_rate'] / 100
        ).round()
        
        results_df['estimated_food_poor_population'] = (
            results_df['Total_Population'] * 
            results_df['subcounty_food_poverty_rate'] / 100
        ).round()
        
        # SDG 1 Classification
        results_df['sdg1_progress_category'] = results_df['subcounty_overall_poverty_rate'].apply(
            self._classify_sdg1_progress
        )
        
        return results_df
    
    def _get_household_size_factor(self, avg_size):
        """Get household size adjustment factor with input validation"""
        if pd.isna(avg_size) or avg_size <= 0:
            return 1.0  # Default to baseline if invalid
            
        if avg_size <= 3:
            return self.demographic_factors['household_size_adjustment']['small_hh_1_3']
        elif avg_size <= 6:
            return self.demographic_factors['household_size_adjustment']['medium_hh_4_6']
        else:
            return self.demographic_factors['household_size_adjustment']['large_hh_7plus']
    
    def _get_age_head_factor(self, pct_elderly, pct_young=0.15):
        """Get age of household head adjustment factor with validation
        
        Args:
            pct_elderly: Proportion of elderly-headed households
            pct_young: Proportion of young-headed households
        """
        # Validate inputs
        if pd.isna(pct_elderly):
            pct_elderly = 0.1  # Default estimate
        if pd.isna(pct_young):
            pct_young = 0.15
            
        # Ensure proportions don't exceed 1.0
        if pct_elderly + pct_young > 1.0:
            # Normalize proportions
            total = pct_elderly + pct_young
            pct_elderly = pct_elderly / total
            pct_young = pct_young / total
        
        # Calculate proportion for middle age group
        pct_adult = max(0, 1 - pct_elderly - pct_young)
        
        # Weighted average across all age categories
        return (
            pct_young * self.demographic_factors['age_head_adjustment']['young_20_39'] +
            pct_adult * self.demographic_factors['age_head_adjustment']['adult_40_59'] +
            pct_elderly * self.demographic_factors['age_head_adjustment']['elderly_70plus']
        )
    
    def _get_density_adjustment(self, population_density):
        """Get population density adjustment factor based on census data"""
        # Calculate density percentiles for urban poverty correlation
        density_75th = np.percentile(population_density, 75)
        density_90th = np.percentile(population_density, 90)
        
        # Higher density areas often correlate with informal settlements/higher poverty in Nairobi
        adjustment = np.where(
            population_density > density_90th, 0.2,    # 20% increase for very high density
            np.where(
                population_density > density_75th, 0.1,  # 10% increase for high density
                0.0  # No adjustment for lower density areas
            )
        )
        
        return adjustment
    
    def _classify_sdg1_progress(self, poverty_rate):
        """Classify SDG 1 progress based on poverty rate using urban Nairobi context"""
        if pd.isna(poverty_rate):
            return "Data Unavailable"
            
        if poverty_rate < sdg_progress_thresholds['excellent']:
            return "Excellent Progress"
        elif poverty_rate < sdg_progress_thresholds['good']:
            return "Good Progress" 
        elif poverty_rate < sdg_progress_thresholds['moderate']:
            return "Moderate Progress"
        elif poverty_rate < sdg_progress_thresholds['slow']:
            return "Slow Progress"
        elif poverty_rate < sdg_progress_thresholds['significant']:
            return "Needs Significant Improvement"
        else:
            return "Needs Urgent Attention"

def create_demographic_poverty_proxies(subcounty_demographics_df):
    """
    Create poverty proxies using demographic characteristics from census data
    """
    
    # Input validation
    required_cols = ['Total_Population', 'Total_Households', 'Land_Area_SqKm']
    missing_cols = [col for col in required_cols if col not in subcounty_demographics_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate derived indicators from census columns
    df = subcounty_demographics_df.copy()
    
    # Population density proxy (higher density often correlates with informal settlements/poverty in Nairobi)
    df['population_density'] = df['Total_Population'] / df['Land_Area_SqKm']
    
    # Household size proxy (larger households often indicate economic stress)
    df['avg_household_size'] = df['Total_Population'] / df['Total_Households']
    
    # Gender ratio (areas with skewed ratios might indicate economic migration patterns)
    if 'Male_Population' in df.columns and 'Female_Population' in df.columns:
        df['male_female_ratio'] = df['Male_Population'] / df['Female_Population']
        # Convert ratio to deviation from 1.0 (balanced ratio)
        df['gender_imbalance'] = np.abs(df['male_female_ratio'] - 1.0)
    else:
        df['gender_imbalance'] = 0
    
    # Group quarters proportion (institutional population vs household population)
    if 'Group_Quarters' in df.columns:
        df['group_quarters_ratio'] = df['Group_Quarters'] / df['Total_Population']
    else:
        df['group_quarters_ratio'] = 0
    
    # Demographic poverty proxy weights (empirically justified in config)
    demographic_weights = {
        'population_density': 0.4,      # Higher density = higher poverty likelihood
        'avg_household_size': 0.3,      # Larger households = higher poverty likelihood  
        'gender_imbalance': 0.2,        # Gender imbalance might indicate economic stress
        'group_quarters_ratio': 0.1     # Higher institutional population might indicate specific conditions
    }
    
    # Normalize each indicator first (to handle different scales) - FIXED: Using MinMaxScaler
    normalized_indicators = {}
    scaler = MinMaxScaler()
    
    for indicator in demographic_weights.keys():
        if indicator in df.columns:
            values = df[indicator].values.reshape(-1, 1)
            normalized_values = scaler.fit_transform(values).flatten()
            normalized_indicators[indicator] = normalized_values
        else:
            normalized_indicators[indicator] = np.zeros(len(df))
    
    # Calculate weighted demographic poverty proxy
    demographic_proxy = np.zeros(len(df))
    for indicator, weight in demographic_weights.items():
        demographic_proxy += normalized_indicators[indicator] * weight
    
    # Final 0-1 normalization
    final_scaler = MinMaxScaler()
    demographic_proxy_normalized = final_scaler.fit_transform(demographic_proxy.reshape(-1, 1)).flatten()
    
    return demographic_proxy_normalized

def create_census_based_poverty_adjustments(subcounty_demographics_df):
    """
    Create poverty adjustments using census data and empirical demographic factors
    """
    df = subcounty_demographics_df.copy()
    
    # Input validation
    if 'Total_Population' not in df.columns or 'Total_Households' not in df.columns:
        raise ValueError("Required columns missing: Total_Population, Total_Households")
    
    # Calculate household size factor using empirically-based thresholds
    df['household_size_factor'] = df['Total_Population'] / df['Total_Households']
    df['size_adjustment'] = df['household_size_factor'].apply(
        lambda x: 0.50 if x <= 3 else (1.0 if x <= 6 else 1.51)  # Using your empirical factors
    )
    
    # Population density adjustment (calibrated against urban poverty patterns)
    if 'Land_Area_SqKm' in df.columns:
        density = df['Total_Population'] / df['Land_Area_SqKm']
        density_75th = np.percentile(density, 75)
        density_90th = np.percentile(density, 90)
        
        df['density_adjustment'] = np.where(
            density > density_90th, 1.20,  # 20% increase for very high density
            np.where(
                density > density_75th, 1.10,  # 10% increase for high density  
                1.0  # Baseline for lower density
            )
        )
    else:
        df['density_adjustment'] = 1.0
    
    # Combine adjustments additively (convert to deviations first)
    df['total_demographic_adjustment'] = (
        (df['size_adjustment'] - 1.0) + 
        (df['density_adjustment'] - 1.0)
    )
    
    return df['total_demographic_adjustment']

def validate_poverty_estimates(subcounty_results, validation_data=None):
    """
    Validate subcounty-level poverty estimates against known benchmarks
    """
    from config import poverty_rates
    
    validation_results = {}
    
    # Check if total poor population aligns with county totals
    total_estimated_poor = subcounty_results['estimated_poor_population'].sum()
    nairobi_total_population = subcounty_results['Total_Population'].sum()
    estimated_county_rate = (total_estimated_poor / nairobi_total_population) * 100
    
    known_county_rate = poverty_rates['overall_poverty']
    validation_results['county_rate_check'] = {
        'estimated_rate': estimated_county_rate,
        'known_county_rate': known_county_rate,
        'difference': abs(estimated_county_rate - known_county_rate)
    }
    
    # Spatial validation - check for reasonable variation
    poverty_variation = subcounty_results['subcounty_overall_poverty_rate'].std()
    validation_results['spatial_variation'] = {
        'std_deviation': poverty_variation,
        'range': (subcounty_results['subcounty_overall_poverty_rate'].min(), 
                 subcounty_results['subcounty_overall_poverty_rate'].max()),
        'coefficient_of_variation': poverty_variation / subcounty_results['subcounty_overall_poverty_rate'].mean(),
        'reasonable': 0.1 < (poverty_variation / subcounty_results['subcounty_overall_poverty_rate'].mean()) < 0.4
    }
    
    # Data quality checks
    validation_results['data_quality'] = {
        'no_missing_values': subcounty_results['subcounty_overall_poverty_rate'].notna().all(),
        'within_bounds': ((subcounty_results['subcounty_overall_poverty_rate'] >= 0) & 
                         (subcounty_results['subcounty_overall_poverty_rate'] <= 100)).all(),
        'reasonable_range': subcounty_results['subcounty_overall_poverty_rate'].between(5, 80).all()
    }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    
    # Initialize poverty mapper
    poverty_mapper = PovertyMapper()
    
    # Example subcounty demographics data (using actual census column names)
    sample_subcounty_data = pd.DataFrame({
        'Administrative_Unit': ['Westlands', 'Kibera', 'Karen', 'Mathare', 'Kilimani'],
        'Total_Population': [50000, 120000, 30000, 80000, 45000],
        'Male_Population': [24500, 58800, 14700, 39200, 22050],
        'Female_Population': [25500, 61200, 15300, 40800, 22950],
        'Total_Households': [15625, 20690, 10345, 13115, 12857],
        'Group_Quarters': [500, 1200, 200, 800, 450],
        'Land_Area_SqKm': [12.5, 5.6, 112.1, 3.1, 18.2],
        'Population_Density': [4000, 21429, 268, 25806, 2473]  # Pre-calculated for reference
    })
    
    # Calculate subcounty-level poverty indicators
    subcounty_poverty_results = poverty_mapper.calculate_subcounty_poverty_indicators(sample_subcounty_data)
    
    # Create demographic poverty proxies
    demographic_proxies = create_demographic_poverty_proxies(sample_subcounty_data)
    subcounty_poverty_results['demographic_poverty_proxy'] = demographic_proxies
    
    # Display results
    print("Subcounty-Level SDG 1 Poverty Indicators:")
    print("=" * 50)
    for _, row in subcounty_poverty_results.iterrows():
        print(f"\n{row['subcounty_name']}:")
        print(f"  Overall Poverty Rate: {row['subcounty_overall_poverty_rate']:.1f}%")
        print(f"  Food Poverty Rate: {row['subcounty_food_poverty_rate']:.1f}%")
        print(f"  Estimated Poor Population: {row['estimated_poor_population']:,.0f}")
        print(f"  Population Density: {row['population_density']:.1f} per sq km")
        print(f"  Demographic Poverty Proxy: {row['demographic_poverty_proxy']:.3f}")
        print(f"  SDG 1 Progress: {row['sdg1_progress_category']}")
    
    # Validate results
    validation = validate_poverty_estimates(subcounty_poverty_results)
    print(f"\nValidation Results:")
    print(f"County Rate Check: {validation['county_rate_check']['difference']:.1f}% difference from known rate")
    print(f"Spatial Variation: CV = {validation['spatial_variation']['coefficient_of_variation']:.2f} (reasonable: {validation['spatial_variation']['reasonable']})")
    print(f"Data Quality: All checks passed = {all(validation['data_quality'].values())}")