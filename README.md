# Nairobi SDG Indicators Analysis

A comprehensive analysis tool for mapping Sustainable Development Goal (SDG) poverty indicators across Nairobi County using 2019 census data and demographic proxies.

## Project Overview

This project analyzes poverty indicators at the subcounty level in Nairobi County, Kenya, using:
- 2019 Population and Housing Census data
- Demographic proxies for poverty estimation
- SDG 1 (No Poverty) progress classification
- Automated report generation

## Features

- **PovertyMapper Class**: Core functionality for calculating subcounty-level poverty indicators
- **Census Data Integration**: Processes raw census data into analytical format
- **SDG Progress Classification**: Categorizes areas based on poverty reduction progress
- **Automated Reporting**: Generates comprehensive Word documents with findings
- **Validation Framework**: Built-in validation for poverty estimates
- **Test Suite**: Comprehensive unit tests for reliability

## Project Structure

```
Nairobi SDG Indicators/
├── src/                          # Source code
│   ├── config.py                 # Configuration and constants
│   ├── poverty_mapper.py         # Main PovertyMapper class
│   └── summary_report.py         # Report generation utilities
├── notebooks/                    # Jupyter notebooks
│   ├── 01-sdg-indicator.ipynb    # SDG analysis workflow
│   └── 2019-census_data-cleaning.ipynb  # Data cleaning pipeline
├── data/                         # Data directory
│   ├── raw/                      # Original census data
│   ├── clean/                    # Processed census data
│   └── processed/                # Analysis results
├── outputs/                      # Generated outputs
│   ├── figures/                  # Charts and visualizations
│   └── reports/                  # Generated reports
├── tests/                        # Test suite
│   └── simple_test.py            # Unit tests for PovertyMapper
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Nairobi SDG Indicators"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

1. **Run the poverty analysis:**
   ```python
   from src.poverty_mapper import PovertyMapper
   import pandas as pd
   
   # Initialize the mapper
   poverty_mapper = PovertyMapper()
   
   # Load your census data
   census_data = pd.read_csv('data/clean/nairobi_census_2019_cleaned.csv')
   subcounty_data = census_data[census_data['Administrative_level'] == 'Sub-County']
   
   # Calculate poverty indicators
   results = poverty_mapper.calculate_subcounty_poverty_indicators(subcounty_data)
   ```

2. **Generate reports:**
   ```python
   from src.summary_report import create_complete_poverty_report
   from src.config import reports_dir
   
   # Generate comprehensive report
   report_file = create_complete_poverty_report(results, reports_dir)
   ```

### Running Tests

```bash
python tests/simple_test.py
```

### Using Jupyter Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open analysis notebooks:**
   - `notebooks/01-sdg-indicator.ipynb` - Main SDG analysis
   - `notebooks/2019-census_data-cleaning.ipynb` - Data preprocessing

## Key Components

### PovertyMapper Class

The core class that calculates poverty indicators using:
- Household size adjustments
- Population density factors
- Gender ratio considerations
- Institutional population adjustments

### Configuration

All parameters are centralized in `src/config.py`:
- Poverty rate baselines
- Demographic adjustment factors
- SDG progress thresholds
- File paths and directories

### Data Requirements

Input data should include:
- `Administrative_Unit`: Area names
- `Total_Population`: Population count
- `Total_Households`: Household count
- `Land_Area_SqKm`: Area in square kilometers
- `Male_Population`, `Female_Population`: Gender breakdown
- `Group_Quarters`: Institutional population

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t nairobi-sdg-indicators .

# Run the container
docker run -p 8888:8888 nairobi-sdg-indicators
```

### Using Docker Compose

```bash
# Start the development environment
docker-compose up -d

# Access Jupyter at http://localhost:8888
```

## Output

The analysis generates:
- **Poverty rates** by subcounty (overall, food, hardcore)
- **Population estimates** of people in poverty
- **SDG progress classifications** for each area
- **Validation metrics** for quality assurance
- **Comprehensive reports** in Word format

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Testing

Run the test suite to ensure functionality:
```bash
python -m pytest tests/ -v
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kenya National Bureau of Statistics for 2019 Census data
- UN Sustainable Development Goals framework
- Kenya County Health Survey (KCHS) for demographic factors

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This tool is designed for research and policy analysis purposes. Results should be validated against official statistics and used in conjunction with other poverty measurement approaches.
