# Gibrat's Law Analysis

This project tests Gibrat's Law of Proportionate Effect across different sectors and firm sizes using a comprehensive dataset of firms.

## Overview

Gibrat's Law states that firm growth is a purely random effect and therefore should be independent of firm size. This project tests this hypothesis using both:
- Asset-based size measures
- Sales-based measures

# Project Structure

```
gibrats-law-analysis/
|
├── data/                             # Data directory
│   └── firms.csv                     # Main dataset
|
├── tables/                           # Output tables
│   ├── summary.tex                   # Summary statistics
│   ├── categorical/                  # Categorical analysis tables
│   │   └── *.tex                    # Individual categorical tables
│   ├── gibrat_firm_results.tex      # Firm-level analysis
│   ├── gibrat_sector_size_results.tex    # Sector analysis (size)
│   └── gibrat_sector_sales_results.tex   # Sector analysis (sales)
|
├── src/                             # Source code
│   ├── demo.py                      # Main execution script
│   ├── export.py                    # LaTeX export utilities
│   └── summaryStatistics.py         # Statistical analysis functions
|
├── .gitignore                       # Git ignore file
└── README.md                        # Project documentation
```

## Code Components

- `demo.py`: Main script that runs the Gibrat's Law analysis
- `export.py`: Functions for exporting results to LaTeX tables
- `summaryStatistics.py`: Functions for data cleaning and statistical analysis

## Data Description

The dataset (`firms.csv`) contains the following key variables:
- Size: Firm size measure based on assets
- Salesn: Sales measure
- NACE_Rev_2_main_section: Industry sector classification
- Other financial and operational metrics

## Methodology

1. **Data Preprocessing**
   - Filters firms with multiple observations
   - Removes sectors with <4% representation
   - Handles missing values and outliers

2. **Statistical Analysis**
   - Tests Gibrat's Law at firm level
   - Tests Gibrat's Law by sector
   - Includes time fixed effects
   - Uses both size and sales measures

3. **Output Generation**
   - Creates STATA-style tables with significance stars
   - Generates LaTeX tables for results
   - Produces summary statistics

## Requirements

```python
pandas
numpy
scipy
statsmodels
