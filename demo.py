import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

import summaryStatistics as sum_stats
import export as exp

def test_gibrats_law(data, size_column, id_column='id', year_column='year', min_obs=30):
    """
    Test Gibrat's Law: ln(Sijt) = αj0 + αj1ln(Sijt-1) + θ'jkTt + εjt

    Parameters:
    -----------
    data : pandas DataFrame
        Must contain columns for firm size, firm ID, and year
    size_column : str
        Name of the column containing firm size measure
    id_column : str
        Name of the column containing firm identifiers
    year_column : str
        Name of the column containing year information
    min_obs : int
        Minimum number of observations required for testing

    Returns:
    --------
    dict
        Dictionary containing test results
    """
    try:
        # Create working copy
        df = data.copy()
        print(f"Initial observations: {len(df)}")
        # 1. Data Cleaning
        # Convert size to numeric and handle errors
        df[size_column] = pd.to_numeric(df[size_column], errors='coerce')
        df = df[df[size_column].notna()]
        df = df[df[size_column] > 0]
        print(f"After removing invalid size values: {len(df)}")
        # Take natural log of size
        df['ln_size'] = np.log(df[size_column])
        # Sort and create lag
        df = df.sort_values([id_column, year_column])
        df['ln_size_lag'] = df.groupby(id_column)['ln_size'].shift(1)
        # Clean data
        df_clean = df.dropna(subset=['ln_size', 'ln_size_lag'])
        print(f"After creating lags and removing NaN: {len(df_clean)}")
        # Create year dummies
        year_dummies = pd.get_dummies(df_clean[year_column], prefix='year', dtype=float)
        if len(df_clean) < min_obs:
            return {'error': f'Insufficient observations. Need {min_obs}, got {len(df_clean)}'}

        # 2. Prepare regression variables
        X = pd.DataFrame()
        X = sm.add_constant(df_clean['ln_size_lag'])
        X['ln_size_lag'] = df_clean['ln_size_lag'].astype(float)
        X = pd.concat([X, year_dummies], axis=1)
        y = df_clean['ln_size'].astype(float)
        print(X)
        # 3. Run regression
        model = sm.OLS(y, X).fit()

        # 4. Test Gibrat's Law (β = 1)
        coef = model.params['ln_size_lag']
        se = model.bse['ln_size_lag']
        t_stat = (coef - 1) / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(df_clean)-2))

        # 5. Calculate growth rates
        growth_rates = df_clean.groupby(id_column).apply(
            lambda x: np.diff(np.log(x[size_column]))
        ).explode()

        # 6. Results
        results = {
            'coefficient': coef,
            'std_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'r_squared': model.rsquared,
            'n_obs': len(df_clean),
            'n_firms': df_clean[id_column].nunique(),
            'gibrats_law_holds': p_value > 0.05,
            'growth_rate_mean': growth_rates.mean(),
            'growth_rate_std': growth_rates.std()
        }

        return results

    except Exception as e:
        return {'error': str(e)}

def format_results(results):
    """Format the test results into a readable string"""
    if 'error' in results:
        return f"Error in testing Gibrat's law: {results['error']}"

    output = [
        "Gibrat's Law Test Results",
        "========================",
        f"Coefficient (β): {results['coefficient']:.3f}",
        f"Standard Error: {results['std_error']:.3f}",
        f"T-statistic (H0: β=1): {results['t_statistic']:.3f}",
        f"P-value: {results['p_value']:.3f}",
        f"Gibrat's Law holds: {results['gibrats_law_holds']}",
        f"R-squared: {results['r_squared']:.3f}",
        f"Number of observations: {results['n_obs']}",
        f"Number of firms: {results['n_firms']}",
        f"Mean growth rate: {results['growth_rate_mean']:.3f}",
        f"Growth rate std dev: {results['growth_rate_std']:.3f}"
    ]

    return '\n'.join(output)

# Create summary table
def create_summary_table(results):
    """Create a summary table of Gibrat's Law test results"""
    rows = []
    for sector in results:
        for measure in results[sector]:
            r = results[sector][measure]
            rows.append({
                'Sector': sector,
                'Size_Measure': measure,
                'Coefficient': round(r['coefficient'], 3),
                'Std_Error': round(r['std_error'], 3),
                'P_Value': round(r['p_value'], 3),
                'Gibrats_Law_Holds': r['gibrats_law_holds'],
                'N_Obs': r['n_obs']
            })
    return pd.DataFrame(rows)

def test_gibrats_law_by_sector(data, size_column, id_column='id_bvd', year_column='year', sector_column='NACE_Rev_2_main_section', min_obs=30):
    """
    Test Gibrat's Law for each sector separately
    """
    results = {}
    sectors = data[sector_column].unique()

    print(f"Testing Gibrat's law for {len(sectors)} sectors")

    for sector in sectors:
        print(f"\nAnalyzing sector: {sector}")

        # Filter data for current sector
        sector_data = data[data[sector_column] == sector].copy()

        # Run Gibrat's test for this sector
        sector_result = test_gibrats_law(
            data=sector_data,
            size_column=size_column,
            id_column=id_column,
            year_column=year_column,
            min_obs=min_obs
        )

        # Store results
        results[sector] = sector_result

    return results

def format_sector_results(results):
    """Format results for all sectors"""
    output = ["Gibrat's Law Test Results by Sector", "==============================="]

    for sector, result in results.items():
        output.append(f"\nSector: {sector}")
        output.append("-" * (len(sector) + 8))

        if 'error' in result:
            output.append(f"Error: {result['error']}")
            continue

        output.extend([
            f"Coefficient (β): {result['coefficient']:.3f}",
            f"Standard Error: {result['std_error']:.3f}",
            f"T-statistic (H0: β=1): {result['t_statistic']:.3f}",
            f"P-value: {result['p_value']:.3f}",
            f"Gibrat's Law holds: {result['gibrats_law_holds']}",
            f"R-squared: {result['r_squared']:.3f}",
            f"Number of observations: {result['n_obs']}",
            f"Number of firms: {result['n_firms']}",
            f"Mean growth rate: {result['growth_rate_mean']:.3f}",
            f"Growth rate std dev: {result['growth_rate_std']:.3f}"
        ])

    return '\n'.join(output)

def create_sector_summary_table(results):
    """Create a summary DataFrame of results by sector"""
    rows = []
    for sector, result in results.items():
        if 'error' in result:
            continue

        rows.append({
            'Sector': sector,
            'Coefficient': round(result['coefficient'], 3),
            'Std_Error': round(result['std_error'], 3),
            'P_Value': round(result['p_value'], 3),
            'Gibrats_Law_Holds': result['gibrats_law_holds'],
            'R_squared': round(result['r_squared'], 3),
            'N_Obs': result['n_obs'],
            'N_Firms': result['n_firms'],
            'Mean_Growth': round(result['growth_rate_mean'], 3),
            'Growth_Std': round(result['growth_rate_std'], 3)
        })

    return pd.DataFrame(rows).sort_values('Sector')

def main():
    ## Part 1 , Summary Statistics and Preprocessing
    dataPath='firms.csv'
    # 1. clean the dataset
    df = sum_stats.load_and_clean_data(dataPath)
    # 2. get summary statistics for numerical values
    sumStatsNum=sum_stats.create_summary_statistics(df,percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    # 2.1 print summary statistics
    print(sumStatsNum)
    # 2.2 export as a latex table
    latex_output = exp.summary_to_latex(
                                        sumStatsNum,
                                        caption="Summary Statistics",
                                        label="tab:summary_stats"
                                        )
    #print(latex_output)
    # 2.3 save to file corresponding to latex format
    latex_table_num = exp.summary_to_latex(sumStatsNum, caption="Summary Statistics")
    exp.save_latex_table(latex_table_num, 'tables/summary.tex')
    # 3. get summary statistics for categorical values
    sumStatsCat = sum_stats.analyze_categorical_variables(df)
    print(sumStatsCat)
    latex_table_cat = exp.categorical_to_latex(sumStatsCat)
    #print(latex_table_cat)
    exp.save_categorical_latex(latex_table_cat, 'tables/categorical')
    '''
        Some firms are being shown 1 time only...
        Exclude those firms from analysis
    '''
    # 4. exlude non frequent observations
    firm_counts = df['id_bvd'].value_counts()
    frequent_firms = firm_counts[firm_counts > 1].index
    df_filtered = df[df['id_bvd'].isin(frequent_firms)]
    # 5. Filter based on sector representation (>5%)
    # 5.1 Calculate sector percentages
    sector_counts = df_filtered['NACE_Rev_2_main_section'].value_counts()
    total_firms = len(df_filtered)
    sector_percentages = (sector_counts / total_firms) * 100
    # 5.2 Get sectors with >5% representation
    significant_sectors = sector_percentages[sector_percentages > 4].index
    # Apply the sector filter
    df_filtered = df_filtered[df_filtered['NACE_Rev_2_main_section'].isin(significant_sectors)]
    # Print information about filtering
    print(f"\nFiltering Results:")
    print(f"Original number of firms: {df['id_bvd'].nunique()}")
    print(f"Number of firms after filtering: {df_filtered['id_bvd'].nunique()}")
    print(f"Original number of observations: {len(df)}")
    print(f"Number of observations after filtering: {len(df_filtered)}")
    df_filtered.to_csv("fFirms.csv",index=False)
    ## PART 2 Test Gibrat's Law
    """
        Firm Analysis
    """
    size_results = test_gibrats_law(
        data=df_filtered,
        size_column='Size',
        id_column='id_bvd',
        year_column='year'
    )
    print(format_results(size_results))
    sales_results = test_gibrats_law(
        data=df_filtered,
        size_column='Salesn',
        id_column='id_bvd',
        year_column='year'
    )
    print(format_results(sales_results))
    # Create and save firm-level LaTeX table
    firm_latex = exp.create_latex_firm_table(size_results, sales_results)
    with open('tables/gibrat_firm_results.tex', 'w') as f:
        f.write(firm_latex)
    """
        Sector Analysis
    """
    # Print results
    # test by sector
    # test with assets as size
    sector_results = test_gibrats_law_by_sector(
        data=df_filtered,
        size_column='Size',
        id_column='id_bvd',
        year_column='year',
        sector_column='NACE_Rev_2_main_section'
    )
    # Create and display summary table
    summary_table = create_sector_summary_table(sector_results)
    print("\nSummary Table:")
    print(summary_table)
    # Print detailed results
    print("\nDetailed Results:")
    print(format_sector_results(sector_results))
    # Save in latex format . STATA Like
    sector_size_latex = exp.create_latex_sector_table(sector_results, "Size")
    with open('tables/gibrat_sector_size_results.tex', 'w') as f:
        f.write(sector_size_latex)
    #
    #
    # test for sales as size
    sector_results = test_gibrats_law_by_sector(
        data=df_filtered,
        size_column='Salesn',  # Changed from 'Size' to 'Salesn'
        id_column='id_bvd',
        year_column='year',
        sector_column='NACE_Rev_2_main_section'
    )
    # Create and display summary table
    summary_table = create_sector_summary_table(sector_results)
    print("\nSummary Table:")
    print(summary_table)
    # Print detailed results
    print("\nDetailed Results:")
    print(format_sector_results(sector_results))
    # Save in latex format . STATA like
    sector_sales_latex = exp.create_latex_sector_table(sector_results, "Sales")
    with open('tables/gibrat_sector_sales_results.tex', 'w') as f:
        f.write(sector_sales_latex)

if __name__ == "__main__":
    main()
