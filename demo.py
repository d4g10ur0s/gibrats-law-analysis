import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

import summaryStatistics as sum_stats
import export as exp
import panelAnalysis as pa

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
        #print(X)
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

'''
    Panel Analysis
'''
def setup_panel_analysis(data, min_obs=30):
   """
   Setup for panel regression analysis by sector

   Parameters:
   -----------
   data : pandas DataFrame
       Panel dataset with firm observations
   min_obs : int
       Minimum observations required per sector

   Returns:
   --------
   dict : Contains prepared model specifications
   """
   # Model specifications
   models = {
       'profitability': {
           'dependent': ['ROA_Net_income', 'PM'],
           'firm_vars': [
               'Size',                    # Firm size
               'Growth_sales',            # Growth
               'Leverage',                # Risk
               'Intangible_fixed_assets'  # R&D proxy
           ],
           'industry_vars': [
               'HI_conc',                # Industry concentration
               'Ownership_Concetration'   # Ownership structure
           ],
           'controls': {
               'industry': 'NACE_Rev_2_main_section',
               'time': 'year'
           }
       },

       'tobins_q': {
           'dependent': ['Tobins_Q_'],
           'firm_vars': [
               'PM',                      # Profitability
               'Growth_sales',            # Growth
               'Leverage',                # Risk
               'Size',                    # Firm size
               'Intangible_fixed_assets'  # R&D proxy
           ],
           'industry_vars': [
               'HI_conc',
               'Ownership_Concetration'
           ],
           'controls': {
               'industry': 'NACE_Rev_2_main_section',
               'time': 'year'
           }
       }
   }

   # Panel identifiers
   panel_ids = {
       'firm': 'id_bvd',
       'time': 'year',
       'sector': 'NACE_Rev_2_main_section'
   }

   # Estimation methods
   methods = ['pooled_ols', 'fe', 're']

   return {
       'models': models,
       'panel_ids': panel_ids,
       'estimation': methods,
       'min_obs': min_obs
   }

def run_panel_analysis(data, model_spec, method='fe'):
    """
    Run panel regression analysis for specified model

    Parameters:
    -----------
    data : pandas DataFrame
        Panel dataset
    model_spec : dict
        Model specification from setup_panel_analysis
    method : str
        'fe', 're', or 'pooled_ols'

    Returns:
    --------
    dict
        Regression results and diagnostics
    """
    try:
        # Get variables
        y_var = model_spec['dependent'][0]
        x_vars = model_spec['firm_vars'] + model_spec['industry_vars']

        # Prepare data with cleaning
        data_clean = data.copy()

        # Remove rows with inf or NaN in dependent or independent variables
        cols_to_check = [y_var] + x_vars
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
        data_clean = data_clean.dropna(subset=cols_to_check)

        if method == 'fe':
            # Use existing fixed effects implementation
            results = pa.run_fixed_effects_analysis(
                data_clean,
                model_spec
            )
            return results

        elif method == 're':
            # Use the new random effects implementation
            results = pa.run_random_effects_analysis(
                data_clean,
                model_spec
            )
            return results

        else:  # pooled_ols
            # Use the custom Pooled OLS function
            results = pa.pooled_ols(
                data=data_clean,
                model_spec=model_spec,
                robust=False
            )
            return results

    except Exception as e:
        return {'error': str(e)}

def run_model_diagnostics(data, model_spec, panel_ids):
    """
    Run panel data model diagnostic tests with improved control variable handling
    """
    try:
        # Get variables
        y_var = model_spec['dependent'][0]
        x_vars = model_spec['firm_vars'] + model_spec['industry_vars']
        entity_id = panel_ids['firm']

        # Add control variables if present
        control_vars = []
        if 'controls' in model_spec:
            if 'time' in model_spec['controls']:
                time_dummies = pd.get_dummies(data[model_spec['controls']['time']],
                                            prefix='year',
                                            drop_first=True)
                data = pd.concat([data, time_dummies], axis=1)
                control_vars.extend(time_dummies.columns)

            if 'industry' in model_spec['controls']:
                industry_dummies = pd.get_dummies(data[model_spec['controls']['industry']],
                                                prefix='industry',
                                                drop_first=True)
                data = pd.concat([data, industry_dummies], axis=1)
                control_vars.extend(industry_dummies.columns)

        # All variables for X matrix
        all_x_vars = x_vars + control_vars

        # Run models with error handling
        try:
            fe_results = run_panel_analysis(data, model_spec, 'fe')
            re_results = run_panel_analysis(data, model_spec, 're')

            if 'error' in fe_results or 'error' in re_results:
                return {'error': 'Error in model estimation'}

            # Extract coefficients and variance matrices for Hausman test
            fe_coef = np.array(list(fe_results['coefficients'].values()))
            re_coef = np.array(list(re_results['coefficients'].values()))

            # Calculate Hausman test statistic
            coef_diff = fe_coef - re_coef
            df = len(coef_diff)

            # Use a simple variance calculation if standard errors are available
            fe_var = np.array([fe_results['std_errors'][var]**2 for var in fe_results['coefficients']])
            re_var = np.array([re_results['std_errors'][var]**2 for var in re_results['coefficients']])
            var_diff = fe_var - re_var

            # Calculate Hausman statistic
            h_stat = np.dot(coef_diff.T, np.dot(np.linalg.pinv(np.diag(var_diff)), coef_diff))
            h_pvalue = 1 - stats.chi2.cdf(h_stat, df)

        except Exception as e:
            print(f"Error in Hausman test: {str(e)}")
            h_stat = np.nan
            h_pvalue = np.nan

        # Calculate residuals for pooled OLS properly
        pooled_results = run_panel_analysis(data, model_spec, 'pooled_ols')
        if 'error' in pooled_results:
            return {'error': 'Error in pooled OLS estimation'}

        # Prepare X matrix and y vector for residual calculation
        X = data[all_x_vars].copy()
        X = sm.add_constant(X)
        y = data[y_var]

        # Calculate fitted values and residuals
        coef_values = np.array(list(pooled_results['coefficients'].values()))

        # Ensure X and coefficients have matching dimensions
        if X.shape[1] != len(coef_values):
            print(f"Warning: X matrix shape {X.shape} doesn't match coefficients length {len(coef_values)}")
            print("Adjusting X matrix to match coefficients...")
            X = X[list(pooled_results['coefficients'].keys())]

        fitted_values = np.dot(X, coef_values)
        resid = y - fitted_values

        try:
            # Breusch-Pagan test for heteroskedasticity
            bp_stat, bp_pvalue = sm.stats.diagnostic.het_breuschpagan(resid, X)
        except:
            bp_stat, bp_pvalue = np.nan, np.nan

        try:
            # Wooldridge test for serial correlation
            groups = pd.Categorical(data[entity_id])
            t_periods = len(data[panel_ids['time']].unique())

            if t_periods > 2:
                # Calculate first differences
                y_diff = data.groupby(entity_id)[y_var].diff()
                x_diff = data.groupby(entity_id)[all_x_vars].diff()

                # Run regression on differences
                model_diff = sm.OLS(y_diff.dropna(), sm.add_constant(x_diff.dropna())).fit()
                resid_diff = model_diff.resid

                # Test for serial correlation in residuals
                sc_stat, sc_pvalue = sm.stats.diagnostic.acorr_breusch_godfrey(resid_diff, nlags=1)
            else:
                sc_stat, sc_pvalue = np.nan, np.nan
        except:
            sc_stat, sc_pvalue = np.nan, np.nan

        # Prepare diagnostics dictionary
        diagnostics = {
            'hausman_test': {
                'statistic': float(h_stat),
                'p_value': float(h_pvalue),
                'conclusion': 'Use FE' if h_pvalue < 0.05 else 'Use RE'
            },
            'heteroskedasticity': {
                'statistic': float(bp_stat),
                'p_value': float(bp_pvalue),
                'conclusion': 'Present' if bp_pvalue < 0.05 else 'Not present'
            },
            'serial_correlation': {
                'statistic': float(sc_stat),
                'p_value': float(sc_pvalue),
                'conclusion': 'Present' if sc_pvalue < 0.05 else 'Not present'
            }
        }

        return diagnostics

    except Exception as e:
        import traceback
        error_msg = f"Error in model diagnostics: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}

def perform_sector_panel_analysis(data, panel_setup=None):
    """
    Perform panel analysis for each sector with improved results formatting

    Parameters:
    -----------
    data : pandas DataFrame
        Panel dataset with firm observations
    panel_setup : dict, optional
        Pre-configured panel analysis setup

    Returns:
    --------
    dict : Sector-wise panel analysis results
    """
    # If no panel setup provided, use default
    if panel_setup is None:
        panel_setup = setup_panel_analysis(data)

    # Initialize results dictionary
    sector_panel_results = {}

    # Get unique sectors
    sectors = data[panel_setup['panel_ids']['sector']].unique()

    # Iterate through each sector
    for sector in sectors:
        print(f"\n{'='*80}")
        print(f"Analysis for Sector: {sector}")
        print(f"{'='*80}")

        # Filter data for current sector
        sector_data = data[data[panel_setup['panel_ids']['sector']] == sector]

        # Skip if insufficient observations
        if len(sector_data) < panel_setup['min_obs']:
            print(f"Insufficient observations for sector {sector}")
            sector_panel_results[sector] = {'error': 'Insufficient observations'}
            continue

        # Initialize sector results
        sector_results = {
            'models': {},
            'diagnostics': {}
        }

        # Run analysis for each model specification
        for model_name, model_spec in panel_setup['models'].items():
            print(f"\nModel: {model_name}")
            print("-" * 40)

            # Run different estimation methods
            for method in panel_setup['estimation']:
                try:
                    print(f"\nEstimation Method: {method.upper()}")
                    print("-" * 30)

                    # Run panel analysis
                    analysis_results = run_panel_analysis(
                        data=sector_data,
                        model_spec=model_spec,
                        method=method
                    )

                    if 'error' in analysis_results:
                        print(f"Error in {method} estimation: {analysis_results['error']}")
                        continue

                    # Store results
                    sector_results['models'][f'{model_name}_{method}'] = analysis_results

                    # Format and print results based on method
                    if method == 'pooled_ols':
                        print("\nPooled OLS Results:")
                        print(pa.format_stata_style_table(analysis_results))
                    elif method == 'fe':
                        print("\nFixed Effects Results:")
                        print(pa.format_fixed_effects_as_stata_table(analysis_results))
                    elif method == 're':
                        print("\nRandom Effects Results:")
                        print(pa.format_random_effects_as_stata_table(analysis_results))

                    # Run diagnostics
                    try:
                        diagnostics = run_model_diagnostics(
                            data=sector_data,
                            model_spec=model_spec,
                            panel_ids=panel_setup['panel_ids']
                        )
                        sector_results['diagnostics'][f'{model_name}_{method}'] = diagnostics

                        if 'error' not in diagnostics:
                            print("\nDiagnostic Tests:")
                            print("-" * 20)
                            for test, result in diagnostics.items():
                                print(f"{test}:")
                                print(f"  Statistic: {result['statistic']:.4f}")
                                print(f"  P-value: {result['p_value']:.4f}")
                                print(f"  Conclusion: {result['conclusion']}")

                    except Exception as e:
                        print(f"Error in diagnostics: {str(e)}")

                except Exception as e:
                    print(f"Error in {method} estimation: {str(e)}")
                    sector_results['models'][f'{model_name}_{method}'] = {'error': str(e)}

        # Store sector results
        sector_panel_results[sector] = sector_results

    return sector_panel_results

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
    """
        Sector Analysis
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
    """
    '''
     Panel Analysis
    print("\nRunning Panel Analysis...")
    print("=" * 80)
    # Setup panel analysis
    panel_setup = setup_panel_analysis(df_filtered)
    sector_panel_results = perform_sector_panel_analysis(df_filtered, panel_setup)
    # Print results for each sector
    for sector, results in sector_panel_results.items():
        print(f"\nSector: {sector}")
        print("=" * 80)
        if 'error' in results:
            print(f"Error in sector analysis: {results['error']}")
            continue
        for model_name, model_results in results['models'].items():
            print(f"\nModel: {model_name}")
            print("-" * 40)
            if 'error' in model_results:
                print(f"Error in model: {model_results['error']}")
                continue
            # Format and print results based on method
            if '_pooled_ols' in model_name:
                print("\nPooled OLS Results:")
                print(pa.format_stata_style_table(model_results))
            elif '_fe' in model_name:
                print("\nFixed Effects Results:")
                print(pa.format_fixed_effects_as_stata_table(model_results))
            elif '_re' in model_name:
                print("\nRandom Effects Results:")
                print(pa.format_random_effects_as_stata_table(model_results))
            # Print diagnostics if available
            if model_name in results['diagnostics']:
                print("\nDiagnostic Tests:")
                print("-" * 20)
                for test, test_results in results['diagnostics'][model_name].items():
                    if isinstance(test_results, dict):
                        print(f"\n{test}:")
                        print(f"  Statistic: {test_results.get('statistic', 'N/A')}")
                        print(f"  P-value: {test_results.get('p_value', 'N/A')}")
                        print(f"  Conclusion: {test_results.get('conclusion', 'N/A')}")
    '''
    '''
        IV Analysis
    '''
    model_spec = {
        'dependent': ['Tobins_Q_'],
        'endogenous': ['Ownership_Concetration'],
        'instruments': ['CPI', 'JSEI', 'LN_SMCap_GDP', 'HI_conc'],
        'controls': ['PM', 'Growth_sales', 'Leverage', 'Size']
    }

    results = pa.run_iv_analysis(df_filtered, model_spec)
    print(pa.format_iv_results(results))
    print(exp.format_iv_stata_style(results))
    '''
        Heterogeneity
    model_spec = {
    'dependent': ['Tobins_Q_'],
    'endogenous': ['Ownership_Concetration'],
    'instruments': ['CPI', 'JSEI', 'LN_SMCap_GDP'],
    'controls': ['PM', 'Growth_sales', 'Leverage', 'Size', 'HI_conc']
    }
    # Run heterogeneity analysis
    het_results = pa.analyze_heterogeneity(df_filtered, model_spec)
    print(pa.format_heterogeneity_results(het_results))
    # Generate LaTeX table
    latex_table = exp.create_heterogeneity_latex_table(het_results)
    # Save LaTeX table
    print(exp.format_heterogeneity_stata_style(het_results))
    with open('heterogeneity_results.tex', 'w') as f:
        f.write(latex_table)
    '''




if __name__ == "__main__":
    main()
