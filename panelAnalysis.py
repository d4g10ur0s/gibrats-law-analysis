import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.linalg import inv

def run_random_effects_analysis(data, model_spec, robust=True):
    """
    Perform random effects regression analysis with comprehensive diagnostics

    Parameters:
    -----------
    data : pandas DataFrame
        Panel dataset
    model_spec : dict
        Model specification containing variable lists
    robust : bool
        Whether to use robust standard errors

    Returns:
    --------
    dict
        Regression results and diagnostics
    """
    try:
        # Create working copy
        df = data.copy()

        # Get variables
        y_var = model_spec['dependent'][0]
        x_vars = model_spec['firm_vars'] + model_spec['industry_vars']

        # Print initial diagnostics
        print(f"Dependent variable: {y_var}")
        print(f"Independent variables: {x_vars}")
        print(f"\nInitial data shape: {df.shape}")

        # Entity (firm) identifier
        entity_id = 'id_bvd'  # Default firm identifier
        time_id = 'year'      # Default time identifier

        # Convert all variables to numeric
        for var in [y_var] + x_vars:
            original_dtype = df[var].dtype
            df[var] = pd.to_numeric(df[var], errors='coerce')
            print(f"Converting {var}: {original_dtype} -> {df[var].dtype}")
            print(f"NaN count for {var}: {df[var].isna().sum()}")

        # Drop missing values
        original_rows = len(df)
        df = df.dropna(subset=[y_var] + x_vars + [entity_id, time_id])
        print(f"\nRows removed due to missing values: {original_rows - len(df)}")
        print(f"Remaining rows: {len(df)}")

        # Check for sufficient observations
        min_obs = 30
        if len(df) < min_obs:
            return {'error': f'Insufficient observations. Need {min_obs}, got {len(df)}'}

        # Calculate group means
        group_means = df.groupby(entity_id)[x_vars + [y_var]].transform('mean')

        # Calculate overall means
        overall_means = df[x_vars + [y_var]].mean()

        # Calculate within and between variations
        X_within = df[x_vars] - group_means[x_vars]
        X_between = group_means[x_vars] - overall_means[x_vars]

        y_within = df[y_var] - group_means[y_var]
        y_between = group_means[y_var] - overall_means[y_var]

        # Get group sizes
        n_i = df.groupby(entity_id).size()
        N = len(df)
        n = len(n_i)
        T = N/n  # Average group size

        # Calculate variance components
        sigma2_within = (y_within**2).sum() / (N - n)
        sigma2_between = (y_between**2).sum() / (n - 1)

        # Calculate theta (random effects transformation parameter)
        theta = 1 - np.sqrt(sigma2_within / (sigma2_within + T * sigma2_between))

        # Transform variables
        X_star = df[x_vars] - theta * group_means[x_vars]
        y_star = df[y_var] - theta * group_means[y_var]

        # Add constant
        X_star = sm.add_constant(X_star)

        # Check for perfect collinearity
        eigenvals = np.linalg.eigvals(X_star.T @ X_star)
        condition_number = np.sqrt(np.max(eigenvals) / np.min(eigenvals))
        print(f"\nCondition number: {condition_number}")

        if condition_number > 1e15:
            print("Warning: High condition number indicates potential collinearity")

        # Fit random effects model
        try:
            model = sm.OLS(y_star, X_star)
            if robust:
                results = model.fit(cov_type='HC3')
            else:
                results = model.fit()
        except np.linalg.LinAlgError as e:
            return {'error': f'Linear algebra error in model fitting: {str(e)}'}
        except Exception as e:
            return {'error': f'Error in model fitting: {str(e)}'}

        # Calculate R-squared statistics
        y_mean = df[y_var].mean()
        tss = np.sum((df[y_var] - y_mean) ** 2)
        ess = np.sum((results.fittedvalues - y_mean) ** 2)
        rss = np.sum(results.resid ** 2)

        r2_within = 1 - (np.sum(y_within**2) / np.sum((df[y_var] - group_means[y_var])**2))
        r2_between = 1 - (np.sum(y_between**2) / np.sum((group_means[y_var] - y_mean)**2))
        r2_overall = 1 - (rss / tss)

        # Get confidence intervals
        conf_int = results.conf_int()

        # Calculate Breusch-Pagan LM test for random effects
        groups = pd.Categorical(df[entity_id])
        ols_resid = sm.OLS(df[y_var], sm.add_constant(df[x_vars])).fit().resid
        lm_stat = N * T / (2 * (T - 1)) * \
                  (np.sum(np.square(pd.DataFrame(ols_resid).groupby(groups).sum())) / \
                   np.sum(np.square(ols_resid)) - 1)**2
        lm_pvalue = 1 - stats.chi2.cdf(lm_stat, 1)

        # Prepare output dictionary
        output = {
            'coefficients': results.params.to_dict(),
            'std_errors': results.bse.to_dict(),
            't_statistics': results.tvalues.to_dict(),
            'p_values': results.pvalues.to_dict(),
            'conf_intervals': {
                var: [conf_int.loc[var, 0], conf_int.loc[var, 1]]
                for var in results.params.index
            },
            'random_effects': {
                'theta': theta,
                'sigma2_within': sigma2_within,
                'sigma2_between': sigma2_between
            },
            'model_stats': {
                'r2_within': r2_within,
                'r2_between': r2_between,
                'r2_overall': r2_overall,
                'num_obs': N,
                'num_entities': n,
                'avg_obs_per_entity': T,
                'f_statistic': results.fvalue,
                'f_p_value': results.f_pvalue,
                'condition_number': condition_number
            },
            'diagnostics': {
                'breusch_pagan_lm': {
                    'statistic': lm_stat,
                    'p_value': lm_pvalue
                },
                'durbin_watson': sm.stats.stattools.durbin_watson(results.resid),
                'jarque_bera': {
                    'statistic': stats.jarque_bera(results.resid)[0],
                    'p_value': stats.jarque_bera(results.resid)[1]
                }
            }
        }

        return output

    except Exception as e:
        import traceback
        error_msg = f"Error in random_effects_analysis: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}

def format_random_effects_as_stata_table(results):
    """
    Format random effects results in Stata-style output with improved Series handling

    Parameters:
    -----------
    results : dict
        Results from random_effects_analysis function

    Returns:
    --------
    str
        Formatted table string
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    try:
        # Convert model statistics to float
        n_obs = int(results['model_stats']['num_obs'])
        n_groups = int(results['model_stats']['num_entities'])
        r2_within = float(results['model_stats']['r2_within'])
        r2_between = float(results['model_stats']['r2_between'])
        r2_overall = float(results['model_stats']['r2_overall'])
        theta = float(results['random_effects']['theta'])

        # Header
        output = [
            "\nRandom-effects GLS regression",
            f"Number of obs: {n_obs}",
            f"Number of groups: {n_groups}",
            "",
            f"R-sq:  within  = {r2_within:.4f}",
            f"       between = {r2_between:.4f}",
            f"       overall = {r2_overall:.4f}",
            "",
            f"theta: {theta:.4f}",
            "",
            "{:<30} {:<12} {:<12} {:<12} {:<12} {:<20}".format(
                "Variable", "Coef.", "Std. Err.", "z", "P>|z|", "[95% Conf. Interval]"
            ),
            "-" * 100
        ]

        # Add coefficients
        for var in results['coefficients']:
            # Convert values to float
            coef = float(results['coefficients'][var])
            std_err = float(results['std_errors'][var])
            t_stat = float(results['t_statistics'][var])
            p_val = float(results['p_values'][var])
            conf_int = [float(results['conf_intervals'][var][0]),
                       float(results['conf_intervals'][var][1])]

            output.append(
                "{:<30} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} [{:>8.4f}, {:>8.4f}]".format(
                    str(var)[:30],
                    coef,
                    std_err,
                    t_stat,
                    p_val,
                    conf_int[0],
                    conf_int[1]
                )
            )

        # Convert diagnostic values to float
        bp_stat = float(results['diagnostics']['breusch_pagan_lm']['statistic'])
        bp_pval = float(results['diagnostics']['breusch_pagan_lm']['p_value'])
        dw_stat = float(results['diagnostics']['durbin_watson'])
        jb_stat = float(results['diagnostics']['jarque_bera']['statistic'])
        jb_pval = float(results['diagnostics']['jarque_bera']['p_value'])

        # Add diagnostics
        output.extend([
            "-" * 100,
            "\nDiagnostic Tests:",
            f"Breusch-Pagan LM test for random effects:",
            f"  chi2({1}) = {bp_stat:.4f}",
            f"  Prob > chi2 = {bp_pval:.4f}",
            "",
            f"Durbin-Watson statistic: {dw_stat:.4f}",
            "",
            f"Jarque-Bera normality test:",
            f"  chi2({2}) = {jb_stat:.4f}",
            f"  Prob > chi2 = {jb_pval:.4f}"
        ])

        return '\n'.join(output)

    except Exception as e:
        return f"Error formatting results: {str(e)}"


def run_fixed_effects_analysis(data, model_spec, robust=True):
    """
    Perform fixed effects regression analysis with improved error handling and debugging
    """
    try:
        # Create working copy
        df = data.copy()

        # Get variables
        y_var = model_spec['dependent'][0]
        x_vars = model_spec['firm_vars'] + model_spec['industry_vars']

        # Add debugging prints
        print(f"Dependent variable: {y_var}")
        print(f"Independent variables: {x_vars}")

        # Entity (firm) identifier
        entity_id = 'id_bvd'  # Default firm identifier
        time_id = 'year'      # Default time identifier

        # Print initial data shape
        print(f"\nInitial data shape: {df.shape}")

        # Ensure all variables are numeric and log the conversion
        for var in [y_var] + x_vars:
            original_dtype = df[var].dtype
            df[var] = pd.to_numeric(df[var], errors='coerce')
            print(f"Converting {var}: {original_dtype} -> {df[var].dtype}")
            print(f"NaN count for {var}: {df[var].isna().sum()}")

        # Drop missing values with logging
        original_rows = len(df)
        df = df.dropna(subset=[y_var] + x_vars + [entity_id, time_id])
        print(f"\nRows removed due to missing values: {original_rows - len(df)}")
        print(f"Remaining rows: {len(df)}")

        # Check for sufficient observations
        min_obs = 30  # Minimum observations required
        if len(df) < min_obs:
            return {'error': f'Insufficient observations. Need {min_obs}, got {len(df)}'}

        # Entity-specific means
        entity_means = df.groupby(entity_id)[x_vars + [y_var]].transform('mean')
        print(f"\nEntity means shape: {entity_means.shape}")

        # Within transformation (demeaning)
        X_within = df[x_vars] - entity_means[x_vars]
        y_within = df[y_var] - entity_means[y_var]

        print(f"Within transformed X shape: {X_within.shape}")
        print(f"Within transformed y shape: {y_within.shape}")

        # Add constant
        X_within = sm.add_constant(X_within)
        print(f"X with constant shape: {X_within.shape}")

        # Check for perfect collinearity
        eigenvals = np.linalg.eigvals(X_within.T @ X_within)
        condition_number = np.sqrt(np.max(eigenvals) / np.min(eigenvals))
        print(f"\nCondition number: {condition_number}")

        if condition_number > 1e15:
            print("Warning: High condition number indicates potential collinearity")

        # Fit model with try-except for specific error handling
        try:
            model = sm.OLS(y_within, X_within)
            if robust:
                results = model.fit(cov_type='HC3')
            else:
                results = model.fit()
        except np.linalg.LinAlgError as e:
            return {'error': f'Linear algebra error in model fitting: {str(e)}'}
        except Exception as e:
            return {'error': f'Error in model fitting: {str(e)}'}

        # Calculate fixed effects (intercepts)
        fe_means = df.groupby(entity_id)[y_var].mean() - np.dot(
            df.groupby(entity_id)[x_vars].mean(),
            results.params[1:]
        )

        # Calculate R-squared statistics
        y_mean = df[y_var].mean()
        tss = np.sum((df[y_var] - y_mean) ** 2)
        ess = np.sum((results.fittedvalues - y_mean) ** 2)
        rss = np.sum(results.resid ** 2)
        r2_within = 1 - (rss / np.sum((y_within) ** 2))
        r2_between = 1 - (np.sum((entity_means[y_var] - y_mean) ** 2) / np.sum((df[y_var] - y_mean) ** 2))
        r2_overall = 1 - (rss / tss)

        # Get confidence intervals
        conf_int = results.conf_int()

        # F-test for fixed effects
        n_entities = len(df[entity_id].unique())
        n_obs = len(df)
        k = len(x_vars)
        f_stat = ((tss - rss) / (n_entities - 1)) / (rss / (n_obs - n_entities - k))
        f_pvalue = 1 - stats.f.cdf(f_stat, n_entities - 1, n_obs - n_entities - k)

        # Prepare output dictionary with additional diagnostics
        output = {
            'coefficients': results.params.to_dict(),
            'std_errors': results.bse.to_dict(),
            't_statistics': results.tvalues.to_dict(),
            'p_values': results.pvalues.to_dict(),
            'conf_intervals': {
                var: [conf_int.loc[var, 0], conf_int.loc[var, 1]]
                for var in results.params.index
            },
            'fixed_effects': {
                'mean': fe_means.mean(),
                'std': fe_means.std(),
                'min': fe_means.min(),
                'max': fe_means.max()
            },
            'model_stats': {
                'r2_within': r2_within,
                'r2_between': r2_between,
                'r2_overall': r2_overall,
                'num_obs': n_obs,
                'num_entities': n_entities,
                'f_statistic': results.fvalue,
                'f_p_value': results.f_pvalue,
                'fe_f_stat': f_stat,
                'fe_f_pvalue': f_pvalue,
                'condition_number': condition_number
            }
        }

        print("\nModel successfully fitted!")
        print(f"Number of observations: {n_obs}")
        print(f"Number of entities: {n_entities}")
        print(f"R-squared within: {r2_within:.4f}")

        return output

    except Exception as e:
        import traceback
        error_msg = f"Error in fixed_effects_analysis: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Print the error for immediate debugging
        return {'error': error_msg}

def format_fixed_effects_results(results):
    """
    Format fixed effects results in a clean, readable format

    Parameters:
    -----------
    results : dict
        Results from fixed_effects_analysis function

    Returns:
    --------
    str
        Formatted results string
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    output = [
        "Fixed Effects Regression Results",
        "===============================",
        "\nModel Statistics:",
        f"R² within:  {results['model_stats']['r2_within']:.4f}",
        f"R² between: {results['model_stats']['r2_between']:.4f}",
        f"R² overall: {results['model_stats']['r2_overall']:.4f}",
        f"Number of observations: {results['model_stats']['num_obs']}",
        f"Number of entities: {results['model_stats']['num_entities']}",
        f"F-statistic: {results['model_stats']['f_statistic']:.4f}",
        f"Prob > F: {results['model_stats']['f_p_value']:.4f}",
        f"\nF test for fixed effects:",
        f"F-statistic: {results['model_stats']['fe_f_stat']:.4f}",
        f"Prob > F: {results['model_stats']['fe_f_pvalue']:.4f}",
        "\nFixed Effects Summary:",
        f"Mean:  {results['fixed_effects']['mean']:.4f}",
        f"Std:   {results['fixed_effects']['std']:.4f}",
        f"Min:   {results['fixed_effects']['min']:.4f}",
        f"Max:   {results['fixed_effects']['max']:.4f}",
        "\nCoefficients:",
        "--------------"
    ]

    # Add coefficient details
    for var in results['coefficients']:
        conf_int = results['conf_intervals'][var]
        output.append(
            f"\n{var}:"
            f"\n  Coefficient: {results['coefficients'][var]:>10.4f}"
            f"\n  Std Error:   {results['std_errors'][var]:>10.4f}"
            f"\n  t-statistic: {results['t_statistics'][var]:>10.4f}"
            f"\n  p-value:     {results['p_values'][var]:>10.4f}"
            f"\n  95% CI:      [{conf_int[0]:>10.4f}, {conf_int[1]:>10.4f}]"
        )

    # Add diagnostic tests
    output.extend([
        "\nDiagnostic Tests:",
        "-----------------",
        f"Durbin-Watson: {results['diagnostics']['durbin_watson']:.4f}",
        f"Condition Number: {results['diagnostics']['condition_number']:.4f}",
        "\nBreusch-Pagan test for heteroskedasticity:",
        f"  Statistic: {results['diagnostics']['breusch_pagan']['statistic']:.4f}",
        f"  p-value: {results['diagnostics']['breusch_pagan']['p_value']:.4f}",
        "\nJarque-Bera test for normality:",
        f"  Statistic: {results['diagnostics']['jarque_bera']['statistic']:.4f}",
        f"  p-value: {results['diagnostics']['jarque_bera']['p_value']:.4f}"
    ])

    return '\n'.join(output)

def format_fixed_effects_as_stata_table(results):
    """
    Format fixed effects results in Stata-style output

    Parameters:
    -----------
    results : dict
        Results from fixed_effects_analysis function

    Returns:
    --------
    str
        Formatted table string
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    # Header
    output = [
        "\n{:<30} {:<12} {:<12} {:<12} {:<12} {:<20}".format(
            "Variable", "Coef.", "Std. Err.", "t", "P>|t|", "[95% Conf. Interval]"
        ),
        "-" * 100
    ]

    # Add coefficients
    for var in results['coefficients']:
        conf_int = results['conf_intervals'][var]
        output.append(
            "{:<30} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} [{:>8.4f}, {:>8.4f}]".format(
                var[:30],
                results['coefficients'][var],
                results['std_errors'][var],
                results['t_statistics'][var],
                results['p_values'][var],
                conf_int[0],
                conf_int[1]
            )
        )

    # Add model statistics
    output.extend([
        "-" * 100,
        f"Number of obs: {results['model_stats']['num_obs']}",
        f"Number of groups: {results['model_stats']['num_entities']}",
        f"R-sq: within = {results['model_stats']['r2_within']:.4f}",
        f"      between = {results['model_stats']['r2_between']:.4f}",
        f"      overall = {results['model_stats']['r2_overall']:.4f}",
        f"F({len(results['coefficients'])-1},{results['model_stats']['num_obs']-results['model_stats']['num_entities']-len(results['coefficients'])+1}) = {results['model_stats']['f_statistic']:.4f}",
        f"Prob > F = {results['model_stats']['f_p_value']:.4f}",
        f"\nF test that all u_i=0:",
        f"F({results['model_stats']['num_entities']-1}, {results['model_stats']['num_obs']-results['model_stats']['num_entities']-len(results['coefficients'])+1}) = {results['model_stats']['fe_f_stat']:.4f}",
        f"Prob > F = {results['model_stats']['fe_f_pvalue']:.4f}"
    ])

    return '\n'.join(output)

def clean_panel_data(data, model_spec):
    """
    Clean and prepare panel data for analysis

    Parameters:
    -----------
    data : pandas DataFrame
        Raw panel dataset
    model_spec : dict
        Model specification containing variable lists

    Returns:
    --------
    pandas DataFrame
        Cleaned dataset ready for analysis
    """
    # Create working copy
    df = data.copy()

    # Get all variables needed
    dependent_var = model_spec['dependent'][0]
    independent_vars = model_spec['firm_vars'] + model_spec['industry_vars']
    all_vars = [dependent_var] + independent_vars

    # Convert all variables to numeric
    for var in all_vars:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing values in key variables
    df = df.dropna(subset=all_vars)

    # Create year dummies if specified in controls
    if 'controls' in model_spec and 'time' in model_spec['controls']:
        year_dummies = pd.get_dummies(df[model_spec['controls']['time']], prefix='year')
        df = pd.concat([df, year_dummies], axis=1)

    # Create industry dummies if specified in controls
    if 'controls' in model_spec and 'industry' in model_spec['controls']:
        industry_dummies = pd.get_dummies(df[model_spec['controls']['industry']], prefix='industry')
        df = pd.concat([df, industry_dummies], axis=1)

    return df

def pooled_ols(data, model_spec, robust=True):
    """
    Perform pooled OLS regression with comprehensive diagnostics and matrix validation

    Parameters:
    -----------
    data : pandas DataFrame
        Panel dataset
    model_spec : dict
        Model specification containing variable lists
    robust : bool
        Whether to use robust standard errors

    Returns:
    --------
    dict
        Regression results
    """
    try:
        # Clean data and ensure numeric types
        df_clean = clean_panel_data(data, model_spec)

        # Get variables
        y_var = model_spec['dependent'][0]
        x_vars = model_spec['firm_vars'] + model_spec['industry_vars']

        # Ensure all variables are numeric
        for var in [y_var] + x_vars:
            df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')

        # Add control variables if present
        dummy_vars = []
        if 'controls' in model_spec:
            if 'time' in model_spec['controls']:
                time_dummies = pd.get_dummies(df_clean[model_spec['controls']['time']],
                                            prefix='year',
                                            drop_first=True)
                df_clean = pd.concat([df_clean, time_dummies], axis=1)
                dummy_vars.extend(time_dummies.columns)

            if 'industry' in model_spec['controls']:
                industry_dummies = pd.get_dummies(df_clean[model_spec['controls']['industry']],
                                                prefix='industry',
                                                drop_first=True)
                df_clean = pd.concat([df_clean, industry_dummies], axis=1)
                dummy_vars.extend(industry_dummies.columns)

        # Combine all variables
        all_x_vars = x_vars + dummy_vars

        # Drop any remaining NaN values
        df_clean = df_clean.dropna(subset=[y_var] + all_x_vars)

        # Check for zero-variance columns
        variances = df_clean[all_x_vars].var()
        valid_vars = variances[variances > 0].index.tolist()

        if len(valid_vars) == 0:
            return {'error': 'No valid independent variables after cleaning'}

        # Prepare X and y
        y = df_clean[y_var].astype(float)
        X = df_clean[valid_vars].astype(float)
        X = sm.add_constant(X)

        # Fit model
        model = sm.OLS(y, X)
        if robust:
            results = model.fit(cov_type='HC3')
        else:
            results = model.fit()

        # Get confidence intervals
        conf_int = results.conf_int()

        # Calculate R-squared
        y_mean = y.mean()
        tss = np.sum((y - y_mean) ** 2)
        rss = np.sum(results.resid ** 2)
        r_squared = 1 - (rss / tss)
        adj_r_squared = 1 - (1 - r_squared) * ((len(y) - 1) / (len(y) - X.shape[1]))

        # Calculate VIF correctly
        #vifs = {}
        #X_no_const = X.iloc[:, 1:]  # Remove constant
        #for i, col in enumerate(X_no_const.columns):
            # For each variable, regress it on all other variables
        #    y_i = X_no_const[col]
        #    X_i = X_no_const.drop(columns=[col])
        #    X_i = sm.add_constant(X_i)
        #    model_i = sm.OLS(y_i, X_i).fit()
        #    vifs[col] = 1 / (1 - model_i.rsquared)

        # Prepare results dictionary
        output = {
            'coefficients': results.params.to_dict(),
            'std_errors': results.bse.to_dict(),
            't_statistics': results.tvalues.to_dict(),
            'p_values': results.pvalues.to_dict(),
            'conf_intervals': {
                var: [conf_int.loc[var, 0], conf_int.loc[var, 1]]
                for var in results.params.index
            },
            'model_stats': {
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'num_obs': len(y),
                'df_model': X.shape[1] - 1,
                'df_resid': len(y) - X.shape[1],
                'aic': results.aic,
                'bic': results.bic
            }
        }

        # Calculate F-statistic
        if X.shape[1] > 1:
            df_model = X.shape[1] - 1
            df_resid = len(y) - X.shape[1]
            mse_model = (tss - rss) / df_model
            mse_resid = rss / df_resid
            f_stat = mse_model / mse_resid
            f_pvalue = 1 - stats.f.cdf(f_stat, df_model, df_resid)

            output['model_stats'].update({
                'f_statistic': f_stat,
                'f_p_value': f_pvalue
            })

        # Calculate diagnostics
        resids = results.resid
        output['diagnostics'] = {
            'durbin_watson': sm.stats.stattools.durbin_watson(resids),
            'condition_number': np.linalg.cond(X),
            #'vif': vifs,
            #'mean_vif': np.mean(list(vifs.values())),
            'breusch_pagan': {
                'statistic': sm.stats.diagnostic.het_breuschpagan(resids, X)[0],
                'p_value': sm.stats.diagnostic.het_breuschpagan(resids, X)[1]
            },
            'jarque_bera': {
                'statistic': stats.jarque_bera(resids)[0],
                'p_value': stats.jarque_bera(resids)[1]
            }
        }

        return output

    except Exception as e:
        import traceback
        error_msg = f"Error in pooled_ols: {str(e)}\n{traceback.format_exc()}"
        return {'error': error_msg}

def format_pooled_ols_results(results):
    """
    Format pooled OLS results in a clear, readable format

    Parameters:
    -----------
    results : dict
        Results from pooled_ols function

    Returns:
    --------
    str
        Formatted results string
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    output = [
        "Pooled OLS Regression Results",
        "============================",
        "\nModel Statistics:",
        f"R-squared: {results['model_stats']['r_squared']:.4f}",
        f"Adjusted R-squared: {results['model_stats']['adj_r_squared']:.4f}",
        f"F-statistic: {results['model_stats']['f_statistic']:.4f}",
        f"F-statistic p-value: {results['model_stats']['f_p_value']:.4f}",
        f"Number of observations: {results['model_stats']['num_obs']}",
        f"AIC: {results['model_stats']['aic']:.4f}",
        f"BIC: {results['model_stats']['bic']:.4f}",
        "\nCoefficients:",
        "--------------"
    ]

    # Add coefficient details
    for var in results['coefficients']:
        conf_int = results['conf_intervals'][var]
        output.append(
            f"\n{var}:"
            f"\n  Coefficient: {results['coefficients'][var]:>10.4f}"
            f"\n  Std Error:   {results['std_errors'][var]:>10.4f}"
            f"\n  t-statistic: {results['t_statistics'][var]:>10.4f}"
            f"\n  p-value:     {results['p_values'][var]:>10.4f}"
            f"\n  95% CI:      [{conf_int[0]:>10.4f}, {conf_int[1]:>10.4f}]"
        )

    # Add diagnostic tests
    output.extend([
        "\nDiagnostic Tests:",
        "-----------------",
        f"Durbin-Watson: {results['diagnostics']['durbin_watson']:.4f}",
        f"Condition Number: {results['diagnostics']['condition_number']:.4f}",
        "\nBreusch-Pagan test for heteroskedasticity:",
        f"  Statistic: {results['diagnostics']['breusch_pagan']['statistic']:.4f}",
        f"  p-value: {results['diagnostics']['breusch_pagan']['p_value']:.4f}",
        "\nJarque-Bera test for normality:",
        f"  Statistic: {results['diagnostics']['jarque_bera']['statistic']:.4f}",
        f"  p-value: {results['diagnostics']['jarque_bera']['p_value']:.4f}"
    ])

    return '\n'.join(output)

def format_stata_style_table(results):
    """
    Format results in a Stata-like table format with control variable handling
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    try:
        # Header
        output = [
            "{:<30} {:<12} {:<12} {:<12} {:<12} {:<20}".format(
                "Variable", "Coef.", "Std. Err.", "t", "P>|t|", "[95% Conf. Interval]"
            ),
            "-" * 100
        ]

        # Sort variables to group controls together
        sorted_vars = sorted(results['coefficients'].keys(),
                           key=lambda x: ('year_' in x or 'industry_' in x, x))

        # Add each variable with explicit type conversion
        for var in sorted_vars:
            try:
                # Convert all numeric values to native Python float
                coef = float(results['coefficients'][var])
                std_err = float(results['std_errors'][var])
                t_stat = float(results['t_statistics'][var])
                p_val = float(results['p_values'][var])
                conf_int_lower = float(results['conf_intervals'][var][0])
                conf_int_upper = float(results['conf_intervals'][var][1])

                # Format variable name
                var_name = str(var)[:30]
                if 'year_' in var or 'industry_' in var:
                    var_name = '  ' + var_name  # Indent control variables

                # Format the row with explicit float conversions
                output.append(
                    "{:<30} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} [{:>8.4f}, {:>8.4f}]".format(
                        var_name,
                        coef,
                        std_err,
                        t_stat,
                        p_val,
                        conf_int_lower,
                        conf_int_upper
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                print(f"Warning: Error formatting variable {var}: {str(e)}")
                continue

        # Add model statistics with explicit type conversion
        try:
            n_obs = int(results['model_stats']['num_obs'])
            r2 = float(results['model_stats']['r_squared'])
            adj_r2 = float(results['model_stats']['adj_r_squared'])
            f_stat = float(results['model_stats']['f_statistic'])
            f_pval = float(results['model_stats']['f_p_value'])

            output.extend([
                "-" * 100,
                f"Number of obs: {n_obs}",
                f"R-squared: {r2:.4f}",
                f"Adj R-squared: {adj_r2:.4f}",
                f"F-statistic: {f_stat:.4f}",
                f"Prob > F: {f_pval:.4f}"
            ])
        except (KeyError, TypeError, ValueError) as e:
            print(f"Warning: Error formatting model statistics: {str(e)}")
            output.extend([
                "-" * 100,
                "Note: Some model statistics could not be formatted"
            ])

        return '\n'.join(output)

    except Exception as e:
        return f"Error formatting table: {str(e)}"

def run_iv_analysis(data, model_spec):
    """
    Perform instrumental variables (IV) regression using model specification

    Parameters:
    -----------
    data : pandas DataFrame
        Panel dataset
    model_spec : dict
        Model specification containing:
        - dependent: list of dependent variable
        - endogenous: list of endogenous variable
        - instruments: list of instrumental variables
        - controls: list of control variables

    Returns:
    --------
    dict
        Results from both stages of IV regression
    """
    try:
        # Create working copy
        df = data.copy()

        # Get variables from model specification
        y_var = model_spec['dependent'][0]
        endogenous = model_spec['endogenous'][0]
        instruments = model_spec['instruments']
        controls = model_spec['controls']

        # Print initial diagnostics
        print(f"\nInitial data shape: {df.shape}")
        print(f"Dependent variable: {y_var}")
        print(f"Endogenous variable: {endogenous}")
        print(f"Instruments: {instruments}")
        print(f"Controls: {controls}")

        # Convert to numeric and handle missing/infinite values
        all_vars = [y_var, endogenous] + instruments + controls
        for var in all_vars:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df[var] = df[var].replace([np.inf, -np.inf], np.nan)
            print(f"{var} - NaN count: {df[var].isna().sum()}")

        # Clean data
        initial_rows = len(df)
        df = df.dropna(subset=all_vars)
        print(f"\nRows removed due to missing values: {initial_rows - len(df)}")
        print(f"Remaining rows: {len(df)}")
        # First stage regression
        X_first = sm.add_constant(df[instruments])
        y_first = df[endogenous]
        print("\nFirst stage regression shapes:")
        print(f"X_first: {X_first.shape}")
        print(f"y_first: {y_first.shape}")
        first_stage = sm.OLS(y_first, X_first).fit()

        # Calculate F-statistic for instrument strength
        r2_first = first_stage.rsquared
        n = len(df)
        k = len(instruments)
        f_stat = (r2_first / k) / ((1 - r2_first) / (n - k - 1))
        f_pvalue = 1 - stats.f.cdf(f_stat, k, n - k - 1)

        # Get predicted values of endogenous variable
        ownership_hat = first_stage.predict(X_first)

        # Second stage regression
        X_second = sm.add_constant(pd.concat([
            pd.Series(ownership_hat, name=endogenous),
            df[controls]
        ], axis=1))
        y_second = df[y_var]

        print("\nSecond stage regression shapes:")
        print(f"X_second: {X_second.shape}")
        print(f"y_second: {y_second.shape}")

        second_stage = sm.OLS(y_second, X_second).fit()

        # Calculate Sargan test for overidentification
        resid_2sls = second_stage.resid
        X_sargan = sm.add_constant(df[instruments])
        sargan_reg = sm.OLS(resid_2sls, X_sargan).fit()
        sargan_stat = n * sargan_reg.rsquared
        sargan_pvalue = 1 - stats.chi2.cdf(sargan_stat, len(instruments) - 1)

        # Calculate Hausman test
        ols_model = sm.OLS(df[y_var], sm.add_constant(df[[endogenous] + controls])).fit()
        coef_diff = second_stage.params - ols_model.params
        var_diff = second_stage.cov_params() - ols_model.cov_params()
        hausman_stat = np.dot(coef_diff, np.dot(np.linalg.pinv(var_diff), coef_diff))
        hausman_pvalue = 1 - stats.chi2.cdf(hausman_stat, len(coef_diff))

        # Calculate Stock-Yogo test critical values
        sy_critical_values = {
            0.10: 22.30,  # 10% maximal size
            0.15: 12.83,  # 15% maximal size
            0.20: 9.54,   # 20% maximal size
            0.25: 7.80    # 25% maximal size
        }

        # Determine instrument strength based on Stock-Yogo
        weak_instruments = "Strong"
        for size, critical_value in sy_critical_values.items():
            if f_stat < critical_value:
                weak_instruments = f"Weak (>{size*100}% bias)"
                break

        # Prepare results dictionary
        results = {
            'first_stage': {
                'coefficients': first_stage.params.to_dict(),
                'std_errors': first_stage.bse.to_dict(),
                't_statistics': first_stage.tvalues.to_dict(),
                'p_values': first_stage.pvalues.to_dict(),
                'r_squared': first_stage.rsquared,
                'adj_r_squared': first_stage.rsquared_adj,
                'n_obs': n
            },
            'second_stage': {
                'coefficients': second_stage.params.to_dict(),
                'std_errors': second_stage.bse.to_dict(),
                't_statistics': second_stage.tvalues.to_dict(),
                'p_values': second_stage.pvalues.to_dict(),
                'r_squared': second_stage.rsquared,
                'adj_r_squared': second_stage.rsquared_adj,
                'n_obs': n
            },
            'diagnostics': {
                'instrument_strength': {
                    'f_statistic': f_stat,
                    'f_pvalue': f_pvalue,
                    'conclusion': weak_instruments,
                    'critical_values': sy_critical_values
                },
                'sargan_test': {
                    'statistic': sargan_stat,
                    'p_value': sargan_pvalue,
                    'conclusion': 'Valid instruments' if sargan_pvalue > 0.05 else 'Invalid instruments'
                },
                'hausman_test': {
                    'statistic': hausman_stat,
                    'p_value': hausman_pvalue,
                    'conclusion': 'Use IV' if hausman_pvalue < 0.05 else 'OLS sufficient'
                }
            }
        }

        return results

    except Exception as e:
        import traceback
        error_msg = f"Error in IV analysis: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}

def format_iv_results(results):
    """
    Format IV regression results in a readable format

    Parameters:
    -----------
    results : dict
        Results from run_iv_analysis

    Returns:
    --------
    str
        Formatted results string
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    output = [
        "Instrumental Variables (2SLS) Regression Results",
        "============================================",
        "\nFirst Stage Results (Dependent Variable: Ownership Concentration)",
        "---------------------------------------------------------"
    ]

    # First stage results
    output.extend([
        f"R-squared: {results['first_stage']['r_squared']:.4f}",
        f"Adjusted R-squared: {results['first_stage']['adj_r_squared']:.4f}",
        "\nCoefficients:",
        "{:<20} {:>12} {:>12} {:>12} {:>12}".format(
            "Variable", "Coef.", "Std.Err.", "t-stat", "P>|t|"
        ),
        "-" * 68
    ])

    for var in results['first_stage']['coefficients']:
        output.append(
            "{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                var[:20],
                results['first_stage']['coefficients'][var],
                results['first_stage']['std_errors'][var],
                results['first_stage']['t_statistics'][var],
                results['first_stage']['p_values'][var]
            )
        )

    # Second stage results
    output.extend([
        "\nSecond Stage Results (Dependent Variable: Tobin's Q)",
        "------------------------------------------------",
        f"R-squared: {results['second_stage']['r_squared']:.4f}",
        f"Adjusted R-squared: {results['second_stage']['adj_r_squared']:.4f}",
        "\nCoefficients:",
        "{:<20} {:>12} {:>12} {:>12} {:>12}".format(
            "Variable", "Coef.", "Std.Err.", "t-stat", "P>|t|"
        ),
        "-" * 68
    ])

    for var in results['second_stage']['coefficients']:
        output.append(
            "{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                var[:20],
                results['second_stage']['coefficients'][var],
                results['second_stage']['std_errors'][var],
                results['second_stage']['t_statistics'][var],
                results['second_stage']['p_values'][var]
            )
        )

    # Diagnostic tests
    output.extend([
        "\nDiagnostic Tests:",
        "-----------------",
        "Instrument Strength (F-test):",
        f"  F-statistic: {results['diagnostics']['instrument_strength']['f_statistic']:.4f}",
        f"  P-value: {results['diagnostics']['instrument_strength']['f_pvalue']:.4f}",
        f"  Conclusion: {results['diagnostics']['instrument_strength']['conclusion']}",
        "\nSargan Test (Overidentification):",
        f"  Chi-square: {results['diagnostics']['sargan_test']['statistic']:.4f}",
        f"  P-value: {results['diagnostics']['sargan_test']['p_value']:.4f}",
        f"  Conclusion: {results['diagnostics']['sargan_test']['conclusion']}",
        "\nHausman Test (Endogeneity):",
        f"  Chi-square: {results['diagnostics']['hausman_test']['statistic']:.4f}",
        f"  P-value: {results['diagnostics']['hausman_test']['p_value']:.4f}",
        f"  Conclusion: {results['diagnostics']['hausman_test']['conclusion']}"
    ])

    return '\n'.join(output)

def analyze_heterogeneity(data, model_spec):
    """
    Analyze country and industry heterogeneity in IV regression with proper data type handling
    """
    try:
        # Create working copy
        df = data.copy()

        # Get variables
        y_var = model_spec['dependent'][0]
        endogenous = model_spec['endogenous'][0]
        instruments = model_spec['instruments']
        controls = model_spec['controls']

        # Convert all variables to numeric
        for var in [y_var, endogenous] + instruments + controls:
            df[var] = df[var].replace([np.inf, -np.inf], np.nan)
            print(f"Converting {var} to numeric. NaN count: {df[var].isna().sum()}")

        # Create country dummies
        country_dummies = pd.get_dummies(df['CountryISOcode'], prefix='country', drop_first=True)
        country_cols = country_dummies.columns.tolist()

        # Create industry dummies
        industry_dummies = pd.get_dummies(df['NACE_Rev_2_main_section'], prefix='industry', drop_first=True)
        industry_cols = industry_dummies.columns.tolist()

        # Clean data
        all_vars = [y_var, endogenous] + instruments + controls
        print(f"Shape after cleaning numeric variables: {df.shape}")

        # Add dummies to dataframe
        df = pd.concat([df, country_dummies, industry_dummies], axis=1)
        df = df.dropna(subset=all_vars)

        # First stage with fixed effects
        X_first = df[instruments + controls + country_cols + industry_cols].astype(float)
        X_first = sm.add_constant(X_first)
        y_first = df[endogenous].astype(float)

        first_stage = sm.OLS(y_first, X_first).fit()

        # Calculate F-statistic for instruments
        r2_first = first_stage.rsquared
        n = len(df)
        k = len(instruments)
        f_stat = (r2_first / k) / ((1 - r2_first) / (n - k - 1))
        f_pvalue = 1 - stats.f.cdf(f_stat, k, n - k - 1)

        # Get predicted ownership concentration
        ownership_hat = first_stage.predict(X_first)

        # Second stage with fixed effects
        X_second = pd.concat([
            pd.Series(ownership_hat, name=endogenous),
            df[controls + country_cols + industry_cols]
        ], axis=1).astype(float)
        X_second = sm.add_constant(X_second)
        y_second = df[y_var].astype(float)

        second_stage = sm.OLS(y_second, X_second).fit(cov_type='HC3')

        # Test joint significance of country dummies
        country_idx = [list(X_second.columns).index(col) for col in country_cols]
        r_matrix_country = np.zeros((len(country_cols), len(second_stage.params)))
        for i, idx in enumerate(country_idx):
            r_matrix_country[i, idx] = 1

        f_test_country = second_stage.f_test(r_matrix_country)

        # Test joint significance of industry dummies
        industry_idx = [list(X_second.columns).index(col) for col in industry_cols]
        r_matrix_industry = np.zeros((len(industry_cols), len(second_stage.params)))
        for i, idx in enumerate(industry_idx):
            r_matrix_industry[i, idx] = 1

        f_test_industry = second_stage.f_test(r_matrix_industry)

        # Collect heterogeneity effects
        results = {
            'model_results': {
                'first_stage': {
                    'coefficients': first_stage.params.to_dict(),
                    'std_errors': first_stage.bse.to_dict(),
                    't_statistics': first_stage.tvalues.to_dict(),
                    'p_values': first_stage.pvalues.to_dict(),
                    'r_squared': float(first_stage.rsquared),
                    'adj_r_squared': float(first_stage.rsquared_adj)
                },
                'second_stage': {
                    'coefficients': second_stage.params.to_dict(),
                    'std_errors': second_stage.bse.to_dict(),
                    't_statistics': second_stage.tvalues.to_dict(),
                    'p_values': second_stage.pvalues.to_dict(),
                    'r_squared': float(second_stage.rsquared),
                    'adj_r_squared': float(second_stage.rsquared_adj)
                }
            },
            'heterogeneity': {
                'country_effects': {
                    'joint_test': {
                        'f_statistic': float(f_test_country.statistic),
                        'p_value': float(f_test_country.pvalue),
                        'df_num': len(country_cols),
                        'df_denom': n - len(X_second.columns)
                    },
                    'individual_effects': {
                        country: {
                            'coefficient': float(second_stage.params[country]),
                            'std_error': float(second_stage.bse[country]),
                            't_statistic': float(second_stage.tvalues[country]),
                            'p_value': float(second_stage.pvalues[country])
                        } for country in country_cols
                    }
                },
                'industry_effects': {
                    'joint_test': {
                        'f_statistic': float(f_test_industry.statistic),
                        'p_value': float(f_test_industry.pvalue),
                        'df_num': len(industry_cols),
                        'df_denom': n - len(X_second.columns)
                    },
                    'individual_effects': {
                        industry: {
                            'coefficient': float(second_stage.params[industry]),
                            'std_error': float(second_stage.bse[industry]),
                            't_statistic': float(second_stage.tvalues[industry]),
                            'p_value': float(second_stage.pvalues[industry])
                        } for industry in industry_cols
                    }
                }
            }
        }

        return results

    except Exception as e:
        import traceback
        error_msg = f"Error in heterogeneity analysis: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}

def format_heterogeneity_results(results):
    """
    Format heterogeneity analysis results in a readable format
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    output = [
        "Heterogeneity Analysis Results",
        "============================",
        "\nCountry Fixed Effects:",
        "-------------------"
    ]

    # Country effects
    country_f = results['heterogeneity']['country_effects']['joint_test']
    output.extend([
        f"Joint Test of Country Effects:",
        f"F-statistic: {country_f['f_statistic']:.4f}",
        f"P-value: {country_f['p_value']:.4f}",
        "\nSignificant Country Effects (p < 0.05):"
    ])

    for country, effects in results['heterogeneity']['country_effects']['individual_effects'].items():
        if effects['p_value'] < 0.05:
            output.append(
                f"{country}: {effects['coefficient']:.4f} (p={effects['p_value']:.4f})"
            )

    # Industry effects
    output.extend([
        "\nIndustry Fixed Effects:",
        "--------------------"
    ])

    industry_f = results['heterogeneity']['industry_effects']['joint_test']
    output.extend([
        f"Joint Test of Industry Effects:",
        f"F-statistic: {industry_f['f_statistic']:.4f}",
        f"P-value: {industry_f['p_value']:.4f}",
        "\nSignificant Industry Effects (p < 0.05):"
    ])

    for industry, effects in results['heterogeneity']['industry_effects']['individual_effects'].items():
        if effects['p_value'] < 0.05:
            output.append(
                f"{industry}: {effects['coefficient']:.4f} (p={effects['p_value']:.4f})"
            )

    return '\n'.join(output)
