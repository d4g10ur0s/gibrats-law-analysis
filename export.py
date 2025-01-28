import pandas as pd

def format_coefficient_with_stars(coef, p_value):
    """Format coefficient with STATA-style significance stars"""
    coef_str = f"{coef:.3f}"
    if p_value < 0.01:
        coef_str += "$^{***}$"
    elif p_value < 0.05:
        coef_str += "$^{**}$"
    elif p_value < 0.1:
        coef_str += "$^{*}$"
    return coef_str

def create_latex_firm_table(size_results, sales_results):
    """Create LaTeX table for firm-level analysis"""
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Gibrat's Law Test Results - Firm Level}",
        "\\label{tab:gibrat_firm}",
        "\\begin{tabular}{lcc}",
        "\\hline\\hline",
        "& Size & Sales \\\\",
        "\\hline"
    ]

    # Add coefficient row with stars
    size_coef = format_coefficient_with_stars(size_results['coefficient'], size_results['p_value'])
    sales_coef = format_coefficient_with_stars(sales_results['coefficient'], sales_results['p_value'])
    latex.append(f"Coefficient ($\\beta$) & {size_coef} & {sales_coef} \\\\")

    # Add standard errors in parentheses
    latex.append(f"& ({size_results['std_error']:.3f}) & ({sales_results['std_error']:.3f}) \\\\")

    # Add other statistics
    latex.extend([
        f"R-squared & {size_results['r_squared']:.3f} & {sales_results['r_squared']:.3f} \\\\",
        f"Observations & {size_results['n_obs']} & {sales_results['n_obs']} \\\\",
        f"Number of Firms & {size_results['n_firms']} & {sales_results['n_firms']} \\\\",
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Notes: Standard errors in parentheses. $^{***}$ p<0.01, $^{**}$ p<0.05, $^{*}$ p<0.1",
        "\\item Gibrat's Law holds if coefficient equals 1.",
        "\\end{tablenotes}",
        "\\end{table}"
    ])

    return '\n'.join(latex)

def create_latex_sector_table(sector_results, measure_type):
    """Create LaTeX table for sector-level analysis"""
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Gibrat's Law Test Results by Sector - {measure_type}}}",
        f"\\label{{tab:gibrat_sector_{measure_type.lower()}}}",
        "\\begin{tabular}{lccccc}",
        "\\hline\\hline",
        "Sector & Coefficient & R-squared & Obs & Firms & Growth Mean \\\\",
        "\\hline"
    ]

    # Add rows for each sector
    for sector, result in sector_results.items():
        if 'error' in result:
            continue

        coef = format_coefficient_with_stars(result['coefficient'], result['p_value'])
        latex.append(
            f"{sector} & {coef} & {result['r_squared']:.3f} & "
            f"{result['n_obs']} & {result['n_firms']} & {result['growth_rate_mean']:.3f} \\\\"
        )

    latex.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Notes: $^{***}$ p<0.01, $^{**}$ p<0.05, $^{*}$ p<0.1",
        "\\item Gibrat's Law holds if coefficient equals 1.",
        "\\end{tablenotes}",
        "\\end{table}"
    ])

    return '\n'.join(latex)

def summary_to_latex(summary_df, caption=None, label=None, float_format=lambda x: '{:,.3f}'.format(x)):
    """
    Convert summary statistics DataFrame to a LaTeX table

    Parameters:
    -----------
    summary_df : pandas.DataFrame
        Summary statistics DataFrame
    caption : str, optional
        Table caption
    label : str, optional
        Table label for referencing
    float_format : callable, optional
        Function to format float numbers

    Returns:
    --------
    str : LaTeX table code
    """
    # Create a copy to avoid modifying the original
    df_latex = summary_df.copy()

    # Rename the index for better readability
    index_mapping = {
        'count': 'N',
        'mean': 'Mean',
        'std': 'Std Dev',
        'min': 'Min',
        'max': 'Max',
        'skew': 'Skewness',
        'kurtosis': 'Kurtosis',
        'missing_count': 'Missing (N)',
        'missing_pct': 'Missing (\%)',
        '5%': 'P5',
        '25%': 'P25',
        '50%': 'P50',
        '75%': 'P75',
        '95%': 'P95'
    }
    df_latex.index = df_latex.index.map(lambda x: index_mapping.get(x, x))

    # Convert to LaTeX
    latex_table = df_latex.to_latex(
        float_format=float_format,
        escape=False,
        caption=caption,
        label=label,
        column_format='l' + 'r' * len(df_latex.columns)  # left align first column, right align others
    )

    # Add some common LaTeX table improvements
    latex_table = latex_table.replace('table', 'table*')  # Make table span multiple columns
    latex_table = latex_table.replace('\\begin{table*}',
                                    '\\begin{table*}[htbp]\n\\centering\n\\small')

    return latex_table

def save_latex_table(latex_table, output_path, standalone=False):
    """
    Save LaTeX table to a file

    Parameters:
    -----------
    latex_table : str
        LaTeX table code
    output_path : str
        Path where to save the file (e.g., 'tables/summary_stats.tex')
    standalone : bool, default False
        If True, adds LaTeX document wrapper for standalone compilation
    """
    # Create directories if they don't exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if standalone:
        # Add document wrapper for standalone compilation
        content = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{siunitx}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\begin{document}\n\n"
            f"{latex_table}\n\n"
            "\\end{document}"
        )
    else:
        content = latex_table

    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"LaTeX table saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving LaTeX table: {str(e)}")

# Example usage combining both functions:
def create_and_save_summary_latex(df, output_path, caption=None, label=None, standalone=False):
    """
    Create summary statistics and save as LaTeX table

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    output_path : str
        Path where to save the LaTeX file
    caption : str, optional
        Table caption
    label : str, optional
        Table label for referencing
    standalone : bool, default False
        If True, creates a standalone LaTeX document
    """
    # Create summary statistics
    summary_stats = create_summary_statistics(df, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    # Convert to LaTeX
    latex_table = summary_to_latex(
        summary_stats,
        caption=caption,
        label=label
    )

    # Save to file
    save_latex_table(latex_table, output_path, standalone=standalone)

def categorical_to_latex(cat_summary, caption_prefix="Distribution of", float_format=lambda x: '{:,.2f}'.format(x)):
    """
    Convert categorical analysis results to LaTeX tables

    Parameters:
    -----------
    cat_summary : dict
        Dictionary output from analyze_categorical_variables
    caption_prefix : str
        Prefix for table captions
    float_format : callable
        Function to format float numbers

    Returns:
    --------
    dict : Dictionary of LaTeX tables
    """
    latex_tables = {}

    for var_name, df in cat_summary.items():
        # Create a copy to avoid modifying the original
        df_latex = df.copy()

        # Format column names
        column_mapping = {
            'Count': 'N',
            'Percentage': '\% of Total',
            'Cumulative %': 'Cumulative \%'
        }
        df_latex.columns = [column_mapping.get(col, col) for col in df_latex.columns]

        # Create caption and label
        clean_name = var_name.replace('_', ' ').title()
        caption = f"{caption_prefix} {clean_name}"
        label = f"tab:dist_{var_name.lower().replace(' ', '_')}"

        # Convert to LaTeX
        latex_table = df_latex.to_latex(
            float_format=float_format,
            escape=False,
            caption=caption,
            label=label,
            column_format='l' + 'r' * len(df_latex.columns)
        )

        # Add LaTeX improvements
        latex_table = latex_table.replace('table', 'table*')
        latex_table = latex_table.replace('\\begin{table*}',
                                        '\\begin{table*}[htbp]\n\\centering\n\\small')

        latex_tables[var_name] = latex_table

    return latex_tables

def save_categorical_latex(cat_latex_tables, output_dir, standalone=False):
    """
    Save categorical LaTeX tables to files

    Parameters:
    -----------
    cat_latex_tables : dict
        Dictionary of LaTeX tables
    output_dir : str
        Directory to save the files
    standalone : bool
        Whether to create standalone LaTeX documents
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for var_name, latex_table in cat_latex_tables.items():
        # Create filename
        filename = f"distribution_{var_name.lower().replace(' ', '_')}.tex"
        output_path = os.path.join(output_dir, filename)

        # Save table
        save_latex_table(latex_table, output_path, standalone)

# Combined function for convenience
def create_and_save_categorical_latex(df, output_dir, standalone=False, **cat_analysis_kwargs):
    """
    Create and save categorical analysis LaTeX tables

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    output_dir : str
        Directory to save the files
    standalone : bool
        Whether to create standalone LaTeX documents
    **cat_analysis_kwargs :
        Additional arguments for analyze_categorical_variables
    """
    # Create categorical analysis
    cat_summary = analyze_categorical_variables(df, **cat_analysis_kwargs)

    # Convert to LaTeX
    latex_tables = categorical_to_latex(cat_summary)

    # Save tables
    save_categorical_latex(latex_tables, output_dir, standalone)

def format_iv_stata_style(results):
    """
    Format IV regression results in Stata-style output

    Parameters:
    -----------
    results : dict
        Results from IV regression analysis

    Returns:
    --------
    str
        Formatted table string in Stata style
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    # Get number of observations from results
    n_obs = len(results['first_stage']['coefficients'])

    # Get instrument variables (those not in second stage)
    second_stage_vars = set(results['second_stage']['coefficients'].keys())
    first_stage_vars = set(results['first_stage']['coefficients'].keys())
    instruments = sorted(list(first_stage_vars - second_stage_vars))

    output = [
        "\nInstrumental Variables (2SLS) Regression",
        f"Number of obs = {n_obs}",
        "",
        "First-stage regressions",
        "-----------------------",
        "",
        "                                                  Ownership_Concentration",
        "-----------------------------------------------------------------------------"
    ]

    # First stage results
    first_stage_vars = sorted(results['first_stage']['coefficients'].keys())
    max_var_length = max(len(str(var)) for var in first_stage_vars)

    output.append("{:<{width}}     Coef.      Std. Err.     t      P>|t|".format(
        "", width=max_var_length))

    for var in first_stage_vars:
        coef = results['first_stage']['coefficients'][var]
        std_err = results['first_stage']['std_errors'][var]
        t_stat = results['first_stage']['t_statistics'][var]
        p_val = results['first_stage']['p_values'][var]

        coef_str = format_coefficient_with_stars(coef, p_val)
        var_pad = str(var).ljust(max_var_length)

        output.append(
            f"{var_pad}    {coef_str:>12}     {std_err:>10.4f}     {t_stat:>8.2f}    {p_val:>6.4f}"
        )

    # First stage statistics
    k = len(instruments)  # number of instruments
    n_first = n_obs  # number of observations

    output.extend([
        "",
        f"F({k}, {n_first - k - 1}) = {results['diagnostics']['instrument_strength']['f_statistic']:.2f}",
        f"Prob > F = {results['diagnostics']['instrument_strength']['f_pvalue']:.4f}",
        f"R-squared = {results['first_stage']['r_squared']:.4f}",
        f"Adj R-squared = {results['first_stage']['adj_r_squared']:.4f}",
        "",
        "Instrumental variables (2SLS) regression",
        "",
        "                                                      Tobin's Q",
        "-----------------------------------------------------------------------------"
    ])

    # Second stage results
    second_stage_vars = sorted(results['second_stage']['coefficients'].keys())

    output.append("{:<{width}}     Coef.      Std. Err.     z      P>|z|".format(
        "", width=max_var_length))

    for var in second_stage_vars:
        coef = results['second_stage']['coefficients'][var]
        std_err = results['second_stage']['std_errors'][var]
        z_stat = results['second_stage']['t_statistics'][var]  # t-stat is z-stat in IV
        p_val = results['second_stage']['p_values'][var]

        coef_str = format_coefficient_with_stars(coef, p_val)
        var_pad = str(var).ljust(max_var_length)

        output.append(
            f"{var_pad}    {coef_str:>12}     {std_err:>10.4f}     {z_stat:>8.2f}    {p_val:>6.4f}"
        )

    # Add diagnostic tests
    output.extend([
        "",
        "Instrumented: Ownership_Concentration",
        f"Instruments: {', '.join(instruments)}",
        "",
        "Tests of endogeneity",
        "Hausman (chi2) = {:.2f} (p = {:.4f})".format(
            results['diagnostics']['hausman_test']['statistic'],
            results['diagnostics']['hausman_test']['p_value']
        ),
        "",
        "Tests of overidentifying restrictions:",
        "Sargan (chi2) = {:.2f} (p = {:.4f})".format(
            results['diagnostics']['sargan_test']['statistic'],
            results['diagnostics']['sargan_test']['p_value']
        ),
        "",
        "First-stage F-statistic = {:.2f}".format(
            results['diagnostics']['instrument_strength']['f_statistic']
        )
    ])

    return '\n'.join(output)

def format_coefficient_with_stars(coef, p_value):
    """Format coefficient with STATA-style significance stars"""
    coef_str = f"{coef:.4f}"
    if p_value < 0.01:
        coef_str += "***"
    elif p_value < 0.05:
        coef_str += "**"
    elif p_value < 0.1:
        coef_str += "*"
    return coef_str

def create_iv_latex_table(results):
    """
    Create LaTeX table for IV regression results

    Parameters:
    -----------
    results : dict
        Results from IV regression analysis

    Returns:
    --------
    str
        LaTeX formatted table
    """
    # Get instrument variables (those not in second stage)
    second_stage_vars = set(results['second_stage']['coefficients'].keys())
    first_stage_vars = set(results['first_stage']['coefficients'].keys())
    instruments = sorted(list(first_stage_vars - second_stage_vars))

    # Get control variables (those in both stages)
    controls = sorted(list(second_stage_vars - {'const'}))

    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Instrumental Variables (2SLS) Regression Results}",
        "\\label{tab:iv_regression}",
        "\\begin{tabular}{lcccc}",
        "\\hline\\hline",
        "& \\multicolumn{2}{c}{First Stage} & \\multicolumn{2}{c}{Second Stage} \\\\",
        "& \\multicolumn{2}{c}{(Ownership Concentration)} & \\multicolumn{2}{c}{(Tobin's Q)} \\\\",
        "\\cline{2-3} \\cline{4-5}",
        "Variable & Coefficient & Std. Error & Coefficient & Std. Error \\\\"
    ]

    # Add instrumental variables first
    for var in instruments:
        first_coef = format_coefficient_with_stars(
            results['first_stage']['coefficients'][var],
            results['first_stage']['p_values'][var]
        )
        first_se = f"({results['first_stage']['std_errors'][var]:.4f})"

        latex.append(
            f"{var} & {first_coef} & {first_se} & \\multicolumn{{2}}{{c}}{{--}} \\\\"
        )

    # Add control variables
    for var in controls:
        first_coef = format_coefficient_with_stars(
            results['first_stage']['coefficients'][var],
            results['first_stage']['p_values'][var]
        )
        first_se = f"({results['first_stage']['std_errors'][var]:.4f})"

        second_coef = format_coefficient_with_stars(
            results['second_stage']['coefficients'][var],
            results['second_stage']['p_values'][var]
        )
        second_se = f"({results['second_stage']['std_errors'][var]:.4f})"

        latex.append(
            f"{var} & {first_coef} & {first_se} & {second_coef} & {second_se} \\\\"
        )

    # Add diagnostics
    latex.extend([
        "\\hline",
        f"Observations & \\multicolumn{{4}}{{c}}{{{len(results['first_stage']['coefficients'])}}} \\\\",
        f"F-statistic & \\multicolumn{{2}}{{c}}{{{results['diagnostics']['instrument_strength']['f_statistic']:.2f}}} & \\multicolumn{{2}}{{c}}{{--}} \\\\",
        f"R-squared & \\multicolumn{{2}}{{c}}{{{results['first_stage']['r_squared']:.4f}}} & \\multicolumn{{2}}{{c}}{{{results['second_stage']['r_squared']:.4f}}} \\\\",
        f"Sargan test (p-value) & \\multicolumn{{4}}{{c}}{{{results['diagnostics']['sargan_test']['p_value']:.4f}}} \\\\",
        f"Hausman test (p-value) & \\multicolumn{{4}}{{c}}{{{results['diagnostics']['hausman_test']['p_value']:.4f}}} \\\\",
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1",
        f"\\item Instruments: {', '.join(instruments)}",
        "\\end{tablenotes}",
        "\\end{table}"
    ])

    return '\n'.join(latex)

def format_coefficient_with_stars(coef, p_value):
    """Format coefficient with STATA-style significance stars"""
    coef_str = f"{coef:.4f}"
    if p_value < 0.01:
        coef_str += "***"
    elif p_value < 0.05:
        coef_str += "**"
    elif p_value < 0.1:
        coef_str += "*"
    return coef_str

def format_heterogeneity_stata_style(results):
    """
    Format heterogeneity analysis results in Stata-style output

    Parameters:
    -----------
    results : dict
        Results from heterogeneity analysis

    Returns:
    --------
    str
        Formatted table string in Stata style
    """
    if 'error' in results:
        return f"Error in analysis: {results['error']}"

    # Get dimensions
    n_obs = len(results['model_results']['first_stage']['coefficients'])
    n_countries = len(results['heterogeneity']['country_effects']['individual_effects'])
    n_industries = len(results['heterogeneity']['industry_effects']['individual_effects'])

    output = [
        "\nInstrumental Variables (2SLS) Regression with Fixed Effects",
        f"Number of obs = {n_obs}",
        f"Number of countries = {n_countries}",
        f"Number of industries = {n_industries}",
        "",
        "First-stage regression",
        "---------------------",
        "",
        "                                                  Ownership_Concentration",
        "-----------------------------------------------------------------------------"
    ]

    # First stage results
    first_stage = results['model_results']['first_stage']
    vars_first = sorted(first_stage['coefficients'].keys())
    max_var_length = max(len(str(var)) for var in vars_first)

    # Add header
    output.append("{:<{width}}     Coef.      Std. Err.     t      P>|t|".format(
        "", width=max_var_length))
    output.append("-" * (max_var_length + 50))

    # Add coefficients
    for var in vars_first:
        coef = first_stage['coefficients'][var]
        std_err = first_stage['std_errors'][var]
        t_stat = first_stage['t_statistics'][var]
        p_val = first_stage['p_values'][var]

        coef_str = format_coefficient_with_stars(coef, p_val)
        var_pad = str(var).ljust(max_var_length)

        output.append(
            f"{var_pad}    {coef_str:>12}     {std_err:>10.4f}     {t_stat:>8.2f}    {p_val:>6.4f}"
        )

    # Add first stage statistics
    output.extend([
        "",
        f"R-squared = {first_stage['r_squared']:.4f}",
        f"Adj R-squared = {first_stage['adj_r_squared']:.4f}",
        "",
        "Second-stage regression",
        "----------------------",
        "",
        "                                                      Tobin's Q",
        "-----------------------------------------------------------------------------"
    ])

    # Second stage results
    second_stage = results['model_results']['second_stage']
    vars_second = sorted(second_stage['coefficients'].keys())

    # Add header
    output.append("{:<{width}}     Coef.      Std. Err.     z      P>|z|".format(
        "", width=max_var_length))
    output.append("-" * (max_var_length + 50))

    # Add coefficients
    for var in vars_second:
        coef = second_stage['coefficients'][var]
        std_err = second_stage['std_errors'][var]
        z_stat = second_stage['t_statistics'][var]
        p_val = second_stage['p_values'][var]

        coef_str = format_coefficient_with_stars(coef, p_val)
        var_pad = str(var).ljust(max_var_length)

        output.append(
            f"{var_pad}    {coef_str:>12}     {std_err:>10.4f}     {z_stat:>8.2f}    {p_val:>6.4f}"
        )

    # Add second stage statistics
    output.extend([
        "",
        f"R-squared = {second_stage['r_squared']:.4f}",
        f"Adj R-squared = {second_stage['adj_r_squared']:.4f}",
        "",
        "Fixed Effects Tests",
        "-----------------"
    ])

    # Country effects
    country_test = results['heterogeneity']['country_effects']['joint_test']
    output.extend([
        "\nJoint test of country fixed effects:",
        f"F({country_test['df_num']}, {country_test['df_denom']}) = {country_test['f_statistic']:.2f}",
        f"Prob > F = {country_test['p_value']:.4f}"
    ])

    # Industry effects
    industry_test = results['heterogeneity']['industry_effects']['joint_test']
    output.extend([
        "\nJoint test of industry fixed effects:",
        f"F({industry_test['df_num']}, {industry_test['df_denom']}) = {industry_test['f_statistic']:.2f}",
        f"Prob > F = {industry_test['p_value']:.4f}",
        "",
        "Significant Fixed Effects (p < 0.05)",
        "--------------------------------"
    ])

    # Add significant country effects
    output.append("\nCountry Effects:")
    for country, effects in sorted(results['heterogeneity']['country_effects']['individual_effects'].items()):
        if effects['p_value'] < 0.05:
            coef_str = format_coefficient_with_stars(effects['coefficient'], effects['p_value'])
            output.append(f"{country:20} {coef_str:>12}  (p={effects['p_value']:.4f})")

    # Add significant industry effects
    output.append("\nIndustry Effects:")
    for industry, effects in sorted(results['heterogeneity']['industry_effects']['individual_effects'].items()):
        if effects['p_value'] < 0.05:
            coef_str = format_coefficient_with_stars(effects['coefficient'], effects['p_value'])
            output.append(f"{industry:20} {coef_str:>12}  (p={effects['p_value']:.4f})")

    return '\n'.join(output)

def create_heterogeneity_latex_table(results):
    """
    Create LaTeX table for heterogeneity analysis results
    """
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{IV Regression Results with Country and Industry Fixed Effects}",
        "\\label{tab:heterogeneity}",
        "\\begin{tabular}{lcccc}",
        "\\hline\\hline",
        "& \\multicolumn{2}{c}{First Stage} & \\multicolumn{2}{c}{Second Stage} \\\\",
        "& \\multicolumn{2}{c}{(Ownership Concentration)} & \\multicolumn{2}{c}{(Tobin's Q)} \\\\",
        "\\cline{2-3} \\cline{4-5}",
        "Variable & Coefficient & Std. Error & Coefficient & Std. Error \\\\"
    ]

    # Add main variables
    first_stage = results['model_results']['first_stage']
    second_stage = results['model_results']['second_stage']

    for var in sorted(first_stage['coefficients'].keys()):
        # First stage
        coef1 = format_coefficient_with_stars(
            first_stage['coefficients'][var],
            first_stage['p_values'][var]
        )
        se1 = f"({first_stage['std_errors'][var]:.4f})"

        # Second stage (if available)
        if var in second_stage['coefficients']:
            coef2 = format_coefficient_with_stars(
                second_stage['coefficients'][var],
                second_stage['p_values'][var]
            )
            se2 = f"({second_stage['std_errors'][var]:.4f})"
        else:
            coef2 = "--"
            se2 = "--"

        latex.append(f"{var} & {coef1} & {se1} & {coef2} & {se2} \\\\")

    # Add statistics
    latex.extend([
        "\\hline",
        f"Observations & \\multicolumn{{4}}{{c}}{{{len(first_stage['coefficients'])}}} \\\\",
        f"R-squared & \\multicolumn{{2}}{{c}}{{{first_stage['r_squared']:.4f}}} & \\multicolumn{{2}}{{c}}{{{second_stage['r_squared']:.4f}}} \\\\",
        "",
        "Fixed Effects Tests & \\multicolumn{4}{c}{F-statistic (p-value)} \\\\",
        f"Country Effects & \\multicolumn{{4}}{{c}}{{{results['heterogeneity']['country_effects']['joint_test']['f_statistic']:.2f} ({results['heterogeneity']['country_effects']['joint_test']['p_value']:.4f})}} \\\\",
        f"Industry Effects & \\multicolumn{{4}}{{c}}{{{results['heterogeneity']['industry_effects']['joint_test']['f_statistic']:.2f} ({results['heterogeneity']['industry_effects']['joint_test']['p_value']:.4f})}} \\\\",
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1",
        "\\item Country and industry fixed effects included in both stages",
        "\\end{tablenotes}",
        "\\end{table}"
    ])

    return '\n'.join(latex)
