"""
A Study on Adoption of Online Vehicle Insurance in Bengaluru
Replication and Analytical Pipeline Code

This script provides the end-to-end analytical pipeline for the study, including:
1. Data Preprocessing & Validation
2. Descriptive Statistics (Table 2)
3. Harman's Single Factor Test for Common Method Bias (CMB)
4. Exploratory Factor Analysis (PCA)
5. Construct Reliability - Alpha, Spearman-Brown, Inter-item r (Table 1)
6. Multicollinearity Diagnostics (VIF)
7. Bivariate Independence Tests (Chi-Square & Fisher's Exact)
8. Econometric Modelling - Logit & LPM (Table 3)

Author: Divyam Kumar, Md Hassan, Suman Kumar
License: MIT
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output in production
warnings.filterwarnings('ignore')

# --- CONFIGURATION & MAPPINGS ---

# Raw column names mapped to shorter variable names for clean code
COLUMN_MAPPING = {
    'Do you currently own & maintain insurance for a motorized vehicle in Bangalore?': 'Owns_Vehicle',
    'Your Age Group': 'Age_Group',
    'What type of vehicle do you primarily purchase/renew insurance for?': 'Vehicle_Type',
    'How did you purchase or renew your vehicle insurance most recently?': 'Recent_Channel_Raw',
    'For my next vehicle insurance renewal, I intend to:': 'Future_Renewal_Intent',
    
    # Core Constructs
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Online platforms allow me to compare quotes and issue policies faster than offline agents.]': 'PE1',
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Instant soft-copy policy issuance via email/WhatsApp is highly convenient.]': 'PE2',
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Navigating online insurance apps/websites and understanding coverage options is easy.]': 'EE1',
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Uploading inspection photos/videos online during renewal requires minimal effort.]': 'EE2',
    'Price Value (PV) & Social Influence (SI)   [Online platforms offer lower premiums and direct discounts due to zero agent commission.]': 'PV1',
    'Price Value (PV) & Social Influence (SI)   [Add-on covers (Zero Dep, Engine Protect) are transparently priced on digital platforms.]': 'PV2',
    'Price Value (PV) & Social Influence (SI)   [Recommendations from colleagues/friends in Bangalore influence my choice to buy online.]': 'SI1',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [I worry about lack of personal assistance during physical claim settlement without an agent.]': 'PR1',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [I feel secure entering my financial and vehicle details on online insurance portals.]': 'PR2',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [I trust digital-first insurers (e.g., Acko/Digit) as much as traditional state/private banks.]': 'PT1',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [Automated digital claim processing (self-inspection via mobile app) is reliable.]': 'PT2',
    
    # Emerging Trends 
    'Perception Toward Emerging Trends   [I would prefer a "Pay-As-You-Drive" policy where premium depends on kilometers driven.]': 'ET1',
    'Perception Toward Emerging Trends   [Mandatory Digilocker integration and IRDAI e-Insurance accounts make online policies easier to manage.]': 'ET2',
    'Perception Toward Emerging Trends   [I am comfortable sharing driving data via telematics/app if it gives me lower premiums.]': 'ET3'
}

LIKERT_DICT = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

DIGITAL_CHANNELS = [
    'Online Aggregator (e.g., PolicyBazaar)',
    'Bank Portal / UPI App (e.g., PhonePe, YONO SBI)',
    'Traditional Insurer Web Portal / App (e.g., Bajaj Allianz)'
]

ALL_LIKERT_COLS = ['PE1', 'PE2', 'EE1', 'EE2', 'PV1', 'PV2', 'SI1', 
                   'PR1', 'PR2', 'PT1', 'PT2', 'ET1', 'ET2', 'ET3']


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Loads dataset, cleans column names, filters validated sample, and engineers features."""
    df = pd.read_csv(filepath)
    
    # Strip whitespace from raw columns and mapping keys to prevent KeyErrors
    df.columns = df.columns.str.strip()
    clean_mapping = {k.strip(): v for k, v in COLUMN_MAPPING.items()}
    df = df.rename(columns=clean_mapping)
    
    # 1. Filter Validated Sample (N=232)
    df = df[df['Owns_Vehicle'].str.strip().str.lower() == 'yes'].copy()
    print(f"\n[*] Validated sample size: N = {len(df)}")
    
    # 2. Map Likert scales to numeric
    for col in ALL_LIKERT_COLS:
        if col in df.columns:
            df[col] = df[col].map(LIKERT_DICT)
            
    # 3. Engineer Outcome Variables
    df['Digital_Intent'] = df['Future_Renewal_Intent'].apply(
        lambda x: 1 if x in ['Continue using an Online Channel', 
                             'Switch completely to an Online Channel (Aggregator / Direct App)'] else 0
    )
    
    df['Recent_Channel'] = df['Recent_Channel_Raw'].apply(
        lambda x: 'Digital' if x in DIGITAL_CHANNELS else 'Offline'
    )
    
    # 4. Compute Composite Scores
    df['PE_score'] = df['PE1']
    df['EE_score'] = df[['EE1', 'EE2']].mean(axis=1)
    df['PV_score'] = df[['PV1', 'PV2']].mean(axis=1)
    df['PT_score'] = df[['PT1', 'PT2', 'PR2']].mean(axis=1)  # PR2 merged into PT based on PCA
    df['PR_score'] = df['PR1']
    df['SI_score'] = df['SI1']
    
    return df


def generate_descriptives(df: pd.DataFrame):
    """Generates the demographic and attitudinal frequencies for Table 2."""
    print("\n--- 1. TABLE 2: DEMOGRAPHIC & ATTITUDINAL PROFILING ---")
    n = len(df)
    
    # Age Groups
    gen_z = len(df[df['Age_Group'] == '18 – 25 years'])
    millennials = len(df[df['Age_Group'] == '26 – 40 years'])
    print(f"Gen Z (18-25 years): {gen_z} ({(gen_z/n)*100:.2f}%)")
    print(f"Millennials (26-40 years): {millennials} ({(millennials/n)*100:.2f}%)")
    
    # Vehicle Type
    evs = len(df[df['Vehicle_Type'].str.contains('Electric Vehicle', case=False, na=False)])
    print(f"Electric Vehicles (EV): {evs} ({(evs/n)*100:.2f}%)")
    
    # Attitudinal Barriers
    payd = len(df[df['ET1'] >= 4])
    anxiety = len(df[df['PR1'] >= 4])
    print(f"PAYD Support (Agree/Strongly Agree): {payd} ({(payd/n)*100:.2f}%)")
    print(f"Claim Anxiety (Agree/Strongly Agree): {anxiety} ({(anxiety/n)*100:.2f}%)")


def test_common_method_bias(df: pd.DataFrame):
    """Executes Harman's Single Factor Test for Common Method Bias."""
    print("\n--- 2. HARMAN'S SINGLE FACTOR TEST (CMB) ---")
    
    X = df[ALL_LIKERT_COLS].dropna()
    pca = PCA()
    pca.fit(X)
    
    variance_first_factor = pca.explained_variance_ratio_[0] * 100
    print(f"Total number of items tested: {X.shape[1]}")
    print(f"Variance explained by a single factor: {variance_first_factor:.2f}%")


def exploratory_factor_analysis(df: pd.DataFrame):
    """Validates structural constructs using PCA."""
    print("\n--- 3. PRINCIPAL COMPONENT ANALYSIS (FACTOR VALIDATION) ---")
    pca_cols = ['PE1', 'PE2', 'EE1', 'EE2', 'PV1', 'PV2', 'PR1', 'PR2', 'PT1', 'PT2']
    X_pca = df[pca_cols].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    pca = PCA(n_components=4, random_state=42)
    pca.fit(X_scaled)
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X_pca.columns)
    print("Factor Loadings (Extracting underlying constructs):")
    print(loadings.round(3))


def evaluate_reliability(df: pd.DataFrame):
    """Computes Construct Reliability for Table 1 (Alpha, Spearman-Brown, Inter-item r)."""
    print("\n--- 4. TABLE 1: PSYCHOMETRIC RELIABILITY ---")
    
    def calc_metrics(items_df, name):
        k = items_df.shape[1]
        
        # Cronbach's Alpha
        var_items = items_df.var(axis=0, ddof=1).sum()
        var_sum = items_df.sum(axis=1).var(ddof=1)
        alpha = (k / (k - 1)) * (1 - (var_items / var_sum)) if var_sum > 0 else np.nan
        
        # Pearson Inter-Item Correlation
        corr_matrix = items_df.corr()
        mean_r = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Spearman-Brown Prophecy
        spearman_brown = (k * mean_r) / (1 + (k - 1) * mean_r)
        
        print(f"[{name}] Alpha: {alpha:.4f} | Mean r: {mean_r:.4f} | Spearman-Brown: {spearman_brown:.4f}")

    calc_metrics(df[['EE1', 'EE2']], "Effort Expectancy (EE)")
    calc_metrics(df[['PV1', 'PV2']], "Price Value (PV)")
    calc_metrics(df[['PT1', 'PT2', 'PR2']], "Revised Perceived Trust (PT)")
    calc_metrics(df[['PE1', 'PE2']], "Initial Performance Expectancy (PE)")


def check_multicollinearity(df: pd.DataFrame, predictors: list):
    """Calculates Variance Inflation Factor (VIF) for the model predictors."""
    print("\n--- 5. MULTICOLLINEARITY (VIF) ---")
    X = df[predictors].dropna()
    X_with_const = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    print(vif_data.round(3))


def test_bivariate_independence(df: pd.DataFrame):
    """Runs Chi-Square and Fisher's Exact tests on Demographics vs Channel."""
    print("\n--- 6. BIVARIATE INDEPENDENCE (Ha1: AGE VS CHANNEL) ---")
    
    # Pearson Chi-Square
    contingency_table = pd.crosstab(df['Age_Group'], df['Recent_Channel'])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Pearson Chi-Square: {chi2:.4f}, p-value: {p_chi2:.4e}, df: {dof}")
    
    # Fisher's Exact Test (Aggregated Age)
    df['Age_Binary'] = df['Age_Group'].apply(lambda x: '<=40' if x in ['18 – 25 years', '26 – 40 years'] else '>40')
    fisher_table = pd.crosstab(df['Age_Binary'], df['Recent_Channel'])
    
    if fisher_table.shape == (2, 2):
        odds_ratio, p_fisher = stats.fisher_exact(fisher_table)
        print(f"Fisher's Exact Test (<=40 vs >40): OR = {odds_ratio:.4f}, p-value = {p_fisher:.4e}")


def run_econometric_models(df: pd.DataFrame, predictors: list):
    """Executes Logistic Regression and Linear Probability Models for Table 3."""
    print("\n--- 7. ECONOMETRIC MODELLING (TABLE 3) ---")
    
    model_data = df[['Digital_Intent'] + predictors].dropna()
    y = model_data['Digital_Intent']
    X = sm.add_constant(model_data[predictors])
    
    # A. MAXIMUM LIKELIHOOD LOGISTIC REGRESSION
    print("\n[ MAXIMUM LIKELIHOOD LOGISTIC REGRESSION ]")
    logit_model = sm.Logit(y, X).fit(disp=0)
    print(logit_model.summary())
    
    print("\n[ Table 3 Diagnostics & Odds Ratios ]")
    print(f"McFadden's Pseudo-R2: {logit_model.prsquared:.4f}")
    print(f"Log-Likelihood: {logit_model.llf:.2f}")
    print(f"LL-Null: {logit_model.llnull:.2f}")
    print(f"LLR p-value: {logit_model.llr_pvalue:.3e}")
    
    print("\nOdds Ratios (OR):")
    odds_ratios = np.exp(logit_model.params)
    print(odds_ratios.round(4))
    
    # B. LINEAR PROBABILITY MODEL (HC3 ROBUST SE)
    print("\n[ LINEAR PROBABILITY MODEL (HC3 ROBUST SE) ]")
    lpm_model = sm.OLS(y, X).fit(cov_type='HC3')
    print(lpm_model.summary())
    print(f"\nLPM R2: {lpm_model.rsquared:.3f}, F-stat: {lpm_model.fvalue:.3f}, p-value: {lpm_model.f_pvalue:.3e}")


def main():
    """Main execution pipeline."""
    # Define dataset path
    DATA_FILE = "survey_data_anonymized.csv"
    
    try:
        # Load and process data
        df = load_and_preprocess(DATA_FILE)
        
        # Run analytical steps
        generate_descriptives(df)
        test_common_method_bias(df)
        exploratory_factor_analysis(df)
        evaluate_reliability(df)
        
        # Predictors for Econometric Models
        predictors = ['PE_score', 'EE_score', 'PV_score', 'PR_score', 'PT_score', 'SI_score']
        
        check_multicollinearity(df, predictors)
        test_bivariate_independence(df)
        run_econometric_models(df, predictors)
        
        print("\n[*] Pipeline execution completed successfully. All manuscript tables replicated.")
        
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE}' was not found. Please ensure it is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()