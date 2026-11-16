# A Study on the Adoption of Online Motor Insurance in Bengaluru: An Integrated UTAUT2 and Risk-Trust Analysis of Channel Inertia
This repository contains the replication code and anonymized primary dataset for the empirical research manuscript:
**"A Study on the Adoption of Online Motor Insurance in Bengaluru: An Integrated UTAUT2 and Risk-Trust Analysis of Channel Inertia."**

## Research Context
The non-life insurance sector in India is undergoing rapid digital disintermediation. However, a behavioural paradox persists in metropolitan hubs like Bengaluru: highly digitized consumers frequently utilize web aggregators for price discovery but revert to traditional offline intermediaries to execute high-stakes motor insurance contracts. 
Drawing on primary cross-sectional survey data from financially independent vehicle owners, this study integrates the **Unified Theory of Acceptance and Use of Technology (UTAUT2)** with a **Risk-Trust framework** to model the cognitive, economic, and relational mechanisms driving this channel inertia. 


**Methodological Framework**:
* **Sample:** Non-probability convenience sample (N = 232 validated responses) of independent motor vehicle owners in Bengaluru.
* **Econometric Models:** Maximum Likelihood Logistic Regression (Logit) paired with a heteroskedasticity-consistent Linear Probability Model (LPM with White's HC3 robust standard errors).
* **Psychometrics:** Exploratory Factor Analysis (PCA), Harman's Single-Factor Test (CMB), and Construct Reliability (Cronbach's Alpha, Spearman-Brown prophecy, Pearson inter-item correlation).

## Repository Structure
* `Research_Analysis_Script.py` - The Python script containing the end-to-end analytical and econometric pipeline.
* `survey_data_anonymized.csv` - The raw, anonymized dataset containing 238 initial responses (which the script automatically filters down to the 232 validated independent vehicle owners used in the study).


## Replication Instructions
To replicate the exact tables, diagnostic metrics, and structural coefficients reported in the manuscript, follow these steps:

**1**. **Prerequisites**
Ensure you have Python 3.8 or higher installed. The script relies entirely on standard academic data science libraries.
You can install the required dependencies using pip:
```bash
pip install pandas numpy scipy statsmodels scikit-learn
```

**2**. **Execution**
Place both Research_Analysis_Script.py and survey_data_anonymized.csv in the same working directory.
Run the script via your terminal or IDE:
```bash
python insurance_adoption_analysis.py
```

**3**. **Output Mapping to Manuscript**
The script will output cleanly formatted console text that directly maps to the manuscript's findings:
* **Demographic & Attitudinal Profiling:** Replicates the exact frequencies and percentages for Table 2 (e.g., Gen Z, Millennials, EV adoption, PAYD telematics support).
* **Harman's Single Factor Test:** Confirms the absence of Common Method Bias (variance < 50%).
* **Exploratory Factor Analysis:** Displays PCA factor loadings.
* **Psychometric Reliability:** Replicates Table 1, calculating Cronbach's Alpha, Spearman-Brown coefficients, and Pearson inter-item correlations for all subscales.
* **Multicollinearity Diagnostics:** Confirms all Variance Inflation Factors (VIF) are strictly < 3.3.
* **Bivariate Independence:** Tests Ha1 using Pearson Chi-Square and Fisher's Exact tests.
* **Econometric Modelling:** Replicates Table 3. It outputs the Maximum Likelihood Logit summary, exact McFadden's Pseudo-R², Log-Likelihood metrics, Odds Ratios (OR), and the corroborating LPM HC3 summary.


**Authors**
Divyam Kumar - divyam0718@gmail.com
Rimjhim Kumari - rimjhim98455@gmail.com 

License & Declarations
* License: This code is released under the MIT License.
* Data Privacy: The dataset (survey_data_anonymized.csv) has been stripped of all personally identifiable information (PII) to preserve respondent anonymity in accordance with ethics guidelines.
* DOI: https://doi.org/10.5281/zenodo.24271443
