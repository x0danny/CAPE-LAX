# CAPE-LAX
CAPE LAX — Carbon-Aware Predictive Engine

LAX Air Freight | Multimodal Carbon Risk Intelligence

Cal State LA · Society of Applied AI in Enterprise Systems (SAIES)
Daniel Ramirez · Brian Ta · Dr. Ming Wang (PI)


Overview

CAPE LAX is a machine learning framework that predicts high-carbon months at Los Angeles International Airport before they occur. Using 21 years of real BTS air freight data and 17 years of Port of LA container statistics, the model flags carbon surge risk 2 to 8 months in advance — enabling logistics operators, port authorities, and sustainability teams to act proactively rather than report reactively.

Key Finding: CAPE LAX flagged September 2020 as HIGH CARBON RISK at 68% confidence — 8 months before the 2021 supply chain surge peaked. The model was trained exclusively on pre-pandemic data (2006–2019) and had never seen conditions like it.


Results

MetricScoreTest Accuracy84%ROC-AUC (with Port of LA)0.931ROC-AUC (without Port of LA)0.922Walk-Forward Mean AUC0.911Precision (High Carbon Risk)0.95Early Warning Window8 months


Files

FileDescriptionCAPE_LAX_notebook.pyFull analysis pipeline — data loading, feature engineering, model training, validation, SHAP explainability, and chartscape_lax_app.pyStreamlit dashboard — model performance cards, corridor chart, live risk predictor, and dataset overviewmonthly_with_port.csvPre-merged monthly dataset combining BTS T-100 air freight records with Port of LA TEU data (required to run both files)


Data Sources


BTS T-100 International Segment — Bureau of Transportation Statistics. 1,706,566 total records, 41,820 LAX freight segments, 2006–2026. https://www.transtats.bts.gov
Port of LA Monthly TEU Statistics — Port of Los Angeles. Monthly container counts 2007–2026. https://portoflosangeles.org/business/statistics/container-statistics
ICAO CORSIA Emission Factor — 0.808 kg CO₂e per tonne-km for dedicated air freighters. Applied as: CO₂e (kg) = Freight Tons × Distance (km) × 0.808



How It Works

When the Port of LA gets congested, ships back up. Time-sensitive cargo gets rerouted to air. LAX volume spikes and carbon spikes with it. Port congestion today is a leading indicator of LAX carbon surges approximately 2 months later. CAPE LAX captures this relationship using a Random Forest classifier trained on 13 monthly features — 9 LAX signals and 4 Port of LA signals. SHAP analysis confirmed port volume 2 months prior as a top independent predictor, discovered by the model on its own.


Running the Dashboard

bashpip install streamlit pandas numpy plotly scikit-learn shap
streamlit run cape_lax_app.py

Requires monthly_with_port.csv in the same directory.


Running the Notebook

bashpip install pandas numpy matplotlib scikit-learn shap
python CAPE_LAX_notebook.py

Requires BTS T-100 CSV files (T100_2006.csv through T100_2026.csv) and monthly_with_port.csv in the same directory.


References


Bureau of Transportation Statistics. T-100 International Segment (All Carriers). https://www.transtats.bts.gov
Port of Los Angeles. Monthly Container Statistics. https://portoflosangeles.org/business/statistics/container-statistics
ICAO. CORSIA Carbon Emissions Calculator Methodology. https://www.icao.int/environmental-protection/CarbonOffset/Pages/default.aspx
Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 30.
