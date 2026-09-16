# FinSight-Capstone-Project
# FinSight Innovations: Data-Driven NSE Analysis

FinSight is an end-to-end quantitative finance and algorithmic trading pipeline designed for short-term prediction and predictive analytics on the Nairobi Securities Exchange (NSE). Built using the rigorous **CRISP-DM methodology**, the platform transforms raw market data across 14 distinct economic sectors into probability-calibrated, leakage-free directional forecasts.

While institutional systems cater heavily to long-term position management, FinSight specializes in filling the market gap for **short-term swing and day traders**. It features an automated data engineering pipeline tracking 28 technical indicators, advanced Bayesian hyperparameter tuning via Optuna and a high-performance gradient-boosted decision tree inference engine powered by CatBoost.

*This project was developed as a collaborative capstone effort by a team of 6 data science students and is deployed in production.*



## Key Architectural Pipeline
- **CRISP-DM Alignment:** Structured end-to-end framework flowing sequentially from Business Understanding, Data Cleaning, and Data Understanding (EDA) to robust Time-Series Modeling, Probability Calibration, and Latency Testing.

- **Feature Engineering Ensembles:** Automatic computation of 28 core quantitative indicators spanning Lagged Momentum, Trend-Following Volatility, Oscillators (RSI/MACD), and Liquidity/Volume ratios.

- **Leakage-Free Validation Engine:** Implements a strict chronological train-test division and a 5-fold Time-Series Cross-Validation wrapper to completely eliminate look-ahead bias and guarantee production generalizability.

- **Bayesian Optimization & Regularization:** Utilizes an integrated Optuna study maximizing the out-of-sample Area Under the Receiver Operating Characteristic (AUROC) curve while tuning L2 leaf regularization to handle financial noise.

- **Probability Accuracy:** Incorporates a Platt-scaling sigmoid calibration layer to ensure model output probabilities correspond strictly to mathematically sensible win frequencies (Brier Score minimization) for better real-world risk management.



## Project Directory Structure

```text
├── data/                                             # Encapsulated data layer
│   ├── NSE_data_all_stocks_2026_upto_jun.csv         # Raw historical price/volume dataset
│   ├── NSE_data_stock_market_sectors_2026.csv        # Sector matrix mappings
│   └── NSE_cleaned_stock_data_with_features.csv      # Unified, engineered analytical dataset
├── FinSight-app/                                     # Production application files and container scripts
│   ├── app.py                                        # Streamlit application dashboard entrypoint
│   └── requirements.txt                              # Specialized production freeze packages
├── catboost_info/                                    # CatBoost diagnostic training logs and loss tracking
├── FinSight.ipynb                                    # Unified EDA, optimization, feature selection, and modeling notebook
├── features.pkl                                      # Serialized list of production-selected feature names
├── model.pkl                                         # Final calibrated and exported predictive inference model
└── README.md                                         # Project documentation
```



##  Installation & Setup

1. **Clone the repository architecture:**
   ```bash
   git clone https://github.com/nelly-mbatha/FinSight-Capstone-Project.git
   cd FinSight-Capstone-Project
   ```

2. **Deploy the mathematical execution environment:**
   Make sure you are utilizing Python 3.9+ and run the following command to load the dependency framework:
   ```bash
   pip install pandas numpy scikit-learn catboost optuna joblib matplotlib seaborn
   ```



##  Pipeline Workflows & Execution

### 1. Data Processing and Feature Ingestion
The pipeline merges the transitional ticker tables with corporate sector classifications, enforces strict datetime indexing, sanitizes object types, handles missing market records, and constructs the 28-dimensional multi-indicator matrix.

### 2. Optimization and Training
To execute the hyperparameter tuning phase, run the core pipeline notebook:
* Open `FinSight.ipynb` and execute the cells sequentially. 
* The notebook handles Exploratory Data Analysis, cleans data anomalies, and initializes the automated Optuna study tracking 30 distinct trials. It systematically tunes optimal leaf depth, learning rates, and sample splits over chronological folds.
* Following optimization, the pipeline applies a `SelectFromModel` wrapper to prune the feature space down to the 10 most predictive, noise-resistant features, exporting the calibrated `model.pkl` and `features.pkl`.



## Production Cloud Deployment (Render)

The web application layer has been containerized and deployed to public production via the **Render** cloud platform infrastructure. The engine automatically handles ongoing web-request traffic and surfaces live predictions from the backend serial pipelines.

### Render Deployment Configuration:
- **Service Type:** Web Service
- **Environment:** `Python` (Python 3.9+)
- **Root Directory:** `FinSight-app` 
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py`

### CI/CD Deployment Integration:
The service is hooked directly into the GitHub repository `main` branch. Any incoming authorized pull-request merge triggering to `main` initiates a automated web-hook, triggering Render to rerun its build environment cleanly, execute validation tasks, and redeploy the live application seamlessly without downtime.



## Pipeline Evaluation Summary

The production configuration was subjected to rigorous validation metrics across the historical out-of-sample evaluation split:

- **Classification Quality:** Reached a Holdout AUROC of **0.6107**, significantly outperforming the ~0.56-0.58 market baseline, confirming a genuine mathematical edge in a highly volatile market.
- **Generalization Integrity:** Achieved an exceptional Generalization Variance of **-0.0172** (Holdout AUROC outperformed Cross-Validation). This proves that the Optuna-tuned L2 leaf regularizers completely neutralized model overfitting.
- **Probability Reliability:** Logged a stable Brier Score profile (**0.2466**), confirming that predictive score bounds remain mathematically reliable for institutional risk mitigation and position sizing.
- **Production Latency:** The specialized inference function processed the full 60-stock portfolio vector in **0.0186 seconds**, blowing past the < 2.0-second operational target to support micro-swing trading windows.



## Team & Cross-Functional Contributions

We collaborated using a structured Git feature-branch workflow to build, test, and merge components into the main production tree while safeguarding the integrity of the project branch.

| Team Member | Core Operational Focus |

| **Nelly Mbatha** | Business Understanding |
| **Nganga Mwaura** | Data Preparation |
| **Susan Mutiso** | Data Preparation |
| **Wilson Kingori** | Modeling |
| **Ahmeddin Abdulahi** | Modeling |
| **Denis Kamau** | Deployment |
