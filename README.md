# AI Studio Challenge Project:
# Incrementality Modeling for Privacy-Conscious Retail Media and Data Clean Room Measurement
---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Anne Argonza     | @ 
| Adam Le          | @    
| Sumant Munagala  | @             
| Penny Wang       | @pennywa      |        

---

## 🎯 **Project Highlights**

- Developed causal uplift models (**Two-Model T-Learner** and **Transformed Outcome**) alongside **Random** and **Response Prediction** baselines to measure true incremental advertising impact for retail media campaigns.
- Achieved **+6.11% Decile 1 incremental visit lift** (916 net visits, 95% CI [4.20%, 8.02%]) with the Transformed Outcome model and **+5.83% lift** (874 net visits, 95% CI [3.91%, 7.75%]) with the Two-Model T-Learner on the sealed 150,000-row held-out test set, outperforming the empirical Random Baseline (+1.67% lift).
- Generated actionable business and privacy recommendations for **SolePeak Footwear** (advertiser) and **MarketBridge Retail Media** (retail media network), showing why concentrating spend on top-ranked uplift deciles avoids the "always-buyer" trap.
- Implemented an **Intention-to-Treat (ITT)** causal evaluation framework across anonymized features (`f0`–`f11`), resolving an official statistical tie between causal models via overlapping 95% Wald confidence intervals and selecting the T-Learner for production stability under 85/15 treatment imbalance.

---

## 👩🏽‍💻 **Setup and Installation**

### Clone the Repository
```bash
git clone [https://github.com/Break-Through-Tech/TBD_Retail-Media-1A-incrementality-modeling-for-privacy-conscious-retail-media.git](https://github.com/Break-Through-Tech/TBD_Retail-Media-1A-incrementality-modeling-for-privacy-conscious-retail-media.git)
cd TBD_Retail-Media-1A-incrementality-modeling-for-privacy-conscious-retail-media
```
------------------------------------------------------------------------------------------------------------------------
### Set Up the Environment & Install Dependencies

#### Option A: Local Machine (Recommended)
1. **Create a virtual environment** inside the project root directory:
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
     *(If PowerShell blocks the script, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then activate again.)*
   * **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### Option B: Google Colab
If running in Google Colab, clone the repository directly into your Colab session (all core dependencies—`pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`—are pre-installed):
```python
!git clone [https://github.com/Break-Through-Tech/Retail-Media-1B-incrementality-modeling-for-privacy-conscious-retail-media.git](https://github.com/Break-Through-Tech/Retail-Media-1B-incrementality-modeling-for-privacy-conscious-retail-media.git)
%cd Retail-Media-1B-incrementality-modeling-for-privacy-conscious-retail-media
```
------------------------------------------------------------------------------------------------------------------------
### Access the Dataset

The repository already includes a pre-processed, reproducible 1,000,000-record random sample (`seed=2026`) of the unbiased **Criteo Uplift Prediction Dataset (v2.1)** inside the `data/` directory—no external downloads or manual unzipping are required.

| File | Purpose | Records | Compressed Size |
| :--- | :--- | :--- | :--- |
| `data/train.csv.gz` | Train models | 700,000 | ~17.3 MB |
| `data/validation.csv.gz` | Compare approaches and tune hyperparameters | 150,000 | ~3.7 MB |
| `data/test.csv.gz` | Perform final held-out evaluation | 150,000 | ~3.7 MB |

`pandas` reads the gzip-compressed CSV files directly:

```python
import pandas as pd

train = pd.read_csv("data/train.csv.gz")
validation = pd.read_csv("data/validation.csv.gz")
test = pd.read_csv("data/test.csv.gz")
```

*(Note: For full source attribution, sampling methodology, and license terms under **CC BY-NC-SA 4.0**, see the documentation inside the `data/` folder.)*
------------------------------------------------------------------------------------------------------------------------
### Run the Notebooks

To reproduce the held-out test scoring and evaluation end-to-end, run `november_held_out_test_scoring_and_evaluation.ipynb` from top to bottom:

#### Option A: Interactive Execution (VS Code / Jupyter / Google Colab)
1. Open `november_held_out_test_scoring_and_evaluation.ipynb` in VS Code, Cursor, JupyterLab, or Google Colab.
2. Select your `.venv` Python kernel (or Google Colab runtime).
3. Click **Restart Kernel and Run All Cells** (or **Runtime → Run all** in Colab) to execute all cells sequentially with deterministic seeds (`seed=42`).

#### Option B: Command-Line Execution
Execute the notebook headlessly from the repository root and save the outputs in place:
```bash
jupyter nbconvert --to notebook --execute --inplace november_held_out_test_scoring_and_evaluation.ipynb
```
*(Model score arrays are automatically saved to `scores/` and evaluation figures are generated inline.)*
------------------------------------------------------------------------------------------------------------------------

## 🏗️ **Project Overview**

- **Connection to the Break Through Tech AI Program:** This project was completed during the Fall 2026 **Break Through Tech AI Studio** semester as an Independent Industry Challenge designed and advised by product leader **Annamalai Annamalai**. It bridges academic machine learning theory with real-world privacy-safe advertising measurement constraints.
- **Host Context, Objective & Scope:** Framed around a fictional teaching scenario between **SolePeak Footwear** (an athletic footwear advertiser) and **MarketBridge Retail Media** (a retailer operating an ecommerce storefront and advertising network), the project models the analytical stage of a privacy-conscious **Data Clean Room** workflow. Using an anonymized sample of the Criteo Uplift Prediction Dataset as an analytical proxy, our objective is to rank audience records by estimated **incremental visit lift** under an **Intention-to-Treat (ITT)** experimental design rather than raw conversion likelihood. Upstream identity matching, user consent governance, and overlap resolution are assumed complete and lie outside project scope; all public reporting is strictly restricted to aggregate cohort metrics.
- **Real-World Significance & Impact:** Conventional response models target high-intent shoppers who would visit organically regardless of ad exposure—causing advertisers to subsidize pre-existing demand (the "always-buyer" trap). By isolating persuadable audiences whose behavior changes *because* of campaign assignment without relying on personally identifiable information (PII), causal uplift modeling improves marketing Return on Ad Spend (ROAS) for advertisers like SolePeak while allowing retail media networks like MarketBridge to validate and price genuinely incremental ad inventory.
---

## 📊 **Data Exploration**

- **Dataset Origin, Format & Size:** We used a reproducible 1,000,000-row random sample (`seed=2026`, ~24.7 MB compressed as `.csv.gz`) drawn across the entire 13,979,592-row unbiased **Criteo Uplift Prediction Dataset (v2.1)**. Scanning the complete ordered source file was essential because the first control record does not appear until source row 5,571,800. The sample is partitioned into `train.csv.gz` (700,000 rows), `validation.csv.gz` (150,000 rows), and `test.csv.gz` (150,000 rows), preserving consistent feature distributions and outcome rates across splits.
- **Data Exploration & Preprocessing:**
  - **Input Features (`f0`–`f11`):** 12 anonymized, continuous numerical behavioral signals containing zero personally identifiable information (PII).
  - **Experimental & Outcome Columns:** `treatment` (binary group assignment), `exposure` (post-assignment ad view indicator), `visit` (primary binary outcome), and `conversion` (optional secondary binary outcome).
  - **Data Readiness & Leakage Prevention:** Validation confirmed zero missing values, zero infinite values, zero exposed control records, and zero conversions recorded without a visit. `treatment`, `exposure`, `visit`, and `conversion` were strictly stripped from the feature matrix to prevent target and post-treatment leakage.
- **Key Insights from EDA:**
  - **Severe Treatment Imbalance:** Across the 1,000,000-row sample, **849,953 records** (85.0%) are assigned to treatment and **150,047 records** (15.0%) to holdout control.
  - **Outcome Rarity:** Overall visits occur in **46,939 records** (~4.69% rate), whereas conversions occur in only **2,839 records** (~0.28% rate)—justifying `visit` as our primary modeling target and `conversion` as a descriptive secondary check.
- **Challenges & Key Assumptions:**
  - **Intention-to-Treat (ITT) Integrity:** Because `exposure` occurs after ad auction assignment, filtering primary analysis to `exposure = 1` would introduce post-treatment selection bias. All models evaluate treatment assignment (`treatment`), not physical banner viewability.
  - **Anonymity & Clean Room Boundaries:** Features `f0`–`f11` have no assigned business meanings, and the lack of user-level join keys restricts conclusions strictly to aggregate cohort metrics rather than individual user exports.

---

## 🧠 **Model Development**

- **Models Implemented:** We developed and compared four distinct ranking approaches:
  1. **Random Ranking Baseline:** Assigns deterministic random scores (`seed=42`) to establish the empirical null lift benchmark across audience deciles.
  2. **Response Prediction Baseline:** A standard classification model trained exclusively on treated records ($W=1$) to predict visit probability $P(\text{visit} = 1 \mid X)$, representing traditional targeting that ranks users by raw conversion likelihood rather than incremental lift.
  3. **Two-Model T-Learner ($C=0.01$):** Fits two independent $L_2$-regularized logistic regression estimators—one on the treated group ($\mu_1(X) = P(Y=1 \mid X, W=1)$) and one on the control group ($\mu_0(X) = P(Y=1 \mid X, W=0)$)—calculating estimated individual uplift as $\hat{\tau}(X) = \mu_1(X) - \mu_0(X)$.
  4. **Transformed Outcome Model ($\alpha=100$):** Computes empirical treatment propensity $p = P(W=1) \approx 0.85$ from the training set (avoiding the invalid $p=0.5$ assumption) and transforms the binary visit outcome into a single continuous target:
     $$Z = \text{visit} \cdot \frac{\text{treatment} - p}{p(1 - p)}$$
     A Ridge regression model ($\alpha=100$) is trained directly on $Z$ to estimate uplift in a single prediction step.

- **Feature Selection & Hyperparameter Tuning:**
  - **Strict Feature Restriction:** Input features were restricted exclusively to the 12 anonymized continuous signals (`f0`–`f11`). `treatment`, `exposure`, `visit`, and `conversion` were excluded to eliminate target leakage and post-assignment bias.
  - **Regularization Tuning:** Because the 85% / 15% treatment-to-control split leaves sparse control observations and magnifies inverse-propensity variance when dividing by $(1 - p) \approx 0.15$, hyperparameters were tuned on `validation.csv.gz` to prioritize variance stability—selecting $C=0.01$ for the T-Learner and $\alpha=100$ for the Transformed Outcome Ridge model before freezing all model parameters.

- **Training Setup & Evaluation Metrics:**
  - **Data Split:** Models were fitted on `train.csv.gz` (70% / 700,000 rows), compared and selected using `validation.csv.gz` (15% / 150,000 rows), and locked prior to touching `test.csv.gz` (15% / 150,000 rows).
  - **Primary Evaluation Metrics:** Models were evaluated by sorting audiences from highest to lowest predicted score (with deterministic tie-breaking), partitioning into 10 equal deciles, and computing **Decile 1 Incremental Visit Lift** ($\text{Visit Rate}_T - \text{Visit Rate}_C$), **95% Wald Confidence Intervals**, **Estimated Net Incremental Visits**, and **Cumulative Uplift Curves**.

---

## 📈 **Results & Key Findings**

### 1. Held-Out Test Set Performance (`data/test.csv.gz` — Decile 1)
All four models were frozen prior to touching the 150,000-row held-out test set. Records were ranked from highest to lowest score and segmented into 10 equal audience deciles (15,000 records per decile) to evaluate top-decile **Intention-to-Treat (ITT) Incremental Visit Lift** ($\text{Visit Rate}_T - \text{Visit Rate}_C$) with 95% Wald confidence intervals:

| Model | Treatment Count | Control Count | Visit Rate (T) | Visit Rate (C) | Decile 1 Uplift | 95% Wald CI | Est. Incremental Visits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Transformed Outcome ($\alpha=100$)** | 12,971 | 2,029 | 26.41% | 20.31% | **+6.11%** | [4.20%, 8.02%] | **916.08** |
| **T-Learner ($C=0.01$)** | 12,962 | 2,038 | 26.73% | 20.90% | **+5.83%** | [3.91%, 7.75%] | **874.37** |
| **Response Baseline** | 12,891 | 2,109 | 36.01% | 30.54% | **+5.47%** | [3.34%, 7.61%] | **821.07** |
| **Random Baseline** | 12,772 | 2,228 | 4.90% | 3.23% | **+1.67%** | [0.85%, 2.49%] | **250.46** |

### 2. Model Comparison & Statistical Tie Resolution
- **Causal Models Outperform Random Targeting:** Both causal estimators nearly quadrupled the top-decile incremental lift of the empirical Random Baseline (+6.11% and +5.83% vs. +1.67%), capturing roughly 874 to 916 net incremental site visits within the top 10% audience segment alone.
- **Official Statistical Tie:** While the Transformed Outcome model achieved a slightly higher point estimate (+6.11% vs. +5.83%), their 95% Wald confidence intervals overlap substantially between **4.20% and 7.75%**. Because sampling uncertainty prevents declaring a statistically distinct winner, the two causal approaches are documented as an official statistical tie.
- **Production Selection (Two-Model T-Learner):** We recommend deploying the **Two-Model T-Learner** in production. Fitting separate regularized estimators for treatment and control groups prevents inverse-propensity variance spikes ($1 / (1 - p)$) if real-world auction win rates shift away from the 85/15 split.

### 3. Business, Fairness & Privacy Insights
- **Exposing the "Always-Buyer" Trap:** The Response Baseline selected users with a **30.54% organic control visit rate** (compared to ~20.3%–20.9% for the causal models), confirming that conventional response prediction wastes ad budget targeting high-intent shoppers who would visit without advertising.
- **Targeted Spend Over Broad Reach:** Across deciles, incremental visit lift concentrates heavily in Deciles 1 and 2 before decaying toward random noise (+1.67%). SolePeak Footwear should restrict campaign budgets to top-ranked deciles rather than broad audience reach, while MarketBridge Retail Media can package top uplift deciles as premium high-incrementality inventory.
- **Downstream Conversion Validation:** Descriptive checks on secondary `conversion` rates confirm healthy checkout activity within top visit-uplift deciles, demonstrating that incremental site visits translate into genuine commercial intent without distorting model rankings on rare conversion labels.
- **Privacy & Anonymity Guardrails:** Because features `f0`–`f11` are anonymized and contain zero demographic or user-level join keys, fairness and performance evaluations are strictly scoped to aggregate cohort behavior—individual model scores are never treated as privacy-safe audience exports.

---

## 🚀 **Next Steps**

- **Model & Dataset Limitations:**
  - **Intention-to-Treat (ITT) vs. Ad Exposure:** Our models estimate the causal impact of *campaign eligibility* (`treatment`) rather than actual banner viewability (`exposure`), meaning lift estimates reflect auction delivery rates alongside creative persuasiveness.
  - **Anonymized Feature & Clean Room Boundaries:** Because `f0`–`f11` lack business definitions and user-level join keys, our pipeline cannot evaluate upstream identity match rates, cross-device resolution, user consent opt-ins, or physical in-store sales attribution. Additionally, individual uplift scores are strictly analytical outputs and cannot be exported as privacy-safe user lists without clean room aggregation guardrails.
  - **Linear Estimator Assumptions:** Both the regularized T-Learner ($C=0.01$) and Transformed Outcome Ridge ($\alpha=100$) models rely on linear decision boundaries that may miss complex non-linear feature interactions across customer segments.

- **Improvements with More Time & Resources:**
  - **Non-Linear Causal Meta-Learners:** With additional compute and cross-validation bandwidth, we would benchmark tree-based and neural uplift estimators—such as Causal Forests, X-Learners, and Dragonnet—to capture higher-order behavioral interactions while applying calibration techniques (e.g., isotonic regression) to control probability variance.
  - **End-to-End Clean Room Governance Simulation:** Partner with data engineering stakeholders to integrate cryptographic hashed identity matching, differential privacy noise injection, and minimum cohort aggregation thresholds ($k$-anonymity) upstream of model scoring.

- **Additional Datasets & Techniques to Explore:**
  - **Conversion- & Revenue-Aware Multi-Task Modeling:** Incorporate richer downstream purchase signals (`conversion` and order value) using two-stage hurdle models or expected value optimization to rank audiences directly by incremental Return on Ad Spend (ROAS) rather than site visits alone.
  - **Multi-Touch & Omnichannel Retail Data:** Combine digital ad auction logs with retailer point-of-sale (POS) loyalty data inside a clean room to measure cross-channel halo effects between online ad impressions and physical store foot traffic.
  - **Dynamic Propensity Drift Monitoring:** Implement automated Population Stability Index (PSI) tracking on features `f0`–`f11` and rolling treatment assignment rates ($p$) to trigger recalibration when seasonal auction competition shifts.

---

## 📝 **License**

- **Dataset:** The prepared dataset files (`data/train.csv.gz`, `data/validation.csv.gz`, `data/test.csv.gz`) are shared for noncommercial educational use under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, following [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)'s distribution terms for the [Criteo Uplift Prediction Dataset (unbiased version 2.1)](https://huggingface.co/datasets/criteo/criteo-uplift). The files represent a reproducible 1,000,000-record random sample (`seed=2026`) partitioned into train, validation, and test splits; Criteo does not endorse this project or its conclusions.
- **Code & Documentation:** Shared for noncommercial educational use in alignment with the repository's terms (see `data/` for full dataset licensing and attribution details).

---

## 📄 **References**

1. Diemert, E., Betlei, A., Renaudin, C., Amini, M.-R., Gregoir, T., & Rahier, T. (2021). *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling*. [arXiv:2111.10106](https://arxiv.org/abs/2111.10106).
2. Criteo AI Lab. *Criteo Uplift Prediction Dataset (Unbiased Version 2.1)*. [Dataset Documentation](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | [Hugging Face Distribution](https://huggingface.co/datasets/criteo/criteo-uplift).

---

## 🙏 **Acknowledgements**

- **Challenge Advisor:** Thank you to **Annamalai Annamalai** for designing this Independent Industry Challenge, preparing the reproducible dataset splits, and guiding our team through causal uplift modeling, Intention-to-Treat (ITT) evaluation, and privacy-conscious Data Clean Room measurement.
- **AI Studio Coach:** Thank you to **Anushka Naik** for her mentorship, feedback, and project management support throughout the Fall 2026 semester.
- **Break Through Tech AI:** Thank you to the **Break Through Tech AI** team at Cornell Tech for their guidance and support throughout the program.
- **Data Attribution:** Thank you to **Criteo AI Lab** for releasing the unbiased Criteo Uplift Prediction Dataset (v2.1) for noncommercial research and educational use.
