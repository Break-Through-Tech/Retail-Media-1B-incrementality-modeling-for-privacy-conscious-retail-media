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

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
