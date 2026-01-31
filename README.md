# 💰 Lending Club Loan Default Prediction

📊 A deep learning project to predict loan defaults for LendingClub customers. Neural networks are applied to classify loans as either Fully Paid or Charged Off based on borrower details, loan attributes, and financial metrics

**Dataset Source:** [Lending Club Loan Dataset](https://www.kaggle.com/wordsforthewise/lending-club)


---

## 📁 Project Structure

* `Lending_Club_Loan_Default_Prediction.ipynb`: Python script containing the complete implementation of exploratory data analysis, data preprocessing, feature engineering, model training, and evaluation.
* `lending_club_loans.csv`: Raw dataset containing loan and borrower details such as loan amount, grade, annual income, and loan status.

---

## 📝 Dataset Overview

The dataset contains loan and borrower information with multiple features describing credit risk and repayment behavior:

* **Features:**
  * `loan_amnt`: The listed amount of the loan applied for by the borrower.
  * `int_rate`: Interest rate on the loan.
  * `installment`: Monthly payment owed by the borrower.
  * `grade` / `sub_grade`: LendingClub assigned loan grade (A-G) and sub-grade.
  * `home_ownership`: Home ownership status (RENT, OWN, MORTGAGE, OTHER).
  * `annual_inc`: Self-reported annual income provided by the borrower.
  * `verification_status`: Income verification status.
  * `loan_status`: Current status of the loan (Fully Paid or Charged Off).
  * `purpose`: Category provided by borrower for the loan request.
  * `dti`: Debt-to-income ratio.
  * `fico`: FICO credit score.
  * `mort_acc`: Number of mortgage accounts.
  * `total_acc`: Total number of credit lines currently in the borrower's credit file.

* **Target:** Binary classification to predict whether a loan will be `Fully Paid (1)` or `Charged Off (0)`.

---

## 📈 Key Insights

### Exploratory Data Analysis:

1. **Grade and Sub-Grade Analysis:**
   * Sub-grades F and G consistently show the highest default rates.
   * Lower grades (A, B, C) have significantly better repayment performance.

2. **Feature Correlations:**
   * Strong correlation between `loan_amnt` and `installment` (near-perfect).
   * Moderate correlations between `total_acc`, `mort_acc`, and loan repayment status.

3. **Loan Amount Patterns:**
   * Higher loan amounts are associated with increased default probability.
   * Charged-off loans have slightly higher average loan amounts compared to fully paid loans.

4. **Employment Length:**
   * No clear linear relationship between employment length and default rates.
   * Feature dropped after analysis showed minimal predictive value.


### Feature Engineering:

* **Binary Target Creation:** Mapped `loan_status` to `loan_repaid` (1 for Fully Paid, 0 for Charged Off).
* **Zip Code Extraction:** Extracted numeric zip codes from `address` strings.
* **Date Conversion:** Converted `earliest_cr_line` to numeric year values.
* **Term Conversion:** Converted `term` from " 36 months" to integer 36.
* **Home Ownership Consolidation:** Replaced rare categories (NONE, ANY) with OTHER to reduce dimensionality.

### Feature Scaling:

* Applied `MinMaxScaler` to normalize all numerical features to [0, 1] range.
* Essential for neural network optimization and gradient descent convergence.

### Categorical Encoding:

* One-hot encoded categorical variables:
  * `sub_grade` (A1-G5 loan quality ratings)
  * `home_ownership` (RENT, OWN, MORTGAGE, OTHER)
  * `verification_status` (Verified, Source Verified, Not Verified)
  * `application_type` (Individual, Joint)
  * `initial_list_status` (Whole, Fractional)
  * `purpose` (debt consolidation, credit card, home improvement, etc.)
  * `zip_code` (geographic risk factors)
* Used `drop_first=True` to avoid multicollinearity.

---

## 🧠 Neural Network Architecture

**Model Structure:**

* **Input Layer:** 78 features (after encoding and preprocessing)
* **Hidden Layer 1:** 78 neurons, ReLU activation, Dropout(0.2)
* **Hidden Layer 2:** 39 neurons, ReLU activation, Dropout(0.2)
* **Hidden Layer 3:** 19 neurons, ReLU activation, Dropout(0.2)
* **Output Layer:** 1 neuron, Sigmoid activation (binary classification)

**Training Configuration:**

* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Epochs:** 600 (with early stopping)
* **Batch Size:** 256
* **Early Stopping:** Patience of 5 epochs monitoring validation loss
* **Actual Training Duration:** 18 epochs (early stopping triggered)

**Regularization:**

* Dropout layers (20%) prevent overfitting
* Early stopping prevents unnecessary training once validation loss plateaus

---

## 📊 Visualizations

* **Count Plots:**
  * Target variable distribution (Fully Paid vs. Charged Off)
  * Grade and sub-grade distributions by loan status

* **Histograms:**
  * Loan amount distribution
  * Employment length distribution

* **Scatter Plots:**
  * Installment vs. Loan Amount relationship
  * Visualized strong linear correlation

* **Box Plots:**
  * Loan amounts by loan status
  * Identified outliers and median differences

* **Correlation Heatmap:**
  * Feature relationships and multicollinearity detection
  * Identified `loan_amnt` and `installment` perfect correlation

* **Performance Plots:**
  * Training vs. Validation Loss over epochs
  * Training vs. Validation Accuracy over epochs
  * Both showed smooth convergence without overfitting

* **Confusion Matrix:**
  * Evaluated true positives, false positives, true negatives, and false negatives
  * Assessed model's ability to correctly classify both loan outcomes

---

## 📉 Model Summary:

1. **Training Accuracy:** 88.15%
2. **Validation Accuracy:** ~88% (consistent performance)
3. **Training Duration:** 18 epochs (early stopping prevented overfitting)
4. **Train/Test Split:** 75% training, 25% testing
5. **Total Features:** 78 (after one-hot encoding)
6. **Regularization:** Dropout layers maintained stable validation performance

**Classification Report Metrics:**

* Precision, Recall, and F1-Score calculated for both classes (Fully Paid and Charged Off)
* Model demonstrates balanced performance across both loan outcomes

---

## 💡 Business Insights

1. **High-Risk Loan Identification:**
   * Sub-grades F and G require stricter approval criteria or higher interest rates
   * Borrowers with high debt-to-income ratios and low credit scores show elevated default risk

2. **Key Predictive Factors:**
   * Sub-grade rating (strongest indicator of loan quality)
   * Total accounts and mortgage accounts (credit history depth)
   * Loan amount and installment size (repayment burden)

---

## 🛠️ Tools and Libraries Used

* **Python:** Base language for all data analysis and modeling.
* **Key Libraries:**
  * `pandas` and `numpy`: Data manipulation and cleaning.
  * `seaborn` and `matplotlib`: Visualizations for EDA and model evaluation.
  * `tensorflow` and `keras`: Deep learning model architecture, training, and callbacks.
  * `scikit-learn`: Preprocessing (MinMaxScaler), train-test split, and metrics evaluation (confusion matrix, classification report).

---

## 📫 Contact

**LinkedIn:** [www.linkedin.com/in/kanishkayadvv](https://www.linkedin.com/in/kanishkayadvv)  
**Author:** Kanishka Yadav
