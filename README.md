# 💰 Lending Club Loan Default Prediction

📊 A machine learning project aimed at predicting loan defaults for Lending Club customers. The goal is to identify whether a loan will be `Fully Paid` or `Charged Off` using loan attributes, borrower details, and financial metrics.

---

## 👨‍💻 Project Overview

This project leverages Deep Learning (Neural Networks) alongside feature engineering and thorough data preprocessing to predict the loan repayment status. It explores patterns in the dataset, cleans the data, performs scaling and encoding, and uses TensorFlow to build a fully functional predictive model.

---

## 📁 Project Structure

* **`lending_club_loans.csv`**: Dataset containing loan and borrower details such as `loan_amnt`, `grade`, `annual_income`, `loan_status`, etc.
* **`lending_club_model.ipynb`**: Full implementation with exploratory data analysis (EDA), preprocessing, normalization, model creation, training, and evaluation.

---

## 📊 Exploratory Data Analysis (EDA)

Key visualizations and data patterns:
1. **Target Balance**:
   - Distribution of `Fully Paid` vs `Charged Off` loans using count plots.
2. **Loan Amount Insights**:
   - Analyzed the distribution of loan amounts and their correlations with payment status.
   - Used scatterplots, boxplots, and aggregated statistics (`groupby`).
3. **Grade and Sub-Grade Analysis**:
   - Visualized high-risk sub-grades (`F` and `G`) with count plots.
   - Identified grades that have higher default probabilities.
4. **Correlation Heatmap**:
   - High correlation between `loan_amnt` and `installment`.

---

## 🛠️ Data Cleaning and Preprocessing

### Key preprocessing steps included:
1. **Handling Missing Values**:
   - High-cardinality columns like `emp_title` and `title` dropped.
   - Median imputation for `mort_acc` and `credit_limit` based on correlated features.
2. **Feature Engineering**:
   - Extracted zip codes from addresses for feature reduction.
   - Mapped `loan_status` to binary values (`0` for `Charged Off` and `1` for `Fully Paid`).
3. **Encoding**:
   - One-hot encoding of categorical variables (e.g., `home_ownership`, `verification_status`, `sub_grade`).
4. **Standardization**:
   - Used `MinMaxScaler` to normalize numerical features between 0 and 1.
5. **Train-Test Split**:
   - 75% train, 25% test split on preprocessed data.

---

## 🔢 Neural Network Model

### Architecture:
- Input: 78-dimensional numerical vectors corresponding to preprocessed features.
- **Layers**:
  - Dense(78, activation='relu') + Dropout(0.2)
  - Dense(39, activation='relu') + Dropout(0.2)
  - Dense(19, activation='relu') + Dropout(0.2)
  - Dense(1, activation='sigmoid') — Output for binary classification.

### Training:
- **Adam Optimizer** used for efficient gradient descent.
- **Binary Crossentropy Loss** for classification.
- Implemented **EarlyStopping** to halt training when validation loss stagnated.

---

## 📈 Model Performance

### Training Results:
1. **Training Accuracy**: ~88.15%
2. **Validation Loss**: Successfully reduced during 18 epochs using EarlyStopping.
3. **Key Metrics**:
   - Accuracy remained steady across train and validation datasets, with well-regularized training thanks to dropouts.

---

## 📊 Key Insights and Visuals

1. **Correlation Heatmap**:
   - `loan_amnt` and `installment` highly correlated.
   - Moderate correlations with other repayment predictors such as `mort_acc` and `total_acc`.

2. **High-Risk Customers**:
   - `F` and `G` sub-grades consistently showed the worst repayment performance.

3. **Loan Repayment Trends**:
   - Higher loan amounts and risk grades (`F`, `G`) were linked with more defaults.
   - Borrowers with shorter employment lengths (<3 years) tended to default more often.

4. **Performance Plots**:
   - Loss and accuracy visualized across epochs to ensure optimal training.

---

## 🛠️ Tools and Technologies

- **Python**: Base language for data analysis and modeling.
- **Libraries and Frameworks**:
  - **Pandas**: Data manipulation and cleaning.
  - **Seaborn & Matplotlib**: Exploratory Data Analysis and visualizations.
  - **TensorFlow & Keras**: Deep learning model design, training, and evaluation.
  - **Scikit-learn**: Preprocessing, scaling, and metrics evaluation.

---

## 🚀 Future Work

1. **Addressing Class Imbalance**:
   - Techniques like oversampling (SMOTE) or weighting class balance can improve default predictions.
2. **Advanced Feature Engineering**:
   - Incorporate external financial indicators for better risk prediction.
3. **Deploying the Model**:
   - Package the model as a REST API for real-time predictions.

---

## 📂 Dataset

The Lending Club Loans dataset contains anonymized data related to customer loans, loan statuses, and financial attributes. For more information, refer to [LendingClub Loan Dataset](https://www.kaggle.com).

---

📫 **Contact**: [www.linkedin.com/in/kanishkayadvv](https://www.linkedin.com/in/kanishkayadvv)  
**Author**: Kanishka Yadav
