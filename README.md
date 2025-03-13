# Insurance Cost Prediction
This project is about predicting medical expenses in order to determine premiums , to assess the risks and to ensure profitability using machine learning models trained on the **Insurance**. Here, predicting medical expenses depend on the factors like age, bmi, smoking habits, gender, dependents and regions. The objective is to predict **Individual Charges**. The results will contribute to minimizing financial risks, improving risk assessment, determining premiums in the insurance sector and enhancing customer benefits.

## Dataset Overview
- **Source**: Insurance
- **Rows**: 1338
- **Columns**: 7 (including the target variable)
- **Target Variable**: 
  - charges
- **Features**:
  - age
  - sex
  - bmi
  - children
  - smoker
  - region
## Project Workflow
### 1. **Exploratory Data Analysis (EDA)**
- Data loading
- Data visualization using **Matplotlib & Seaborn**
### 2. **Feature Engineering**
- Checking missing values
- Removing duplicates
- Checking corrupted values
- Encoding categorical features(`OneHotEncoder & LabelEncoder`)
- Feature scaling (`MinMaxScaler`)
- Correlation
- Splitting dataset into training & testing sets (`train_test_split`)
  
### 3. **Machine Learning Models**
The project implements multiple machine learning algorithms & one deep learning algorithm for regression:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **K-Neighbors Regressor(KNN)**
- **Multi-layer Perceptron (MLP) Regressor**

### 4. **Hyperparameter Tuning**
- **RandomizedSearchCV** is used to optimize model parameters.

### 5. **Model Evaluation**
- **R2 Score**
- **Mean Squared Error**       
- **Mean Absolute Error**
- **Root Mean Squared Error** 

## Installation & Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.10 and above
- Jupyter Notebook
- Required libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)

### Running the Notebook
1. Clone the repository:
   git clone https://github.com/SPV-413/Insurance-Cost-Prediction.git
2. Navigate to the project folder:
   cd Insurance-Cost-Prediction
3. Open the Jupyter Notebook:
   jupyter notebook PRCP-1021-InsCostPred.ipynb

## Results
- **XGBoost Regressor** achieved the highest R2 Score.
- The project successfully demonstrates data preprocessing, model training, evaluation, and hyperparameter tuning.

## Contact
For any inquiries, reach out via GitHub or email.
