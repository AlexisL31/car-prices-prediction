# Car Prices Prediction

Building and evaluating three regression models that predict the selling price of used cars from their specifications. The project covers the full data science pipeline from data preparation through to model comparison, applying Linear Regression, Random Forest and Gradient Boosting to an 8,130 row dataset.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)

## Introduction

The growth of data driven decision making has allowed machine learning to become a critical tool when extracting value from structured datasets. In the automotive industry, predicting vehicle prices accurately is important for dealerships, buyers and online consumers as it supports pricing strategies, inventory management and consumer decision making. The aim of this project is to develop predictive models for estimating car selling prices using a structured dataset that contains vehicle attributes such as age, fuel type, transmission and mileage, framing the work as a supervised regression task that predicts a continuous target variable based on input features.

To address this, three machine learning models were implemented and evaluated which were Linear Regression, Random Forest Regressor and Gradient Boosting Regressor. These models were selected to compare a linear baseline against more advanced ensemble methods that are capable of capturing complex, non linear relationships within the data.

## Dataset

The dataset contains 8,130 records and 12 columns covering vehicle specifications, ownership and selling prices. It includes a mixture of numerical and categorical features which together describe the condition, history and market position of each car.

| | |
|---|---|
| Rows | 8,130 |
| Columns | 12 |
| Numerical features | year, selling_price, km_driven, mileage, engine, seats |
| Categorical features | name, fuel, seller_type, transmission, owner |
| Mixed (required cleaning) | max_power, stored as text with the unit suffix bhp |

The selling_price column was used as the target variable, while the remaining features were treated as the predictors after preprocessing.

## Data Preparation and Preprocessing

The dataset was cleaned and formatted to ensure suitability for machine learning modelling. Missing values were identified and handled using median imputation, which preserves the distribution of the features while preventing the bias that can arise from extreme values. The median was chosen over the mean because the affected variables such as engine size and max power are right skewed with outliers, and the median is more robust to that distribution.

The max_power column required additional cleaning because it was stored as an object with the unit suffix bhp and some blank entries. The suffix was removed, blank entries were replaced with NaN and the column was cast to float so that numerical operations could be performed on it. A unique value analysis was also carried out to check the variability of each feature. Although some categorical columns had only a few distinct values, no columns were removed because that low cardinality reflects the real world structure of vehicle data rather than a data quality issue.

Categorical variables such as fuel, seller type and transmission were encoded using dummy variable encoding. One category from each feature was dropped to prevent multicollinearity, ensuring that redundant information would not negatively affect the performance of each model. The dataset was then split into training and testing sets using an 80 to 20 ratio with a fixed random state to ensure reproducibility, providing 6,504 training samples and 1,626 testing samples across 16 features after encoding.

## Methodology

Three regression models were implemented to evaluate different approaches.

Linear Regression was used as the baseline because of its simplicity and interpretability, as it assumes a linear relationship between the features and the target variable. While efficient, its performance is limited when the underlying relationships are complex and non linear.

Random Forest Regressor is an ensemble learning method that constructs multiple decision trees and aggregates their predictions. This approach reduces variance and improves generalisation across the dataset. It is effective for capturing non linear relationships and interactions between features.

Gradient Boosting Regressor builds models sequentially, where each new model attempts to correct the errors of the previous ones. This process allows for high predictive accuracy, however it also increases the risk of overfitting if the parameters are not carefully controlled.

## Evaluation Metrics

The model performance was evaluated using three key metrics. Mean Absolute Error measures the average magnitude of prediction errors. Root Mean Squared Error penalises larger errors more heavily, making it useful for identifying models that produce large prediction deviations. R² score represents the proportion of variance in the target variable that is explained by the model. Together these metrics provide a thorough assessment of both the accuracy and reliability of the predictions.

## Results

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | −4.35 | 1,851,269 |
| Random Forest | 0.97 | 137,748 |
| Gradient Boosting | 0.96 | 160,308 |

Linear Regression produced poor results, with a negative R² score indicating that the model performed worse than a simple mean based prediction. The high RMSE further confirms that the prediction errors were significant. This suggests that the assumption of a linear relationship between the features and car prices is invalid for this dataset, as car pricing is influenced by many complex interactions between variables which a linear model cannot effectively capture.

Random Forest significantly outperformed Linear Regression, achieving an R² score of 0.97 and a much lower RMSE. This means the model is able to explain 97 percent of the variance in the target variable, demonstrating an extremely accurate predictive capability. The strong performance of Random Forest is due to its ensemble structure, which combines multiple decision trees to capture non linear relationships and reduce overfitting.

Gradient Boosting also showed strong performance, achieving an R² score of 0.96. While slightly lower than Random Forest, it still provides a highly accurate prediction. The small difference in performance can be due to its sensitivity to the parameters and the sequential nature of boosting, which can lead to overfitting if not tuned optimally.

## Comparing the Models

The results clearly show that the ensemble methods outperform the linear model for this task. Both Random Forest and Gradient Boosting effectively capture non linear relationships within the dataset, whereas Linear Regression cannot perform as well because of its simple structure. Between the two ensemble methods, Random Forest produced the best overall performance, suggesting it provides the optimal balance between accuracy and generalisation for this car prices dataset.

In addition to the headline numbers, several wider insights can be drawn from these results. Car prices are influenced by complex, non linear relationships between features, ensemble methods are significantly more effective than linear models for this kind of structured data, and model selection plays a critical role in predictive analysis. These findings align with existing research that highlights the effectiveness of tree based ensemble methods for structured data problems.

## Reflection

Working on this project gave me a clearer understanding of how machine learning models behave in practice, especially when applied to a real world dataset. At the start I assumed that building a model would mainly be about applying algorithms, but the data preparation and model selection also play a big role when analysing real world datasets.

One of the main challenges was dealing with the missing values and the categorical variables. Choosing median imputation rather than removing rows helped preserve the dataset while maintaining the stability of the data distribution. Another key learning point came from comparing the performance of the models. I expected Linear Regression to perform reasonably well, but the negative R² score showed the limitations of linear models when dealing with the complex, non linear relationships present in car pricing.

To improve the project further, I would focus on fine tuning the ensemble methods, creating new features such as car age or combining related variables to improve the prediction accuracy, and applying cross validation to ensure that the models generalise more reliably across different subsets of the data.

## Repository Contents

```
car-prices-prediction/
├── README.md
├── car_prices_prediction.ipynb
├── report.docx
├── requirements.txt
├── .gitignore
└── LICENSE
```

The dataset itself (CarPrices2026.csv) is not committed to this repository. Download it from Kaggle and place it at the project root before running the notebook.

## How to Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook car_prices_prediction.ipynb
```

## About Me

Built by Alexis Lim, an MSc Applied Data Science candidate at Royal Holloway.  I am actively looking for entry level Data Analyst and Analytics Intern roles in London.

LinkedIn: [alexis-lim-634325134](https://www.linkedin.com/in/alexis-lim-634325134/)
Email: a.lim10@hotmail.co.uk
