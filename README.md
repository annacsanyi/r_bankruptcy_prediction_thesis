## Corporate Bankruptcy Prediction Thesis
This repository contains the code and results from my master’s thesis, which examined the predictive performance of different machine learning models for forecasting corporate bankruptcies. The study combines firm-level financial data with industry and macroeconomic indicators, and introduces a stress testing framework to assess model robustness under crisis conditions.

### Project Overview
- Objective: Compare the effectiveness of traditional and machine learning models in predicting corporate bankruptcy.
- Data: Firm-level financial indicators, complemented by industry and macroeconomic variables.

### Models Tested:
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network
Evaluation Metrics: AUC, F1-score, recall, precision

### Key Findings
- XGBoost consistently outperformed the other models across most settings.
- Random Forest showed stable performance and fewer false positives, making it useful in regulatory contexts.
- Logistic Regression remained valuable for interpretability and recall, especially with macroeconomic variables.
- Neural Network performance was limited by class imbalance and technical constraints in R.
- The stress testing framework revealed how models react under economic crises (e.g., 2008, COVID-19), though it remained static in design.

### Contributions
- Demonstrates the strengths of ensemble learning in imbalanced classification problems.
- Introduces a replicable macroeconomic stress testing framework for bankruptcy prediction.
- Highlights the continuing importance of firm-level fundamentals in risk forecasting.

### Limitations & Future Directions
- Models were limited to a one-year prediction horizon.
- Static stress testing design could be expanded into dynamic, time-aware models (e.g., RNNs).
- Only accounting-based indicators were used; adding market-based data (equity volatility, credit spreads, etc.) could improve early warning.
- Extension to non-US firms would broaden applicability.

### Tools & Languages
- R (caret, randomForest, xgboost, tensorflow, ggplot2)
