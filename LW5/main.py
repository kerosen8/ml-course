import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor



plt.figure(figsize=(10, 6))

def detect_outliers(column):
    column = np.array(column)
    
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_indices = np.where((column < lower_bound) | (column > upper_bound))[0]
    
    return outliers_indices

df = pd.read_csv("../data/OnlineNewsPopularityReduced.csv", delimiter=',')

X_1 = df[['n_non_stop_unique_tokens', 'timedelta', 'n_tokens_title', 'average_token_length', 'LDA_03', 'num_imgs', 'num_videos']]
y_1 = df['shares']

outlier_indices_shares = detect_outliers(df['shares'])
outlier_indices_average_token_length = detect_outliers(df['average_token_length'])
outlier_indices_n_non_stop_unique_tokens = detect_outliers(df['n_non_stop_unique_tokens'])

combined_outlier_indices = np.unique(np.concatenate((outlier_indices_shares, outlier_indices_average_token_length, outlier_indices_n_non_stop_unique_tokens)))

X = X_1.drop(index=combined_outlier_indices).reset_index(drop=True)
y = y_1.drop(index=combined_outlier_indices).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 1

base_model = DecisionTreeRegressor(max_depth=4, random_state=42)

n_estimators_range = np.arange(10, 210, 20)
learning_rate_range = np.logspace(-2, 0.5, 10)

ada_boost = AdaBoostRegressor(estimator=base_model, random_state=42)

ada_boost.fit(X_train, y_train)
y_test_pred = ada_boost.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(f"MSE for AdaBoost: {mse}")

train_scores_n, test_scores_n = validation_curve(
    ada_boost, X_train, y_train,
    param_name="n_estimators", param_range=n_estimators_range,
    scoring="neg_mean_squared_error", cv=5
)

train_scores_lr, test_scores_lr = validation_curve(
    ada_boost, X_train, y_train,
    param_name="learning_rate", param_range=learning_rate_range,
    scoring="neg_mean_squared_error", cv=5
)

train_mean_n = -np.mean(train_scores_n, axis=1)
test_mean_n = -np.mean(test_scores_n, axis=1)
train_mean_lr = -np.mean(train_scores_lr, axis=1)
test_mean_lr = -np.mean(test_scores_lr, axis=1)

plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, train_mean_n, label="Train Score", marker='o')
plt.plot(n_estimators_range, test_mean_n, label="Test Score", marker='o')
plt.title("Validation Curve for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(learning_rate_range, train_mean_lr, label="Train Score", marker='o')
plt.plot(learning_rate_range, test_mean_lr, label="Test Score", marker='o')
plt.title("Validation Curve for learning_rate")
plt.xlabel("learning_rate")
plt.xscale("log")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

### 2

base_model = DecisionTreeRegressor(max_depth=4, random_state=42)

n_estimators_range = np.arange(10, 210, 20)
learning_rate_range = np.logspace(-2, 0.5, 10)

gradient_boost = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
)

gradient_boost.fit(X_train, y_train)
y_test_pred = gradient_boost.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(f"MSE for Gradient Boosting: {mse}")

# n_estimators val curve
train_scores_n, test_scores_n = validation_curve(
    gradient_boost, X_train, y_train,
    param_name="n_estimators", param_range=n_estimators_range,
    scoring="neg_mean_squared_error", cv=5
)

# learning_rate val curve
train_scores_lr, test_scores_lr = validation_curve(
    gradient_boost, X_train, y_train,
    param_name="learning_rate", param_range=learning_rate_range,
    scoring="neg_mean_squared_error", cv=5
)

train_mean_n = -np.mean(train_scores_n, axis=1)
test_mean_n = -np.mean(test_scores_n, axis=1)
train_mean_lr = -np.mean(train_scores_lr, axis=1)
test_mean_lr = -np.mean(test_scores_lr, axis=1)

# n_estimators val curve visualisation
plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, train_mean_n, label="Train Score", marker='o')
plt.plot(n_estimators_range, test_mean_n, label="Test Score", marker='o')
plt.title("Validation Curve for n_estimators (Gradient Boosting)")
plt.xlabel("n_estimators")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

# learning_rate val curve visualisation
plt.subplot(1, 2, 2)
plt.plot(learning_rate_range, train_mean_lr, label="Train Score", marker='o')
plt.plot(learning_rate_range, test_mean_lr, label="Test Score", marker='o')
plt.title("Validation Curve for learning_rate (Gradient Boosting)")
plt.xlabel("learning_rate")
plt.xscale("log")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

### 3

# XGBoost Implementation
xgboost_model = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=42, eval_metric="rmse"
)

n_estimators_range = np.arange(10, 210, 20)
learning_rate_range = np.logspace(-2, 0.5, 10)

# Validation curve for n_estimators
train_scores_n_xgb, test_scores_n_xgb = validation_curve(
    xgboost_model, X_train, y_train,
    param_name="n_estimators", param_range=n_estimators_range,
    scoring="neg_mean_squared_error", cv=5
)

# Validation curve for learning_rate
train_scores_lr_xgb, test_scores_lr_xgb = validation_curve(
    xgboost_model, X_train, y_train,
    param_name="learning_rate", param_range=learning_rate_range,
    scoring="neg_mean_squared_error", cv=5
)

# Process results
train_mean_n_xgb = -np.mean(train_scores_n_xgb, axis=1)
test_mean_n_xgb = -np.mean(test_scores_n_xgb, axis=1)
train_mean_lr_xgb = -np.mean(train_scores_lr_xgb, axis=1)
test_mean_lr_xgb = -np.mean(test_scores_lr_xgb, axis=1)

# LightGBM Implementation
lightgbm_model = lgb.LGBMRegressor(random_state=42, force_col_wise=True)

# Validation curve for n_estimators
train_scores_n_lgb, test_scores_n_lgb = validation_curve(
    lightgbm_model, X_train, y_train,
    param_name="n_estimators", param_range=n_estimators_range,
    scoring="neg_mean_squared_error", cv=5
)

# Validation curve for learning_rate
train_scores_lr_lgb, test_scores_lr_lgb = validation_curve(
    lightgbm_model, X_train, y_train,
    param_name="learning_rate", param_range=learning_rate_range,
    scoring="neg_mean_squared_error", cv=5
)

# Process results
train_mean_n_lgb = -np.mean(train_scores_n_lgb, axis=1)
test_mean_n_lgb = -np.mean(test_scores_n_lgb, axis=1)
train_mean_lr_lgb = -np.mean(train_scores_lr_lgb, axis=1)
test_mean_lr_lgb = -np.mean(test_scores_lr_lgb, axis=1)

# Plotting results
plt.figure(figsize=(15, 10))

# XGBoost n_estimators
plt.subplot(2, 2, 1)
plt.plot(n_estimators_range, train_mean_n_xgb, label="Train Score", marker="o")
plt.plot(n_estimators_range, test_mean_n_xgb, label="Test Score", marker="o")
plt.title("XGBoost: Validation Curve for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

# XGBoost learning_rate
plt.subplot(2, 2, 2)
plt.plot(learning_rate_range, train_mean_lr_xgb, label="Train Score", marker="o")
plt.plot(learning_rate_range, test_mean_lr_xgb, label="Test Score", marker="o")
plt.title("XGBoost: Validation Curve for learning_rate")
plt.xlabel("learning_rate")
plt.xscale("log")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

# LightGBM n_estimators
plt.subplot(2, 2, 3)
plt.plot(n_estimators_range, train_mean_n_lgb, label="Train Score", marker="o")
plt.plot(n_estimators_range, test_mean_n_lgb, label="Test Score", marker="o")
plt.title("LightGBM: Validation Curve for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

# LightGBM learning_rate
plt.subplot(2, 2, 4)
plt.plot(learning_rate_range, train_mean_lr_lgb, label="Train Score", marker="o")
plt.plot(learning_rate_range, test_mean_lr_lgb, label="Test Score", marker="o")
plt.title("LightGBM: Validation Curve for learning_rate")
plt.xlabel("learning_rate")
plt.xscale("log")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Feature Importance Analysis for XGBoost
xgboost_model.set_params(n_estimators=100, learning_rate=0.1)
xgboost_model.fit(X_train, y_train)
xgb_importance = xgboost_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, xgb_importance)
plt.title("Feature Importance (XGBoost)")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Feature Importance Analysis for LightGBM
lightgbm_model.set_params(n_estimators=100, learning_rate=0.1)
lightgbm_model.fit(X_train, y_train)
lgb_importance = lightgbm_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, lgb_importance)
plt.title("Feature Importance (LightGBM)")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.grid()
plt.show()


xgboost_model.set_params(n_estimators=100, learning_rate=0.1)
xgboost_model.fit(X_train, y_train)

y_train_pred_xgb = xgboost_model.predict(X_train)
y_test_pred_xgb = xgboost_model.predict(X_test)

mse_test_xgb = mean_squared_error(y_test, y_test_pred_xgb)

print(f"XGBoost - MSE: {mse_test_xgb:.4f}")

# Train and evaluate LightGBM model
lightgbm_model.set_params(n_estimators=100, learning_rate=0.1)
lightgbm_model.fit(X_train, y_train)

y_train_pred_lgb = lightgbm_model.predict(X_train)
y_test_pred_lgb = lightgbm_model.predict(X_test)

mse_test_lgb = mean_squared_error(y_test, y_test_pred_lgb)

print(f"LightGBM - MSE: {mse_test_lgb:.4f}")
