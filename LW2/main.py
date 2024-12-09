import numpy as np
import pandas as pd # type: ignore
import seaborn as sns
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
import matplotlib.pyplot as plt

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

# Cleaned data

outlier_indices_shares = detect_outliers(df['shares'])
outlier_indices_average_token_length = detect_outliers(df['average_token_length'])
outlier_indices_n_non_stop_unique_tokens = detect_outliers(df['n_non_stop_unique_tokens'])

combined_outlier_indices = np.unique(np.concatenate((outlier_indices_shares, outlier_indices_average_token_length, outlier_indices_n_non_stop_unique_tokens)))

X = X_1.drop(index=combined_outlier_indices).reset_index(drop=True)
y = y_1.drop(index=combined_outlier_indices).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression

default_model = LinearRegression()
default_model.fit(X_train, y_train)
default_y_pred = default_model.predict(X_test)

default_model_mse = mean_squared_error(y_test, default_y_pred)
default_model_mae = mean_absolute_error(y_test, default_y_pred)
default_model_rmse = root_mean_squared_error(y_test, default_y_pred)
default_model_mape = mean_absolute_percentage_error(y_test, default_y_pred)

print("MSE for linear regression:", default_model_mse)
print("MAE for linear regression:", default_model_mae)
print("RMSE for linear regression:", default_model_rmse)
print("MAPE for linear regression:", default_model_mape)

# Ridge, Lasso, Elastic regularization with turned hyperparameters

ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet(max_iter=5000)

aplhas = np.logspace(-4, 4, 10)

param_grid_ridge = {
    'alpha': aplhas
}

param_grid_lasso = {
    'alpha': aplhas
}

param_grid_elastic = {
    'alpha': aplhas, 
    'l1_ratio': [0.5]
}

grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid_lasso, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_elastic = GridSearchCV(estimator=elastic, param_grid=param_grid_elastic, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search_ridge.fit(X_train, y_train)
grid_search_lasso.fit(X_train, y_train)
grid_search_elastic.fit(X_train, y_train)

best_ridge = grid_search_ridge.best_estimator_
best_lasso = grid_search_lasso.best_estimator_
best_elastic = grid_search_elastic.best_estimator_

y_pred_ridge = best_ridge.predict(X_test)
y_pred_lasso = best_lasso.predict(X_test)
y_pred_elastic = best_elastic.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)

# Comparsion of results

print(f"MSE Ridge: {mse_ridge}")
print(f"MSE Lasso: {mse_lasso}")
print(f"MSE ElasticNet: {mse_elastic}")

# Validation curves

plt.plot(aplhas, -grid_search_ridge.cv_results_['mean_test_score'], label='Ridge', color='blue')
plt.plot(aplhas, -grid_search_lasso.cv_results_['mean_test_score'], label='Lasso', color='green')
plt.plot(aplhas, -grid_search_elastic.cv_results_['mean_test_score'], label='Elastic', color='red')

plt.xscale('log')
plt.xlabel('Regularization')
plt.ylabel('MSE')
plt.title('Validation Curves for Ridge, Lasso, and ElasticNet')
plt.legend()
plt.grid()
plt.show()

plt.subplot(1, 3, 1)
plt.barh(X.columns, best_elastic.coef_, color='blue')
plt.title('ElasticNet Coefficients')
plt.xlabel('Coefficient Value')

plt.tight_layout()
plt.show()
