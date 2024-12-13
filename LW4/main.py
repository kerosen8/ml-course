import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns


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

regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")

### 2

kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

regressor = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=regressor,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=kf,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X, y)

best_params = grid_search.best_params_
best_mse = -grid_search.best_score_

print("Best params for model:")
print(best_params)
print(f"MSE using cross-validation: {best_mse:.2f}")

# validation curves

parameters = {
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

for param_name, param_range in parameters.items():
    train_scores, test_scores = validation_curve(
        DecisionTreeRegressor(random_state=42),
        X, y,
        param_name=param_name,
        param_range=param_range,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    
    plt.plot(param_range, train_scores_mean, label='Train MSE', color='blue', marker='o')
    plt.plot(param_range, test_scores_mean, label='Test MSE', color='orange', marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error')
    plt.title(f'Validation Curve for {param_name}')
    plt.legend()
    plt.grid()
    plt.show()

# tree

best_regressor = DecisionTreeRegressor(
    max_depth=3,
    max_features=None,
    min_samples_leaf=5,
    min_samples_split=2,
    random_state=42
)
best_regressor.fit(X, y)

plt.figure(figsize=(16, 10))
plot_tree(best_regressor, feature_names=X.columns, filled=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

# feature importances
feature_importances = best_regressor.feature_importances_

sns.barplot(x=feature_importances, y=X.columns)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid()
plt.show()

### 3

# rf

rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)

y_pred_rf = rf_default.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f"Mean Squared Error (MSE) for Random Forest (default parameters): {mse_rf:.2f}")

# hyperparams

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=kf,
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

y_pred_rf_best = best_rf.predict(X_test)
mse_rf_best = mean_squared_error(y_test, y_pred_rf_best)

print(f"Best Parameters for Random Forest: {rf_grid.best_params_}")
print(f"Mean Squared Error (MSE) for Random Forest (optimized): {mse_rf_best:.2f}")

# validation curves

parameters_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

for param_name, param_range in parameters_rf.items():
    train_scores, test_scores = validation_curve(
        RandomForestRegressor(random_state=42),
        X, y,
        param_name=param_name,
        param_range=param_range,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1
    )

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.plot(param_range, train_scores_mean, label='Train MSE', color='blue', marker='o')
    plt.plot(param_range, test_scores_mean, label='Test MSE', color='orange', marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error')
    plt.title(f'Validation Curve for {param_name}')
    plt.legend()
    plt.grid()
    plt.show()

# feat importances

feature_importances_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_features = feature_importances_rf.head(10)
sns.barplot(data=top_features, x='Importance', y='Feature')
plt.title('Top-10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid()
plt.show()
