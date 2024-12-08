import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

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

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### 1

knn_regressor = KNeighborsRegressor(n_neighbors=5)

knn_regressor.fit(X_train_scaled, y_train)

y_pred = knn_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f"MAE (RadiusNeighborsRegressor): {mae}")

mean_shares = y_test.mean()
median_shares = y_test.median()
std_shares = y_test.std()

print(f'Середнє значення shares: {mean_shares}')
print(f'Медіана значень shares: {median_shares}')
print(f'Стандартне відхилення shares: {std_shares}')

### 2

kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'n_neighbors': list(range(1, 51))}
knn_regressor = KNeighborsRegressor()
grid_search = GridSearchCV(knn_regressor, param_grid, cv=kf, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_scaled, y_train)

results = grid_search.cv_results_
k_values = param_grid['n_neighbors']
mse_scores = -results['mean_test_score']

best_k = grid_search.best_params_['n_neighbors']
best_mse = -grid_search.best_score_

print(f'Найкраще значення k: {best_k}')
print(f'Найкраще значення MSE на валідаційній вибірці: {best_mse}')

plt.plot(k_values, mse_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Число сусідів (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Залежність MSE від числа сусідів (k)')
plt.grid()
plt.show()

### 3

p_values = np.linspace(1, 10, 20)
best_p = None
best_score = float('-inf')

for p in p_values:
    knn = KNeighborsRegressor(n_neighbors=best_k, metric='minkowski', p=p, weights='distance')
    scores = cross_val_score(knn, X, y, cv=5, scoring='neg_mean_absolute_error')
    mean_score = scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_p = p

print(f"Оптимальне значення параметра p: {best_p}")
print(f"Найкраще середнє значення MAE: {-best_score}")

### 4

# RadiusNeighborsRegressor

param_grid = {'radius': np.linspace(0.1, 10, 50)}
grid_search = GridSearchCV(
    RadiusNeighborsRegressor(weights='distance', metric='minkowski', p=2),
    param_grid,
    scoring='neg_mean_absolute_error',
    cv=5
)
grid_search.fit(X_train_scaled, y_train)
best_radius = grid_search.best_params_['radius']
print(f"Оптимальний радіус: {best_radius}")

radius_regressor = RadiusNeighborsRegressor(radius=best_radius, weights='distance', metric='minkowski', p=2)

radius_regressor.fit(X_train_scaled, y_train)

y_pred_radius = radius_regressor.predict(X_test_scaled)

mse_radius = mean_squared_error(y_test, y_pred_radius)
mae_radius = mean_absolute_error(y_test, y_pred_radius)

print(f"MSE (RadiusNeighborsRegressor): {mse_radius}")
print(f"MAE (RadiusNeighborsRegressor): {mae_radius}")
