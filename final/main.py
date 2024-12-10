import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def plot_validation_curve(model_grid, param_name, params=None):
    results_df = pd.DataFrame(model_grid.cv_results_)
    param_column = 'param_' + param_name

    grouped = results_df.groupby(param_column)['mean_test_score'].mean()
    x_values = grouped.index
    y_values = grouped.values

    plt.plot(x_values, y_values, marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean Test F1 Score')
    plt.title(f'Validation curve for {param_name}')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    # Preprocessing

    df = pd.read_csv('../data/vodafone_music_subset.csv', delimiter=',')

    print(f"Кількість змінних до препроцессінгу: {df.shape[1]}")

    df = df.loc[:, df.isnull().sum() <= len(df) * 0.5]
    df = df.drop(df.filter(regex='id|voice|calls|^sms|cost').columns, axis=1)

    print(f"Кількість змінних після препроцессінгу: {df.shape[1]}")

    y = df['target']
    X = df.drop('target', axis=1)

    # Data cleaning

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()
    pipe = Pipeline([('removenan', imputer), ('scale', scaler)])
    X = pipe.fit_transform(X)

    # Splitting the data into test and training parts

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # 1. DecisionTreeClassifier

    decision_tree_classifier = DecisionTreeClassifier(random_state=42)
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)

    print(f"DecisionTreeClassifier - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    # Hyperparams selection for DecisionTreeClassifier

    tree_params = {
        'min_samples_split': np.arange(9, 19, 3),
        'max_depth': np.arange(3, 20, 2),
        'min_samples_leaf': [1, 5, 10, 20]
    }
    tree_grid = GridSearchCV(decision_tree_classifier, tree_params, cv=kf, scoring='f1', n_jobs=-1)
    tree_grid.fit(X_train, y_train)
    y_pred = tree_grid.best_estimator_.predict(X_test)

    print(f"DecisionTreeClassifier using best hyperparams - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier using best hyperparams - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier using best hyperparams - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier using best hyperparams - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"DecisionTreeClassifier using best hyperparams - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    for param_name, param_values in tree_params.items():
        plot_validation_curve(tree_grid, param_name, param_values)

    # 2. AdaBoostClassifier

    ada_boost_classifier = AdaBoostClassifier(n_estimators=500, random_state=42, algorithm='SAMME')
    ada_boost_classifier.fit(X_train, y_train)
    y_pred = ada_boost_classifier.predict(X_test)

    print(f"AdaBoostClassifier - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    # Hyperparams selection for DecisionTreeClassifier

    ada_boost_params = {
        'n_estimators': np.arange(300, 901, 100),
    }
    ada_boost_grid = GridSearchCV(ada_boost_classifier, ada_boost_params, cv=kf, scoring='f1', n_jobs=-1)
    ada_boost_grid.fit(X_train, y_train)
    y_pred = ada_boost_grid.best_estimator_.predict(X_test)

    print(f"AdaBoostClassifier using best hyperparam - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier using best hyperparam - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier using best hyperparam - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier using best hyperparam - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"AdaBoostClassifier using best hyperparam - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    for param_name, param_values in ada_boost_params.items():
        plot_validation_curve(ada_boost_grid, param_name, param_values)

    # 3. XGBoostClassifier

    xgb_classifier = xgb.XGBClassifier(random_state=42)
    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)

    print(f"XGBoostClassifier - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"XGBoostClassifier - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"XGBoostClassifier - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"XGBoostClassifier - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"XGBoostClassifier - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    # Hyperparams selection for DecisionTreeClassifier

    xgb_params = {
        'n_estimators': range(300, 501, 100),
        'max_depth': [4, 6, 8]
    }
    xgb_grid = GridSearchCV(xgb_classifier, xgb_params, cv=kf, scoring='f1', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    y_pred = xgb_grid.best_estimator_.predict(X_test)

    print(f"XGBoostClassifier using best hyperparams - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"XGBoostClassifier using best hyperparams - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"XGBoostClassifier using best hyperparams - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"XGBoostClassifier using best hyperparams - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"XGBoostClassifier using best hyperparams - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    for param_name, param_values in xgb_params.items():
        plot_validation_curve(xgb_grid, param_name, param_values)

    # 4. KNeighborsClassifier

    k_neighbors_classifier = KNeighborsClassifier(metric='minkowski', p=2, weights='uniform')
    k_neighbors_classifier.fit(X_train, y_train)
    y_pred = k_neighbors_classifier.predict(X_test)

    print(f"KNeighborsClassifier - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    # Hyperparams selection for KNeighborsClassifier

    k_neighbors_params = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
        'p': [1, 2], 
        'weights': ['uniform', 'distance']
    }

    k_neighbors_grid = GridSearchCV(k_neighbors_classifier, k_neighbors_params, cv=kf, scoring='f1', n_jobs=-1)
    k_neighbors_grid.fit(X_train, y_train)
    y_pred = k_neighbors_grid.best_estimator_.predict(X_test)

    print(f"KNeighborsClassifier using best hyperparams - Precision metric score: {precision_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier using best hyperparams - Recall metric score: {recall_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier using best hyperparams - F1 metric score: {f1_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier using best hyperparams - Accuracy metric score: {accuracy_score(y_test, y_pred)}")
    print(f"KNeighborsClassifier using best hyperparams - ROC AUC metric score: {roc_auc_score(y_test, y_pred)}")

    for param_name, param_values in k_neighbors_params.items():
        plot_validation_curve(k_neighbors_grid, param_name, param_values)
        