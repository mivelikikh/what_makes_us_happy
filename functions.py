import pandas as pd
import numpy as np

## visualisation
import matplotlib.pyplot as plt

## ML
from sklearn.model_selection import train_test_split
    ## knn
from sklearn.neighbors import KNeighborsRegressor
    ## linear regression
from sklearn.linear_model import LinearRegression
    ## polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
    ## decision tree
from sklearn.tree import DecisionTreeRegressor
    ## metrics
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# to plot scatterplot
def plot_scatterplot(dataset, title=str, title_fontsize=int, ax_fontsize=int,
                     fig_width=int, fig_height=int,
                     nrows=int, ncols=int):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    if nrows == 1 and ncols == 1:
        axes = [axes]

    for col, ax in zip(dataset.columns, axes.flat):
        ax.scatter(dataset.index, dataset[col], label=col)
        ax.set_xlabel("Index", fontsize=ax_fontsize)
        ax.set_title(col, fontsize=ax_fontsize)
        
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(tick) for tick in xticks], rotation=0)
        ax.grid(True, color='gray', linestyle='--')

    fig.suptitle(title, y=1.0 , fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()
    

# to plot boxplot
def plot_boxplot(dataset, title=str, title_fontsize=int, ax_fontsize=int,
                 fig_width=int, fig_height=int, nrows=int, ncols=int,
                 box_color='blue', whisker_color='black'):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    if nrows == 1 and ncols == 1:
        axes = [axes]

    for col, ax in zip(dataset.columns, axes.flat):
        bp = ax.boxplot(dataset[col], boxprops={'color': box_color},
                        whiskerprops={'color': whisker_color})
        ax.set_xlabel(col, fontsize=ax_fontsize)
        ax.set_xticklabels([col], rotation=0)
        ax.grid(True, color='gray', linestyle='--')

    fig.suptitle(title, y=1.0, fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()

    
# to calculate skewness
def calculate_skewness(dataset):
    skewness = []
    
    for column in dataset.columns:
        skewness.append(round(stats.skew(dataset[column]), 3))
    return skewness


# to calculate metrics
def calculate_metrics(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, model_name, scaler_used):
    """
    Computes the statistics: MAE, MSE, R^2, adjusted R^2 
    on the training and test sets.

    Args:
    X_train: training set features
    X_test: test set features
    y_train: training set target variable
    y_test: test set target variable
    y_train_pred: model train set predictions
    y_test_pred: model test set predictions
    model_name: prediction model usde (string)
    scaler_used: scaling approach used before (string)

    Returns:
    Dataframe with MAE, MSE, R^2, adjusted R^2 on the training and test sets.
    """
    
    # train set
    mae_train = round(mean_absolute_error(y_train, y_train_pred), 4)
    mse_train = round(mean_squared_error(y_train, y_train_pred), 4)
    r2_train = round(r2_score(y_train, y_train_pred), 4)
    adj_r2_train = round(1 - (1 - r2_train) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1)), 4)

    # test set
    mae_test = round(mean_absolute_error(y_test, y_test_pred), 4)
    mse_test = round(mean_squared_error(y_test, y_test_pred), 4)
    r2_test = round(r2_score(y_test, y_test_pred), 4)
    adj_r2_test = round(1 - (1 - r2_test) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)), 4)

    # metrics dataframe
    model_metrics = pd.DataFrame({'train': [mae_train, mse_train, r2_train, adj_r2_train],
                                 'test': [mae_test, mse_test, r2_test, adj_r2_test]},
                                 index=['MAE', 'MSE', 'R^2', 'adj. R^2'])
    
    model_metrics.columns.name = model_name + scaler_used
    
    return model_metrics


# to run KNN and to compute metrics
def run_knn(feature_matrix, y, n_neighbors, scaler_used):
    """
    Runs KNN model
    Computes the statistics: MAE, MSE, R^2, adjusted R^2 
    on the training and test sets.

    Args:
    feature_matrix: training set features
    y: target variable
    n_neighbors: number of neighbors
    scaler_used: scaling approach used before (string)

    Returns:
    Dataframe with MAE, MSE, R^2, adjusted R^2 on the training and test sets.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=88)
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    return calculate_metrics(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "KNN_", scaler_used)


# to run linear regression and to compute metrics
def run_linreg(feature_matrix, y, scaler_used):
    """
    Runs Linear Regression model
    Computes the statistics: MAE, MSE, R^2, adjusted R^2 
    on the training and test sets.

    Args:
    feature_matrix: training set features
    y: target variable
    scaler_used: scaling approach used before (string)

    Returns:
    Dataframe with MAE, MSE, R^2, adjusted R^2 on the training and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=99)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    
    return calculate_metrics(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "LinReg_", scaler_used)


# to run polynomial regression model and to compute metrics
def poly_reg(degree, feature_matrix, y, scaler_used):
    """
    Runs Polynomial Regression model
    Computes the statistics: MAE, MSE, R^2, adjusted R^2 
    on the training and test sets.

    Args:
    degree: the degree of polynomial
    feature_matrix: training set features
    y: target variable
    scaler_used: scaling approach used before (string)

    Returns:
    Dataframe with MAE, MSE, R^2, adjusted R^2 on the training and test sets.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=99)
    
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train)

    y_train_pred = lin_reg.predict(X_train_poly)
    y_test_pred = lin_reg.predict(X_test_poly)
    
    return calculate_metrics(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Poly_", scaler_used)


# to run poly_ref + ridge + cv and to compute metrics
def ridge_regression(degree, X_train, y_train, X_test, y_test, scaler_used, cross_validation=False):
    """
    Runs Polynomial Regression model with Ridge regularization and optional cross-validation
    Computes the statistics: MAE, MSE, R^2 on the train and test sets

    Args:
    degree: degree of polynomial features (integer)
    X_train: training set features
    y_train: training set target variable
    X_test: test set features
    y_test: test set target variable
    scaler_used: scaling approach used before (string)
    cross_validation: whether to perform cross-validation: default False (boolean)

    Returns:
    Dataframe with MAE, MSE, R2 on the training and test sets.
    """

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    if cross_validation:
        ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5, fit_intercept=False)
    else:
        ridge_model = Ridge(alpha=1.0, fit_intercept=False)

    ridge_model.fit(X_train_poly, y_train)

    # metrics: train set
    predictions_train = ridge_model.predict(X_train_poly)
    mae_train = round(mean_absolute_error(y_train, predictions_train), 4)
    mse_train = round(mean_squared_error(y_train, predictions_train), 4)
    r2_train = round(r2_score(y_train, predictions_train), 4)
    adj_r2_train = round(1 - (1 - r2_train) * ((X_train_poly.shape[0] - 1) / (X_train_poly.shape[0] - X_train_poly.shape[1] - 1)), 4)

    # metrics: test set
    predictions_test = ridge_model.predict(X_test_poly)
    mae_test = round(mean_absolute_error(y_test, predictions_test), 4)
    mse_test = round(mean_squared_error(y_test, predictions_test), 4)
    r2_test = round(r2_score(y_test, predictions_test), 4)
    adj_r2_test = round(1 - (1 - r2_test) * ((X_test_poly.shape[0] - 1) / (X_test_poly.shape[0] - X_test_poly.shape[1] - 1)), 4)
    
    # metrics dataframe
    if cross_validation:
        ridge_metrics = pd.DataFrame({'train': [mae_train, mse_train, r2_train, adj_r2_train],
                                      'test': [mae_test, mse_test, r2_test, adj_r2_test]},
                                      index=['MAE', 'MSE', 'R^2', 'adj. R^2'])
    else:
        ridge_metrics = pd.DataFrame({'train': [mae_train, mse_train, r2_train, adj_r2_train],
                                      'test': [mae_test, mse_test, r2_test, adj_r2_test]},
                                      index=['MAE', 'MSE', 'R^2', 'adj. R^2'])
    
    ridge_metrics.columns.name = scaler_used

    return ridge_metrics


# to run poly_ref + lasso + cv and to compute metrics
def lasso_regression(degree, X_train, y_train, X_test, y_test, scaler_used, cross_validation=False):
    """
    Runs Polynomial Regression model with Lasso regularization and optional cross-validation
    Computes the statistics: MAE, MSE, R^2 on the train and test sets

    Args:
    degree: degree of polynomial features (integer)
    X_train: training set features
    y_train: training set target variable
    X_test: test set features
    y_test: test set target variable
    scaler_used: scaling approach used before (string)
    cross_validation: whether to perform cross-validation: default False (boolean)

    Returns:
    Dataframe with MAE, MSE, R2 on the training and test sets.
    """

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    if cross_validation:
        lasso_model = LassoCV(alphas=[0.1, 1.0, 10.0], cv=5, fit_intercept=False)
    else:
        lasso_model = Lasso(alpha=1.0, fit_intercept=False)

    lasso_model.fit(X_train_poly, y_train)

    # metrics: train set
    predictions_train = lasso_model.predict(X_train_poly)
    mae_train = round(mean_absolute_error(y_train, predictions_train), 4)
    mse_train = round(mean_squared_error(y_train, predictions_train), 4)
    r2_train = round(r2_score(y_train, predictions_train), 4)
    adj_r2_train = round(1 - (1 - r2_train) * ((X_train_poly.shape[0] - 1) / (X_train_poly.shape[0] - X_train_poly.shape[1] - 1)), 4)

    # metrics: test set
    predictions_test = lasso_model.predict(X_test_poly)
    mae_test = round(mean_absolute_error(y_test, predictions_test), 4)
    mse_test = round(mean_squared_error(y_test, predictions_test), 4)
    r2_test = round(r2_score(y_test, predictions_test), 4)
    adj_r2_test = round(1 - (1 - r2_test) * ((X_test_poly.shape[0] - 1) / (X_test_poly.shape[0] - X_test_poly.shape[1] - 1)), 4)
    
    # metrics dataframe
    if cross_validation:
        lasso_metrics = pd.DataFrame({'train': [mae_train, mse_train, r2_train, adj_r2_train],
                                      'test': [mae_test, mse_test, r2_test, adj_r2_test]},
                                      index=['MAE', 'MSE', 'R^2', 'adj. R^2'])
    else:
        lasso_metrics = pd.DataFrame({'train': [mae_train, mse_train, r2_train, adj_r2_train],
                                      'test': [mae_test, mse_test, r2_test, adj_r2_test]},
                                      index=['MAE', 'MSE', 'R^2', 'adj. R^2'])
    
    lasso_metrics.columns.name = scaler_used

    return lasso_metrics


# to run tree and to compute metrics
def run_tree(feature_matrix, y, scaler_used):
    """
    Runs Decision Tree model
    Computes the statistics: MAE, MSE, R^2, adjusted R^2 
    on the training and test sets.

    Args:
    feature_matrix: training set features
    y: target variable
    scaler_used: scaling approach used before (string)

    Returns:
    Dataframe with MAE, MSE, R^2, adjusted R^2 on the training and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=99)
    
    model = DecisionTreeRegressor(random_state=99)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return calculate_metrics(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, "Tree_", scaler_used)