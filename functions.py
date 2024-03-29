import pandas as pd
import numpy as np

## visualisation
import matplotlib.pyplot as plt

## ML
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
    ## ElasticNet, Ridge, Lasso
from sklearn.linear_model import ElasticNet, Ridge, Lasso
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# to plot histogram
def plot_histogram(data, num_bins=10, title='', xlabel='', ylabel=''):
    """
    Plot a histogram of the given data.

    Parameters:
    - data: List or Numpy array containing the data points.
    - num_bins: Number of bins to use in the histogram (default: 10).
    - title: Title of the histogram (default: '').
    - xlabel: Label for the x-axis (default: '').
    - ylabel: Label for the y-axis (default: '').

    Returns:
    - None (the histogram plot will be displayed).
    """
    plt.hist(data, bins=num_bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    

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

    fig.suptitle(title, y=1.0, fontsize=title_fontsize)

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


def plot_pca_explained_variance(best_parameters, pca_n_components, X):
    explained_variance_ratios = []
    for n_components in pca_n_components:
        pipeline = Pipeline([('polynomial',
                              PolynomialFeatures(degree=best_parameters['polynomial__degree'],
                                                 interaction_only=best_parameters['polynomial__interaction_only'])),
                             ('scaler', best_parameters['scaler']),
                             ('pca', PCA(n_components=n_components))])
        pipeline.fit(X)
        explained_variance_ratios.append(pipeline['pca'].explained_variance_ratio_)
      
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.plot(pca_n_components, explained_variance_ratios, "+", linewidth=2)
    ax.set_ylabel("PCA explained variance ratio")
    
    plt.show()


# get grid search
def get_grid_search_regressor(regressor, regressor_parameters_grid,
                              poly_degrees, pca_n_components,
                              score_functions,
                              refit=False,
                              random_state=42):
    pipeline = Pipeline([('polynomial', PolynomialFeatures()),
                         ('scaler', None),
                         ('pca', PCA()),
                         ('regressor', regressor)])
    param_grid = {'polynomial__degree': poly_degrees,
                  'polynomial__interaction_only': [True, False],
                  'scaler': [MinMaxScaler(), StandardScaler()],
                  'pca__n_components': pca_n_components}
    for parameter, values in regressor_parameters_grid.items():
        param_grid[f'regressor__{parameter}'] = values
    
    regressor = GridSearchCV(estimator=pipeline,
                             param_grid=param_grid,
                             scoring=score_functions,
                             cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
                             refit=refit)
    
    return regressor


# to get metrics (parameter, mean_test, std_test)
def get_results(grid_search_regressor, X, y):
    grid_search_regressor.fit(X, y)
    results_table = pd.DataFrame(grid_search_regressor.cv_results_)
    
    columns = [f'param_{parameter}' for parameter in grid_search_regressor.param_grid.keys()] +\
              [f'mean_test_{score_func}' for score_func in grid_search_regressor.scoring] +\
              [f'std_test_{score_func}' for score_func in grid_search_regressor.scoring]
    results_table = results_table[columns]
    
    return results_table


# to get results for the best model
def get_best_model(results_table, score_function):
    coefficient = 1.0
    score_function_name = score_function
    if score_function in ['mae', 'mean_absolute_error', 'MAE']:
        coefficient = -1.0
        score_function_name = 'neg_mean_absolute_error'
    if score_function in ['mse', 'mean_squared_error', 'MSE']:
        coefficient = -1.0
        score_function_name = 'neg_mean_squared_error'
    
    columns = [column_name for column_name in results_table.columns if 'param' in column_name] +\
              [f'mean_test_{score_function_name}', f'std_test_{score_function_name}']
    new_columns = ['_'.join(column_name.split('_')[1:])
                   for column_name in results_table.columns if 'param' in column_name] +\
                  [f'mean_{score_function}', f'std_{score_function}']
    columns_rename_map = dict((old_name, new_name) for old_name, new_name in zip(columns, new_columns))
    
    best_parameters = results_table[columns]
    best_parameters = best_parameters.rename(columns=columns_rename_map)
    
    score_function_name = 'mean_' + score_function
    best_parameters = best_parameters.loc[[best_parameters[score_function_name].idxmax()]]
    best_parameters[score_function_name] = best_parameters[score_function_name] * coefficient
    best_parameters.reset_index(drop=True)
    
    return best_parameters


# to combine best models in one table
def get_best_models(results_table, score_functions):
    best_parameters = pd.concat(get_best_model(results_table, score_function)
                                for score_function in score_functions)
    best_parameters = best_parameters.reset_index(drop=True)
    
    return best_parameters






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