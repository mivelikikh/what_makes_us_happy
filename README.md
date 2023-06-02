# What makes us happy?

## Introduction
The pursuit of happiness has fascinated humanity throughout the ages. Ancient philosophers and modern-day researchers have sought to uncover its source, however the secret formula to happiness remains yet to be found. In our project, we join those who have embarked on this quest, exploring the factors that influence happiness and striving to find a predictive model for happiness. Join us on this captivating journey as we dive into modeling happiness and its determinants, and inspire a happier and more fulfilling world. Moreover, it is important to talk about happiness, as it contributes to several Sustainable Development Goals (SDGs), such as Goals 1, 2, 3, 4, 5, 6, 10, and 16 highlighting its importance in creating a sustainable and fulfilling world.

In line with this, the aim of this project is to answer the question, "What factors influence the level of happiness in countries? To answer this question, we applied different methods and algorithms to find the best one that predicts the happiness (given by the feature "Happiness Score.")

### Contributors
- Maria Velikikh ([@mivelikikh](https://github.com/mivelikikh))
- Emilija Vukasinovic ([@emavuk](https://github.com/emavuk))
- Paula Ramirez Ortega ([@Pramirezortega](https://github.com/Pramirezortega))

In this project [`2016_world_metrics.csv`](https://www.kaggle.com/code/dariasvasileva/merging-world-metrics-sets/output) (37.3 KB) dataset is used.

To see the full project please refer to the notebook [`full_project.ipynb`](https://github.com/mivelikikh/what_makes_us_happy/blob/main/full_project.ipynb).

To watch the video of us explaining the most important parts of the project please refer to this [link](https://youtu.be/DU891iL73Bo).

## Exploratory Data Analysis

We began by examining the dataset to understand its characteristics. Our dataset `world_metrics` contains information on health and life expectancy data as well as on ecological footprint, human freedom scores, and happiness scores for 137 countries in 2016. Overall, it includes 30 features, out of which one is the “country name”, another the “happiness score” (our target variable) and the remaining 28, the predictors.

We discovered our data, checking for outliers, correlations between variables and happiness scores, and more. Computing the correlations, helped us define features with a high correlation (> $|0.6|$) out of which we compiled a subset, with which we worked later in our analysis. Thereafter, we used different clustering methods, such as clustering based on quintiles but also more advanced methods such as Kmeans and Hierarchical clustering. The clustering helped us better understand the distribution of our data based on the countries and their corresponding scores. One example of the clustering based on quintiles can be found below, showing that Europe, North America, Australia, and partially South America are the happiest regions in the world. We also see that Africa, as we expected, appears as the least happiest region.

<img width="651" alt="map_1" src="https://github.com/mivelikikh/what_makes_us_happy/assets/98487867/8ac19b06-6212-4df1-9cd8-38925c9db936">

## Prediction Models

After extensively discovering our data we applied a complex strategy to analyze how the performance of our models could be influenced by various input parameters. Our objective was to investigate the following aspects:

1. The difference between different data samples:
  - The full dataset `world_metrics` (considering the effect of each original feature);
  - The subset `world_metrics_subset` (containing only the most correlated features with the target feature);
2. The effect of artificially constructed features:
  - Does the use of `PolynomialFeatures()` in the preprocessing step improve model performance?
3. The effect of scalling technique:
  - `MinMaxScaller()`
  - `StandardScaller()`
4. The effect of dimensionality reduction with the use of Principal Component Analysis (PCA):
  - Do we need all the features from the dataset?
  - Do we need only a few?

Through this approach, we aimed to assess the evolution of the prediction model's performance. In each scenario, we worked with the following models to evaluate their performance:

1. ElasticNet (to see what should we prefer: Ridge vs. Lasso)
2. Ridge
3. Lasso
4. kNN
5. Decision Tree
6. Random Forest
7. Support Vector Regression

To predict, we used a set of custom functions that are placed in the file [`functions.py`](https://github.com/mivelikikh/what_makes_us_happy/blob/main/functions.py). These functions were designed to work together in the workflow for performing grid search on a regressor, obtaining results, and extracting the best models based on different scoring functions. The code output includes both the model settings and the calculated metrics. The model settings provide information about the chosen algorithm, hyperparameters, and preprocessing steps used. The calculated metrics consist of the mean $R^2$, mean MAE, mean MSE, and standard deviations for each metric, allowing us to observe the minimum and maximum values across the five folds that was created with the use of Cross-Validation.

## Results

This table compares all the models in terms of the obtained **mean MSE**. We use the following abreviations for compactness of the table:

- F = Full Sample Approach
- S = Subset Sample Approach

|sample|model name|mean MSE|std||sample|model name|mean MSE|std|
|---|---|---|---|---|---|---|---|---|
|F|SVR|0.3355|0.0841||S|SVR|<span style="color: darkgreen;">0.2842</span>|<span style="color: darkgreen;">0.0424</span>|
|F|Random Forest|0.3465|0.0761||S|Random Forest|<span style="color: darkgreen;">0.2961</span>|<span style="color: darkred;">0.0831</span>|
|F|Ridge|0.3512|0.0804||S|Ridge|<span style="color: darkgreen;">0.3499</span>|<span style="color: darkgreen;">0.0565</span>|
|F|ElasticNet|0.3519|0.0806||S|ElasticNet|<span style="color: darkgreen;">0.3510</span>|<span style="color: darkgreen;">0.0576</span>|
|F|Lasso|0.3577|0.0945||S|Lasso|<span style="color: darkgreen;">0.3561</span>|<span style="color: darkgreen;">0.0522</span>|
|F|kNN|0.3619|0.1106||S|kNN|<span style="color: darkgreen;">0.3073</span>|<span style="color: darkgreen;">0.0727</span>|
|F|Decision Tree|0.5657|0.1843||S|Decision Tree|<span style="color: darkgreen;">0.4734</span>|<span style="color: darkgreen;">0.1301</span>|

Especially, we found the following:

**Full Sample Approach.** Among the regression models, SVR (Support Vector Regression) has the lowest mean MSE of 0.3355, followed closely by Random Forest, and Ridge. ElasticNet, Lasso and kNN have slightly higher mean MSE values, and Decision Tree performs the worst with a mean MSE of 0.5657.

**Subset Approach**. SVR still has the lowest mean MSE of 0.2842, followed by Random Forest and kNN. Ridge, ElasticNet, and Lasso perform similarly, while Decision Tree has the highest mean MSE of 0.4734.

Overall, SVR consistently performs well on both the Full and Subset datasets, achieving the lowest mean MSE in both cases. This indicates that SVR is the most accurate model in predicting happiness scores based on the given features. It demonstrates good generalization ability and robustness across different dataset sizes.

## Limitations and further research

In conclusion, our investigation did not uncover a definitive formula for happiness. However, we gained valuable insights into the factors associated with happiness and their alignment with the Sustainable Development Goals. It is important to acknowledge that the parameters used in our analysis offer a generalized perspective on happiness and may not fully capture its individual and multifaceted nature. Nevertheless, these insights have fueled our determination to continue our quest for understanding happiness and contribute to a happier world.

Looking ahead, we have several plans for future research. We intend to create subsets of factors based on our own definition of happiness, explore how happiness is depicted in cartoons, and investigate cultural perspectives on happiness. Additionally, we aim to expand our dataset by including currently missing countries, enabling us to gain a more comprehensive global view and examine regional variations. By pursuing these avenues and incorporating additional data, we seek to enhance the comprehensiveness of our findings and uncover new insights into the complex nature of happiness.
