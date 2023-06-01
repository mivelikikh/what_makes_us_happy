# What makes us happy?

## Introduction

The pursuit of happiness has fascinated humanity throughout the ages. Ancient philosophers and modern-day researchers have sought to uncover its source, however the secret formula to happiness remains yet to be found. In our project, we join those who have embarked on this quest, exploring the factors that influence happiness and striving to find a predictive model for happiness. Join us on this captivating journey as we dive into modeling happiness and its determinants, and inspire a happier and more fulfilling world.Exploring the Philosophical and Empirical Foundations of HappinessFor centuries, philosophers have pondered the key to happiness. Plato and Aristotle emphasized virtue and moral excellence, while Epicurus focused on the pursuit of pleasure. Kant stressed reason and morality, while Nietzsche found fulfillment in embracing life's challenges. More recently, research has taken a more empirical approach, seeking concrete factors that influence happiness. Various happiness reports and indices have emerged, measuring well-being through indicators beyond traditional economic measures like GDP.

Happiness research spans multiple disciplines, examining subjective well-being, socio-economic factors, psychological factors, social relationships, culture, health, politics, and the environment. The importance of this topic is underscored by its impact on individuals, societies, and the well-being industry as a whole.

Moreover, it is important to talk about happiness, as it contributes to several Sustainable Development Goals (SDGs) by addressing various aspects of well-being and societal progress. Poverty reduction (Goal 1), economic factors (such as income inequality), access to basic needs like food and water (Goals 2 and 6), good health and well-being (Goal 3), education and lifelong learning opportunities (Goal 4) are interconnected with happiness. Additionally, happiness is linked to peace, justice, and strong institutions (Goal 16), which ensure safety, access to justice, and good governance. Overall, happiness contributes to multiple SDGs, highlighting its importance in creating a sustainable and fulfilling world.


## Aim of the project

This project aims to answer the question "What factors influence the level of happiness in countries?". To answer this question, different methods and algorithms are applied to find the algorithm that best predicts the feature "Happiness Score". The contributions are made by:

- Maria Velikikh ([@mivelikikh](https://github.com/mivelikikh))
- Emilija Vukasinovic ([@emavuk](https://github.com/emavuk))
- Paula Ramirez Ortega ([@Pramirezortega](https://github.com/Pramirezortega))

In this project [`2016_world_metrics.csv`](https://www.kaggle.com/code/dariasvasileva/merging-world-metrics-sets/output) (37.3 KB) dataset is used.

## Exploratory Data Analysis

We began by examining the dataset to understand its characteristics, including checking for missing values, outliers, duplicate observations, and analyzing the distribution of the data. At the end of this exploratory data analysis we found out that the datasample was well prepared by its creator (no NaNs, no duplicates but some outliers).

1. The original dataset contains information on health and life expectancy data as well as on ecological footprint, human freedom scores, and happiness scores for 137 countries in 2016.
2. The dataset includes statistics for 29 distinct features.

Afterwards, we discovered the correlations between the variables and the happiness score as it is our target variable. We aimed to identify those features that are most strongly associated with the happiness score. Based on this, we created a subset of variables that were used in our further analysis.

## Clustering

Our initial analysis involved creating a vanilla clustering using only the happiness score feature. We used quintile splitting for this purpose, where Q0 is the quintile with the most unhappy countries and Q3 in the one with the happiest countries. This step helped us understand the distribution of happiness across the countries.
We took 2 approaches: based on the current range of happiness scores (minimal and maximal presented in the dataset), and the range from 0 to 10, since normally this indicator falls in this interval. Here you can see the results of the two clusterings.


