# ML Algorithms from Scratch
This implementations are done for ENGR421 Fall'20. Engr421 is an introductory course to machine learning.

## Algorithms
For using ML algorithms & models, we need only few lines of code. For example, naive bayes can be used as in the following:
```
from sklearn.naive_bayes import GaussianNB
>>> gnb = GaussianNB()
>>> y_pred = gnb.fit(X_train, y_train).predict(X_test)
```
However, this does not help to understand concepts and logic behind the models.

Therefore, this repository contains implementations from scratch only using basic numpy functions.
The implementations are not meant to be faster or more efficent than builtin methods from scikitlearn, sklearn etc.


## ENGR-421
Topics covered are regression, classification, clustering, and dimensionality reduction methods; supervised and unsupervised models; linear and nonlinear models; parametric and nonparametric models; combinations of multiple models; comparisons of multiple models and model selection.

Course focuses on both mathematical background and real life applications of the algorithms.

This repository has solutions to biweekly assignments which are implemented from scratch.
- [HW2: Naive bayes classifier](https://github.com/iremddemir/ml-algorithms-from-scratch/tree/main/hw2_naive_bayes_classifier)
- [HW3: Discrimination by regression is applied with K-sigmoid function](https://github.com/iremddemir/ml-algorithms-from-scratch/tree/main/hw3_discrimination_by_regression)
- [HW4: Nonparametric Regression](https://github.com/iremddemir/ml-algorithms-from-scratch/tree/main/hw4_nonparametric_regression)
- [HW5: Regression using decision tree](https://github.com/iremddemir/ml-algorithms-from-scratch/tree/main/hw5_decision_tree_regression)
- [HW6: One vs All Support Vector classification](https://github.com/iremddemir/ml-algorithms-from-scratch/tree/main/hw6_oVa_svm_classification)
- [HW7: Expectation Maximization clustiring](https://github.com/iremddemir/ml-algorithms-from-scratch/tree/main/hw7_expectation_maximization_clustering)
- [HW8: Spectral clustiring](https://github.com/iremddemir/ml-algorithms-from-scratch/tree/main/hw8_spectral_clustering)
