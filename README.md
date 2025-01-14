# FLAML-Optuna-GridSearch-Benchmark
Benchmarking the following hyperparameter tuning approaches: FLAML AutoML, Optuna automated hyperparameter search, and manual grid search, on the Solubility dataset from Applied Predictive Modeling by Max Kuhn.

* The tasks is a regression task that uses 228 molecular characteristics to predict the solubility of each molecule on a continuous scale.
* 5-fold cross-validation is used during the model tuning process to evaluate RMSE.
* For all tuning approaches, an xgboost model was used.
* All tuning methods aside from no tuning performed comparably, with a slight edge to the automated tuning approaches.
* Conclusion: Automated hyperparameter tuning achieves state-of-the art performance. At the same time, it is much less computationally expensive compared to a full grid search, and requires much less active development time compared to an involved manual hyperparameter tuning approach. 

| Tuning method | Cross-val RMSE| Test set RMSE  |
| ------------- |:-------------:| :-------------:|
| No tuning     | 0.81          | 0.66           |
| Manual search | 0.70          | 0.60           |
| Optuna search | 0.67          | 0.57           |
| FLAML AutoML  | 0.69          | 0.58           |
