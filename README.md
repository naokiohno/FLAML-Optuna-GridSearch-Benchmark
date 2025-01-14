# FLAML-Optuna-GridSearch-Benchmark
Benchmarking the following hyperparameter tuning approaches: FLAML AutoML, Optuna automated hyperparameter search, and manual grid search. The training dataset is the Solubility dataset from Applied Predictive Modeling by Max Kuhn (2016).

* The task is regression. 228 molecular characteristics are used to predict the solubility of each molecule on a continuous scale.
* 5-fold cross-validation is used during the model tuning process to evaluate RMSE.
* For all tuning approaches, an xgboost model was used.
* All tuning methods aside from no tuning performed comparably, with a slight edge to the automated tuning approaches.

**Conclusion:** Automated hyperparameter tuning achieves state-of-the art performance. At the same time, it is much less computationally expensive compared to a full grid search, and requires much less active development time compared to an involved manual hyperparameter tuning approach. It is a valuable addition to a data scientist's skill set, especially when working on time-sensitive projects, or when quick prototyping is required. When lacking deep subject matter knowledge on a certain problem, these approaches create a great baseline model.

| Tuning method | Cross-val RMSE| Test set RMSE  |
| ------------- |:-------------:| :-------------:|
| No tuning     | 0.81          | 0.66           |
| Manual search | 0.70          | 0.60           |
| Optuna search | 0.67          | 0.57           |
| FLAML AutoML  | 0.69          | 0.58           |
