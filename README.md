# Cancer diagnosis classifier

**Implementing a K-Nearest Neighbors model to classify tumors as malignant or benign.**

## Project overview
This project leverages the K-Nearest Neighbors (KNN) algorithm—a distance-based supervised learning model—to classify tumors as either Malignant or Benign based on physical characteristics.

## The data
The data for this project was obtained through a publicly available Kaggle dataset on tumor characteristics. The dataset can be accessed [here](https://www.kaggle.com/datasets/erdemtaha/cancer-data). All the data is contained within a single file, Cancer_Data.csv, which can be found in the data folder of this project. The dataset included 32 columns.

## Development of the project
Given that the idea of this Project was to classify tumors as malignant or benign according to their physical characteristics, a KNN model was chosen for classification, as it made it possible to group tumors with similar characteristics and derive the prediction from there. The reasoning behind this decision was that tumors with similar characteristics are likely to be of the same kind, so a model that derives predictions based on distance appeared to be the most sensible choice.

There is one caveat in this situation though. Given that the model makes predictions based on distance, all data had to be standardized to avoid having differences in units skew predictions.

### Hyperparameter tuning
Given that the purpose of this model is to predict whether tumors are malignant, the cost for a false negative is extremely high. Therefore, the hyperparameters of the model were tuned for the highest possible recall. The two hyperparameters in question were:
**Number of neighbors (K):** The number of closest points used to determine the classification category.
**Weighting:** How neighbors were weighted (either uniformly or by distance).

A grid search with cross validation was used to determine what the best set of hyperparameters would be, so a pipeline was set to handle the standardization of the training data for each time the test fold changed. Once this was done, the optimal value of K was found to be 3, with the weighting being applied uniformly, not by distance.

### Model evaluation
Despite tuning the model for recall specifically, all 4 metrics were analyzed to see how the model was performing as a whole. A data frame was created with the metrics both for training and validation data to see how well the model was generalizing. The results are included in Table 1 below.

| Metric | Training score | Test score |
|:--- | :--- | :--- |
| Recall | 0.9585 | 0.9302 |
| Accuracy | 0.9846 | 0.9473 |
| Precision | 1	| 0.9302 |
| F1 Score | 0.9788 | 0.9302 |

*Table 1: Training and test scores for the 4 evaluation metrics of the tuned model.*

As seen in the table above, despite a slight drop in performance with the validation data, the metrics remain relatively consistent when presented with unseen entries, which suggests that the model is not overfitting the data and generalizing well.

### Limitations & Robustness
While the model is performing well and generalizing well to new data, it's worth noting that the dataset used to build it contained only 569 entries. The reduced sample size increases the probability of having a particularly easy-to-predict dataset that artificially inflates the model's performance. If in a bigger sample there was a considerable amount of variation in one or multiple variables, it would be quite likely that the model would start underperforming. This is something that should be accounted for when using this classifier.

Also, it is clear that some of the regressors are likely to be correlated, as features such as the perimeter and the area of a tumor are likely to be positively correlated for instance. While this would hurt the reliability of specific coefficients when building a model, it does not represent an issue with distance-based algorithms such as KNN. What's more, this does not mean that the regressors should be reduced, as, while it might help with interpretability, it can considerably hurt the model's predictive capabilities.

The full analysis, including the evaluation of the model, can be found [here](tumor_classifier.ipynb).

## How to explore this repository
**Analysis:** View the full the analysis in the [Jupyter Notebook](tumor_classifier.ipynb) for full EDA and modelling.

**Reproduction:** Run `pip install -r requirements.txt` to set up the environment.