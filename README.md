# ROC/AUC for Binary Classification¶
For this documentation, we'll be working with a human resource dataset. Our goal is to find out the employees that are likely to leave in the future and act upon our findings, i.e. retain them before they choose to leave. This dataset contains 12000 observations and 7 variables, each representing :

* `satisfaction_level`: The satisfaction level on a scale of 0 to 1
* `last_evaluation` : Last project evaluation by a client on a scale of 0 to 1

* `Number_project` : Represents the number of projects worked on by employee in the last 12 month

* `average_montly_hours`: Average number of hours worked in the last 12 month for that employee

* `left`: 1 if the employee left the company, 0 if they're still working here. This is our response variable

To train and evaluate the model, we’ll perform a simple train/test split. 80 percent of the dataset will be used to actually train the model, while the rest will be used to evaluate the accuracy of this model, i.e. out of sample error. Note that the best practice is to split it in three ways train/validation/test split.



## Sklearn Transformer¶

We then convert perform some generic data preprocessing including standardizing the numeric columns and one-hot-encode the categorical columns (the "Newborn" variable is treated as a categorical variable) and convert everything into a numpy array that sklearn expects. This generic preprocessing step is written as a custom sklearn Transformer. You don't have to follow this structure if you prefer your way of doing it.


After training our model, we need to evaluate whether its any good or not and the most straightforward and intuitive metric for a supervised classifier's performance is accuracy. Unfortunately, there are circumstances where simple accuracy does not work well. For example, with a disease that only affects 1 in a million people, a completely bogus screening test that always reports "negative" will be 99.9999% accurate. Unlike accuracy, ROC curves are less sensitive to class imbalance; the bogus screening test would have an AUC of 0.5, which is like not having a test at all.


## ROC curves¶

**ROC curve (Receiver Operating Characteristic)** is a commonly used way to visualize the performance of a binary classifier and AUC (Area Under the ROC Curve) is used to summarize its performance in a single number. Most machine learning algorithms have the ability to produce probability scores that tells us the strength in which it thinks a given observation is positive. Turning these probability scores into yes or no predictions requires setting a threshold; cases with scores above the threshold are classified as positive, and vice versa. Different threshold values can lead to different result:

* A higher threshold is more conservative about labeling a case as positive; this makes it less likely to produce false positive (an observation that has a negative label but gets classified as positive by the model) results but more likely to miss cases that are in fact positive (lower true positive rate)

* A lower threshold produces positive labels more liberally, so it creates more false positives but also generate more true positives


