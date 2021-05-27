import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from process import Preprocess
from binary import _binary_clf_curve
import matplotlib.patches as patches


filename='HR_comma_sep.csv'
data = pd.read_csv(filename)



label_col='left'

label=data[label_col].values

data=data.drop(label_col,axis=1)
print('labels distribution:', np.bincount(label) / label.size)

test_size = 0.2

random_state = 1234
data_train, data_test, y_train, y_test = train_test_split(
    data, label, test_size = test_size, random_state = random_state, stratify = label)


num_col=['satisfaction_level','last_evaluation','number_project','average_montly_hours',
    'time_spend_company']

cat_col=['Work_accident']

preprocess = Preprocess(num_col, cat_col)


X_train = preprocess.fit_transform(data_train)
X_test = preprocess.transform(data_test)


tree=RandomForestClassifier(max_depth=4)
tree.fit(X_train,y_train)



y_true=np.array([1,0,1,0,1])
y_score = np.array([0.45, 0.4, 0.35, 0.35, 0.8])

tps, fps, thresholds = _binary_clf_curve(y_true, y_score)

tpr = np.hstack((0, tps / tps[-1]))
fpr = np.hstack((0, fps / fps[-1]))


plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['font.size'] = 12

fig = plt.figure()
plt.plot(fpr, tpr, marker = 'o', lw = 1)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')

def _roc_auc_score(y_true, y_score):
    

    # ensure the target is binary
    if np.unique(y_true).size != 2:
        raise ValueError('Only two class should be present in y_true. ROC AUC score '
                         'is not defined in that case.')
    
    tps, fps, _ = _binary_clf_curve(y_true, y_score)

    # convert count to rate
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # compute AUC using the trapezoidal rule;
    # appending an extra 0 is just to ensure the length matches
    zero = np.array([0])
    tpr_diff = np.hstack((np.diff(tpr), zero))
    fpr_diff = np.hstack((np.diff(fpr), zero))
    auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
    return auc

auc_score = _roc_auc_score(y_true, y_score)
print('auc score:', auc_score)

tree_test_pred = tree.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, tree_test_pred, pos_label = 1)

# AUC score that summarizes the ROC curve
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw = 2, label = 'ROC AUC: {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1],
         linestyle = '--',
         color = (0.6, 0.6, 0.6),
         label = 'random guessing')
plt.plot([0, 0, 1], [0, 1, 1],
         linestyle = ':',
         color = 'black', 
         label = 'perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()