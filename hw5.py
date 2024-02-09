# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
"""
# Homework 5
**Mauricio Vargas-Estrada**
"""

# %% [markdown]
# # 0.) Import the Credit Card Fraud Data From CCLE
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %%
df = pd.read_csv("data/fraudTest.csv")
# %%
df_select = df.copy()
# %%
df_select = df_select[
    ["trans_date_trans_time", "category", "amt", "city_pop", "is_fraud"]
]
# %%
df_select["trans_date_trans_time"] = pd.to_datetime(
    df_select.trans_date_trans_time
)
# %%
df_select["time_var"] = [
    i.second for i in df_select["trans_date_trans_time"]
]
# %%
X = pd.get_dummies(df_select, ["category"])
X = X.drop(["trans_date_trans_time", "is_fraud"], axis = 1)
# %%
y = df["is_fraud"]
# %% [markdown]
# # 1.) Use scikit learn preprocessing to split the data into 70/30 in out of sample
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %%
# Splitting the data between training and testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=999
)
# %%
# Splitting the testing data between testing and holdout.
X_test, X_holdout, y_test, y_holdout = train_test_split(
    X_test, y_test,
    test_size=0.5
)
# %% [markdown]
"""
The process of preprocessing the data will be embedded in the pipeline. This prevents data leakage and involuntary mistakes in the process of evaluating the models.
"""
# %% [markdown]
# # 2.) Make three sets of training data (Oversample, Undersample and SMOTE)
# %%
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# %% [markdown]
"""
Like in the previous questions, the pipeline will be used to prevent data leakage and involuntary mistakes in the process of evaluating the models, so the process of balancing the data will be embedded in those.
"""
# %% [markdown]
# # 3.) Train three logistic regression models
# %%
log_over = Pipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('balancer', RandomOverSampler()),
        ('model', LogisticRegression())
    ]
)
log_over.fit(X_train, y_train)
# %%
log_under = Pipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('balancer', RandomUnderSampler()),
        ('model', LogisticRegression())
    ]
)
log_under.fit(X_train, y_train)
# %%
log_smote = Pipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('balancer', SMOTE()),
        ('model', LogisticRegression())
    ]
)
log_smote.fit(X_train, y_train)
# %% [markdown]
# # 4.) Test the three models
# %% [markdown]
"""
The three model are going to be tested in-sample and out-sample.
"""
# %%
def print_scores(x, y, over, under, smote, title = 'Out of Sample'):
    # Calculating the score
    over = over.score(x, y)
    under = under.score(x, y)
    smote = smote.score(x, y)
    
    temp = f"""
    Test Scores for {title}
    Accuracy
    --------
    - Over Sample: {over:.4f}
    - Under Sample: {under:.4f}
    - SMOTE: {smote:.4f}
    """
    print(temp)
# %% [markdown]
"""
Testing the three models in-sample.
"""
# %%
print_scores(
    X_train, y_train,
    log_over, log_under, log_smote,
    title = 'In-Sample'
)
# %% [markdown]
"""
Testing the three models out-sample.
"""
# %%
print_scores(
    X_test, y_test,
    log_over, log_under, log_smote,
    title = 'Out-Sample'
)
# %% [markdown]
"""
In sample, the under-sampler performs better, but the difference between the three balancing methods is considerable. The conclusion is held in the out of sample metrics. To ensure the performance of the models, a cross-validation would be necessary.
"""
# %%
# We see SMOTE performing with higher accuracy but is ACCURACY really the best measure?
# %% [markdown]
"""
Accuracy is not the best measure for this dataset. We are more concerned about to detect fraud, so we want to maximize the number of true positives and minimize the number of false negatives. In other words, we want to maximize the sensitivity (recall) and minimize the false negative rate.
"""
# %% [markdown]
# # 5.) Which performed best in Out of Sample metrics?
# %%
print_scores(
    X_holdout, y_holdout,
    log_over, log_under, log_smote,
    title = 'HoldOut-Sample'
)
# %% [markdown]
"""
The conclusion is similar using the holdout sample. The under-sampler performs better in-sample, followed by over-sampler and SMOTE. Given the pipelines, the SMOTE balancing method can be tunned using a grid search. 
"""
# %%
from sklearn.metrics import confusion_matrix
# %%
y_true = y_test

# %%
y_pred = log_over.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm
# %%
print("Over Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))
# %%
y_pred = log_under.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm
# %%
print("Under Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))
# %%
y_pred = log_smote.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm
# %%
print("SMOTE Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))

# %% [markdown]
# # 6.) Pick two features and plot the two classes before and after SMOTE.
# %%
X_smote, y_smote = Pipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('balancer', SMOTE())
    ]
).fit_resample(X_train, y_train)
X_smote = pd.DataFrame(X_smote, columns = X_train.columns)
y_smote = pd.Series(y_smote, name = 'is_fraud')

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 12))
sns.scatterplot(
    data=pd.concat([X_train, y_train], axis = 1),
    x='amt',
    y='city_pop',
    hue='is_fraud',
    alpha=0.5,
    ax=ax[0]
)
ax[0].set_title('Before SMOTE')
sns.scatterplot(
    data=pd.concat([X_smote, y_smote], axis = 1),
    x='amt',
    y='city_pop',
    hue='is_fraud',
    alpha=0.5,
    ax=ax[1]
)
ax[1].set_title('After SMOTE')

for a in ax:
    a.set_xlabel('Amount')
    a.set_ylabel('Population')
    a.legend(['Not Fraud', 'Fraud'])
plt.show()
# %% [markdown]
# # 7.) We want to compare oversampling, Undersampling and SMOTE across our 3 models (Logistic Regression, Logistic Regression Lasso and Decision Trees).
#
# # Make a dataframe that has a dual index and 9 Rows.
# # Calculate: Sensitivity, Specificity, Precision, Recall and F1 score. for out of sample data.
# # Notice any patterns across performance for this model. Does one totally out perform the others IE. over/under/smote or does a model perform better DT, Lasso, LR?
# # Choose what you think is the best model and why. test on Holdout

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
# %%
model_configs = {
    'log': LogisticRegression(),
    'lasso': LogisticRegression(
        penalty = 'l1', C = 0.5, solver = 'liblinear'
    ),
    'tree': DecisionTreeClassifier()
}

balancing_configs = {
    'over': RandomOverSampler(),
    'under': RandomUnderSampler(),
    'smote': SMOTE()
}
# %%
trained_models = {}
scores_for_df = {}
# %%
for i,j in balancing_configs.items():
    for k,l in model_configs.items():
        pipe = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('balancer', j),
                ('model', l)
            ]
        )
        pipe.fit(X_train, y_train)
        trained_models[(i,k)] = pipe
        # Compute precision, recall, f1 score and store them in a dictionary
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sensitivity = cm[1,1] /( cm[1,0] + cm[1,1])
        specificity = cm[0,0] /( cm[0,0] + cm[0,1])
        accuracy = pipe.score(X_test, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        scores_for_df[(i,k)] = {
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1': f1,
        }
# %%
pd.DataFrame(scores_for_df).T
# %% [markdown]
"""
In term of balancing method, the Smote method performs better, specially in the decision tree model, evaluating the sensitivity. In terms of models, the decision tree outperforms the logistic regression and the lasso logistic regression, but is less robust given that it happens only with the random undersampler balancing method. 
"""
