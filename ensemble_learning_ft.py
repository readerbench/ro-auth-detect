'''
AA ensemble learning classifier 
FT feature dataset 
Feature selection via Kruskal Wallis 
Gridsearch with cross-validation to find the best hyperparameters
'''

import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'fulltext_rbi.csv'
df = pd.read_csv(data_path)
df = df.fillna(0)

index_columns = df.columns[2:]
results = []

for index_column in index_columns:
    groups = [df[index_column][df['author'] == author] for author in df['author'].unique()]

    if any(len(set(group)) > 1 for group in groups):
        stat, p_value = kruskal(*groups)
        results.append((index_column, stat, p_value))

results.sort(key=lambda x: x[1], reverse=True)
top_results = results[:100]
feature_columns = [col[0] for col in top_results]

X = df[feature_columns]
y = df['author']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42)

mlp = MLPClassifier()
xgboost = XGBClassifier()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
extra_trees = ExtraTreesClassifier()
svm = SVC(probability=True)
lr = LogisticRegression()

param_grid_xgboost = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
}

param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

param_grid_extra_trees = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

classifiers_params = {
    'mlp': (mlp, {}),  
    'xgboost': (xgboost, param_grid_xgboost),
    'decision_tree': (decision_tree, param_grid_dt),
    'random_forest': (random_forest, param_grid_dt),
    'extra_trees': (extra_trees, param_grid_extra_trees),
    'svm': (svm, param_grid_svm),
    'lr': (lr, param_grid_lr)
}

best_classifiers = {}
for clf_name, (clf, param_grid) in classifiers_params.items():
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train_encoded)
    best_classifiers[clf_name] = grid_search.best_estimator_

voting_clf = VotingClassifier(estimators=[
    ('mlp', best_classifiers['mlp']),
    ('xgboost', best_classifiers['xgboost']),
    ('decision_tree', best_classifiers['decision_tree']),
    ('random_forest', best_classifiers['random_forest']),
    ('extra_trees', best_classifiers['extra_trees']),
    ('svm', best_classifiers['svm']),
    ('lr', best_classifiers['lr'])
], voting='soft')  # 'soft' for probability voting

voting_clf.fit(X_train, y_train_encoded)
y_pred_encoded = cross_val_predict(voting_clf, X_test, y_test_encoded, cv=5, method='predict_proba')
y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded.argmax(axis=1))
y_pred_decoded_int = label_encoder.transform(y_pred_decoded)

precision = precision_score(y_test_encoded, y_pred_decoded_int, average='weighted')
recall = recall_score(y_test_encoded, y_pred_decoded_int, average='weighted')
f1 = f1_score(y_test_encoded, y_pred_decoded_int, average='weighted')
accuracy = accuracy_score(y_test_encoded, y_pred_decoded_int)
error_rate = 1 - accuracy

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Error Rate: {error_rate:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_decoded_int))

cm = confusion_matrix(y_test_encoded, y_pred_decoded_int)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
