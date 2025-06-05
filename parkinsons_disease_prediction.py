import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from imblearn.over_sampling import SMOTE

# Load and Preprocess the data
df = pd.read_csv("parkinsons_disease.csv")  # Load dataset
df.drop(columns=['name'], inplace=True)  # Drop 'name' column if it exists

X = df.drop(columns=['status'])  # Features
y = df['status']  # Target

# Split and preprocess data (standardize and handle imbalance using SMOTE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Standardize training data
X_test = scaler.transform(X_test)  # Standardize test data

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)  # Handle class imbalance

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'XGBoost': xgb.XGBClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3, n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)
tuned_rf = grid_rf.best_estimator_

# Hyperparameter tuning for SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=3, n_jobs=-1, verbose=1)
grid_svm.fit(X_train, y_train)
tuned_svm = grid_svm.best_estimator_

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10]
}
grid_xgb = GridSearchCV(xgb.XGBClassifier(), param_grid_xgb, cv=3, n_jobs=-1, verbose=1)
grid_xgb.fit(X_train, y_train)

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6, 10]
}
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=3, n_jobs=-1, verbose=1)
grid_gb.fit(X_train, y_train)
tuned_gb = grid_gb.best_estimator_

# Ensembling with Voting Classifier (Soft Voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', tuned_rf),
        ('svm', tuned_svm),
        ('xgb', grid_xgb.best_estimator_),
        ('gb', tuned_gb)
    ],
    voting='soft'  # Use soft voting for probability-based ensemble
)
ensemble_model.fit(X_train, y_train)

# Evaluate models
results = {}

# Function to evaluate models
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Mean CV Accuracy': cross_val_score(model, X_train, y_train, cv=cv).mean()
    }
    results[model_name] = metrics

# Evaluate tuned models
evaluate_model(tuned_rf, 'Random Forest Tuned')
evaluate_model(tuned_svm, 'SVM Tuned')
evaluate_model(grid_xgb.best_estimator_, 'XGBoost Tuned')
evaluate_model(tuned_gb, 'Gradient Boosting Tuned')
evaluate_model(ensemble_model, 'Ensemble Model')

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Performance Summary:")
print(results_df)

# Visualize model comparison metrics
metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=results_df.index, y=results_df[metric], hue=results_df.index,
                palette='coolwarm', legend=False)
    plt.title(f'Model Comparison - {metric}')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# Feature Importance - Gradient Boosting
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tuned_gb.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Feature Importances (Gradient Boosting):")
print(importances.head(10))

# Visualize top 10 feature importances
plt.figure(figsize=(8, 6))
sns.barplot(data=importances.head(10), x='Importance', y='Feature', hue='Feature',
            palette='viridis', legend=False)
plt.title('Top 10 Feature Importances (Gradient Boosting)')
plt.tight_layout()
plt.show()

# Confusion Matrix and ROC Curve - Gradient Boosting
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, tuned_gb.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', "Parkinson's"])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Gradient Boosting')
plt.show()

fpr, tpr, _ = roc_curve(y_test, tuned_gb.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Gradient Boosting (AUC = {roc_auc_score(y_test, tuned_gb.predict_proba(X_test)[:, 1]):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend()
plt.grid(True)
plt.show()
