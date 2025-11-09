# ===================== capstone.py =====================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
import gradio as gr

# ---------------------- Helper Functions ----------------------

def fill_missing_values(df):
    for col in ['gender', 'major_discipline', 'company_type']:
        df.fillna({col: 'Other'}, inplace=True)
    for col in ['education_level', 'experience', 'company_size',
                'last_new_job', 'enrolled_university']:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_outliers_func(df):
    z_scores = zscore(df['city_development_index'])
    df = df[(abs(z_scores) <= 2.5)]

    Q1 = df['training_hours'].quantile(0.25)
    Q3 = df['training_hours'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.0 * IQR
    upper = Q3 + 1.0 * IQR
    df = df[(df['training_hours'] >= lower) & (df['training_hours'] <= upper)]
    return df


def encode_data(df, reference_cols=None):
    unordered = ['gender', 'enrolled_university', 'major_discipline',
                 'company_type', 'city', 'education_level',
                 'experience', 'company_size', 'last_new_job', 'relevent_experience']

    df = pd.get_dummies(df, columns=unordered, drop_first=True)

    # 🧹 Sanitize column names for XGBoost
    df.columns = (
        df.columns.str.replace('[', '(', regex=False)
                  .str.replace(']', ')', regex=False)
                  .str.replace('<', 'lt_', regex=False)
                  .str.replace('>', 'gt_', regex=False)
    )

    if reference_cols is not None:
        # Ensure reference columns also sanitized
        reference_cols = (
            pd.Index(reference_cols)
            .str.replace('[', '(', regex=False)
            .str.replace(']', ')', regex=False)
            .str.replace('<', 'lt_', regex=False)
            .str.replace('>', 'gt_', regex=False)
        )
        for col in reference_cols:
            if col not in df:
                df[col] = 0
        df = df[reference_cols]

    return df


# ---------------------- Load & Clean Training Data ----------------------

loadData = pd.read_csv("aug_train.csv")
loadData = fill_missing_values(loadData)
loadData = remove_outliers_func(loadData)

x = loadData.drop('target', axis=1)
y = loadData['target']

x = encode_data(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------- Handle Imbalance ----------------------

# Detect categorical features for SMOTENC
categorical_features = [i for i, col in enumerate(x.columns) if not np.issubdtype(x[col].dtype, np.number)]
sm = SMOTENC(categorical_features=categorical_features, random_state=42)
X_bal, y_bal = sm.fit_resample(x_train, y_train)

# ---------------------- Train Models ----------------------

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_bal, y_bal)
lin_pred_train = np.round(lin_reg.predict(X_bal))
lin_acc_train = accuracy_score(y_bal, lin_pred_train) * 100

# Logistic Regression
log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_bal, y_bal)
log_pred_train = log_reg.predict(X_bal)
log_acc_train = accuracy_score(y_bal, log_pred_train) * 100

# Class weight for imbalance
count_0, count_1 = np.bincount(y_bal)
scale = count_0 / count_1

# XGBoost (Balanced)
xgb = XGBClassifier(
    n_estimators=60,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    random_state=42
)
xgb.fit(X_bal, y_bal)

xgb_acc_train = accuracy_score(y_bal, xgb.predict(X_bal)) * 100
xgb_acc_test = accuracy_score(y_test, xgb.predict(x_test)) * 100

print(f"\n✅ XGBoost Training Accuracy: {xgb_acc_train:.2f}%")
print(f"✅ XGBoost Test Accuracy: {xgb_acc_test:.2f}%")

# ---------------------- Model Evaluation ----------------------

y_pred = xgb.predict(x_test)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, digits=3))

# ---------------------- Visualization ----------------------

train_models = ['Linear Regression', 'Logistic Regression', 'XGBoost']
train_accuracies = [lin_acc_train, log_acc_train, xgb_acc_train]

lin_pred_test = np.round(lin_reg.predict(x_test))
log_pred_test = log_reg.predict(x_test)
xgb_pred_test = xgb.predict(x_test)

lin_acc_test = accuracy_score(y_test, lin_pred_test) * 100
log_acc_test = accuracy_score(y_test, log_pred_test) * 100
xgb_acc_test = accuracy_score(y_test, xgb_pred_test) * 100

test_models = ['Linear Regression', 'Logistic Regression', 'XGBoost']
test_accuracies = [lin_acc_test, log_acc_test, xgb_acc_test]


def plot_model_accuracies():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(train_models, train_accuracies, color=['skyblue', 'lightgreen', 'orange'])
    axes[0].set_title('Training Accuracy')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 100)
    axes[1].bar(test_models, test_accuracies, color=['lightcoral', 'lightseagreen', 'gold'])
    axes[1].set_title('Test Accuracy')
    axes[1].set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig("static/model_accuracy.png")
    plt.close()

plot_model_accuracies()
print("\n📊 Saved accuracy comparison plot as 'static/model_accuracy.png'")

# ---------------------- TEST DATASET PREDICTION ----------------------

testData = pd.read_csv("aug_test.csv")
testData = fill_missing_values(testData)
testData = remove_outliers_func(testData)

# Store enrollee_id for output
enrollee_ids = testData['enrollee_id']

testData = encode_data(testData, reference_cols=x.columns)

# Predict on test dataset
test_pred = xgb.predict(testData)

# Save predictions
results = pd.DataFrame({
    'enrollee_id': enrollee_ids,
    'prediction': test_pred
})
results.to_csv("test_predictions_with_id.csv", index=False)
print("\n💾 Saved predictions to 'test_predictions_with_id.csv'")

# ---------------------- For API Integration ----------------------

label_encoders = None
encode_ordered_columns = None
