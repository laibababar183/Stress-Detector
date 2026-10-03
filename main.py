import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. LOAD DATASET
print("Loading dataset...")
train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')

# Combine (~410k rows)
df = pd.concat([train_df, test_df], ignore_index=True)
df = df.dropna()

# Target Column
target_col = 'condition' if 'condition' in df.columns else df.columns[-1]

# 2. SELECT TOP 3 FEATURES (Paper ke Table 2 ke mutabiq)
top_3_features = ['RMSSD', 'MEAN_RR', 'SDRR']
available_cols = [c for c in df.columns if c.upper() in [f.upper() for f in top_3_features]]

if len(available_cols) == 3:
    X = df[available_cols]
    print(f"Selected Top 3 Features: {available_cols}")
else:
    X = df[[c for c in df.columns if any(f.lower() in c.lower() for f in ['rmssd', 'mean_rr', 'sdrr'])]]
    print(f"Features selected: {X.columns.tolist()}")

y = df[target_col]

# Encoding & Scaling
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 80/20 TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# 4. MODEL 1: DECISION TREE
print("\nTraining Model 1: Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=12, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_preds)
dt_prec = precision_score(y_test, dt_preds, average='weighted')
dt_rec = recall_score(y_test, dt_preds, average='weighted')
dt_f1 = f1_score(y_test, dt_preds, average='weighted')

# 5. MODEL 2: LINEAR REGRESSION
print("Training Model 2: Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Linear Regression continuous numbers predict karta hai, 
# isliye rounding and clipping se classification labels (0, 1, 2) banaye gaye hain:
raw_preds = lr_model.predict(X_test)
lr_preds = np.clip(np.round(raw_preds), 0, len(le.classes_) - 1).astype(int)

lr_acc = accuracy_score(y_test, lr_preds)
lr_prec = precision_score(y_test, lr_preds, average='weighted', zero_division=0)
lr_rec = recall_score(y_test, lr_preds, average='weighted')
lr_f1 = f1_score(y_test, lr_preds, average='weighted')

# 6. PERFORMANCE COMPARISON TABLE
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Decision Tree': [dt_acc, dt_prec, dt_rec, dt_f1],
    'Linear Regression': [lr_acc, lr_prec, lr_rec, lr_f1]
})

print("\n================ MODEL PERFORMANCE COMPARISON ================")
print(comparison_df.to_string(index=False))

# 7. SAVE BEST MODEL AS .PKL FILE
best_model = dt_model if dt_f1 > lr_f1 else lr_model

with open('stress_detection_top3_model.pkl', 'wb') as file:
    pickle.dump({'model': best_model, 'scaler': scaler, 'encoder': le, 'features': list(X.columns)}, file)

print("\nBest model successfully saved as 'stress_detection_top3_model.pkl'!")