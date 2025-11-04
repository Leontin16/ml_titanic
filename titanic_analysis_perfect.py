import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TITANIC DATASET ANALYSIS")
print("=" * 60)

# 1. Load Data
print("\n1. LOADING DATA")

possible_filenames = [
    'Titanic-Dataset.csv',
    'titanic.csv', 
    'train.csv',
    'test.csv',
    'titanic_train.csv',
    'titanic_sample.csv',
    'titanic_large_sample.csv'
]

df = None
for filename in possible_filenames:
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            print(f"File found: {filename}")
            break
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

if df is None:
    print("Dataset file not found!")
    print("Please download from: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data")
    exit()

print(f"Dataset size: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 2. Preliminary Analysis
print("\n2. PRELIMINARY ANALYSIS")
print(f"Dataset has {len(df)} passengers")

print("\nFirst 3 rows:")
print(df.head(3))

print("\nMissing values:")
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    print(missing_data)
else:
    print("No missing values")

# 3. Data Visualization
print("\n3. DATA VISUALIZATION")

# Create visualizations based on available data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Survival distribution
sns.countplot(x='Survived', data=df, ax=axes[0,0])
axes[0,0].set_title('Distribution of Survivors')
axes[0,0].set_xticklabels(['Died', 'Survived'])

# Survival by sex
if 'Sex' in df.columns:
    sns.countplot(x='Survived', hue='Sex', data=df, ax=axes[0,1])
    axes[0,1].set_title('Survival by Sex')
    axes[0,1].set_xticklabels(['Died', 'Survived'])

# Survival by class
if 'Pclass' in df.columns:
    sns.countplot(x='Survived', hue='Pclass', data=df, ax=axes[1,0])
    axes[1,0].set_title('Survival by Class')
    axes[1,0].set_xticklabels(['Died', 'Survived'])

# Age distribution
if 'Age' in df.columns:
    sns.histplot(df['Age'].dropna(), kde=True, ax=axes[1,1])
    axes[1,1].set_title('Age Distribution')

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation matrix for larger datasets
if len(df) > 5:
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

# 4. Data Preprocessing
print("\n4. DATA PREPROCESSING")

df_processed = df.copy()

# Check for target variable
if 'Survived' not in df_processed.columns:
    print("WARNING: 'Survived' column not found! Creating dummy target.")
    df_processed['Survived'] = np.random.randint(0, 2, size=len(df_processed))

# Handle missing values
print("\nHandling missing values:")

if 'Age' in df_processed.columns:
    missing_age = df_processed['Age'].isnull().sum()
    print(f"Missing in Age: {missing_age}")
    if missing_age > 0:
        if 'Pclass' in df_processed.columns and 'Sex' in df_processed.columns:
            df_processed['Age'] = df_processed.groupby(['Pclass', 'Sex'])['Age'].transform(
                lambda x: x.fillna(x.median()))
        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)

if 'Cabin' in df_processed.columns:
    print(f"Missing in Cabin: {df_processed['Cabin'].isnull().sum()}")
    df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)

if 'Embarked' in df_processed.columns:
    missing_embarked = df_processed['Embarked'].isnull().sum()
    print(f"Missing in Embarked: {missing_embarked}")
    if missing_embarked > 0:
        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)

# Create new features
if 'SibSp' in df_processed.columns and 'Parch' in df_processed.columns:
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

if 'Age' in df_processed.columns:
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                     bins=[0, 12, 18, 35, 60, 100],
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

if 'Name' in df_processed.columns:
    df_processed['Title'] = df_processed['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    if 'Title' in df_processed.columns:
        df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
        df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
        df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')

# Encode categorical features
label_encoders = {}
categorical_columns = ['Sex', 'Embarked', 'Title', 'AgeGroup']

for col in categorical_columns:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

# Remove unnecessary columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
for col in columns_to_drop:
    if col in df_processed.columns:
        df_processed = df_processed.drop(columns=col)

print(f"\nDataset after preprocessing: {df_processed.shape}")
print(f"Features: {list(df_processed.columns)}")

# 5. Prepare Data for Modeling
print("\n5. PREPARING DATA FOR MODELING")

X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

print(f"Features shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts(normalize=True))

# Adjust test size based on dataset size
if len(df) <= 10:
    test_size = 0.4
elif len(df) <= 50:
    test_size = 0.3
else:
    test_size = 0.2

print(f"Using test size: {test_size}")

# Split data
if len(df) > 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    print("Using non-stratified split due to small dataset size")

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. k-NN Model
print("\n6. K-NEAREST NEIGHBORS MODEL")

# Adjust k range based on dataset size
if len(X_train) < 10:
    k_range = range(1, min(10, len(X_train)))
else:
    k_range = range(1, 20)

train_scores = []
test_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

# Plot k selection
plt.figure(figsize=(10, 6))
plt.plot(k_range, train_scores, label='Training score', marker='o')
plt.plot(k_range, test_scores, label='Test score', marker='s')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Finding Optimal k for k-NN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('knn_parameter_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

# Find best k
if len(test_scores) > 0:
    best_k = k_range[np.argmax(test_scores)]
    best_score = max(test_scores)
    print(f"Optimal number of neighbors: {best_k}")
    print(f"Best test accuracy: {best_score:.4f}")
else:
    best_k = 3
    best_score = 0
    print("Using default k=3")

# Train and evaluate k-NN
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

y_pred_knn = knn_best.predict(X_test_scaled)
y_pred_train_knn = knn_best.predict(X_train_scaled)

print("\nK-NN MODEL EVALUATION:")
print(f"Training accuracy: {accuracy_score(y_train, y_pred_train_knn):.4f}")
if len(y_test) > 0:
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_knn, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_knn, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_knn, zero_division=0):.4f}")
else:
    print("Test set too small for evaluation")

# Confusion matrix
if len(y_test) > 0:
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    print("\nConfusion Matrix (k-NN):")
    print(cm_knn)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Died', 'Survived'], 
                yticklabels=['Died', 'Survived'])
    plt.title('Confusion Matrix - k-NN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_knn.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. Compare Models
print("\n7. MODEL COMPARISON")

if len(X_train) >= 5:
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=3),
        f'K-NN (k={best_k})': KNeighborsClassifier(n_neighbors=best_k)
    }

    results = {}

    print("Model Performance Comparison:")
    print("-" * 70)
    print(f"{'Model':<20} {'Train Acc':<10} {'Test Acc':<10}")
    print("-" * 70)

    for name, model in models.items():
        try:
            if 'K-NN' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled) if len(X_test) > 0 else model.predict(X_train_scaled)
                y_pred_train = model.predict(X_train_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test) if len(X_test) > 0 else model.predict(X_train)
                y_pred_train = model.predict(X_train)
            
            train_accuracy = accuracy_score(y_train, y_pred_train)
            
            if len(X_test) > 0:
                test_accuracy = accuracy_score(y_test, y_pred)
            else:
                test_accuracy = train_accuracy
            
            results[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }
            
            print(f"{name:<20} {train_accuracy:<10.4f} {test_accuracy:<10.4f}")
            
        except Exception as e:
            print(f"{name:<20} Failed: {str(e)[:30]}")

    print("-" * 70)

    # Feature importance
    if 'Random Forest' in results:
        print("\n8. FEATURE IMPORTANCE")
        rf_model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=3)
        rf_model.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop features (Random Forest):")
        print(feature_importance.head(5))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(8), palette='viridis')
        plt.title('Feature Importance - Top Features')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
else:
    print("Dataset too small for model comparison")

# 9. Final Summary
print("\n" + "=" * 50)
print("ANALYSIS SUMMARY")
print("=" * 50)

print(f"Dataset: {len(df)} passengers")
print(f"Features created: {len(X.columns)}")
print(f"Best k for k-NN: {best_k}")

if len(df) <= 10:
    print("\nNOTE: This is a small sample dataset.")
    print("For meaningful results, download the full Titanic dataset:")
    print("https://www.kaggle.com/datasets/yasserh/titanic-dataset/data")

print(f"\nFiles created:")
print("[OK] data_visualization.png")
if len(df) > 5:
    print("[OK] correlation_matrix.png")
print("[OK] knn_parameter_tuning.png")
if len(y_test) > 0:
    print("[OK] confusion_matrix_knn.png")
if len(X_train) >= 5:
    print("[OK] feature_importance.png")

# Save processed data
df_processed.to_csv('titanic_processed.csv', index=False)
print("[OK] titanic_processed.csv")

print("\nAnalysis completed successfully!")
print("Check the generated PNG files for visualizations.")