
"""
Week 4: Machine Learning Models for Stock Prediction
WiDS 2025 Project

This script builds and evaluates multiple ML models:
- Logistic Regression
- Random Forest
- XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, install if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not installed. Will use Logistic Regression and Random Forest only.")


class StockPredictionModel:
    """Base class for stock prediction models"""

    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def prepare_features(self, df, feature_cols, target_col='Target_Next_Day'):
        """Prepare features and target for modeling"""
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle any remaining NaN values
        X = X.fillna(X.mean())

        return X, y

    def train(self, X_train, y_train):
        """Train the model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }

        return metrics, y_pred, y_pred_proba


def select_features(df):
    """Select relevant features for modeling"""
    feature_cols = [
        # Price features
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Price_Change_Pct', 'Volatility',

        # Momentum features
        'Momentum_1d', 'Momentum_3d',

        # Moving averages
        'MA_5', 'MA_10',

        # Volume features
        'Volume_Change',

        # Sentiment features
        'Sentiment_Mean', 'Sentiment_Std', 'News_Count',
        'Positive_Mean', 'Negative_Mean',
        'Sentiment_Lag1', 'News_Count_Lag1'
    ]

    # Only keep features that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]

    return available_features


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression model"""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION")
    print("="*60)

    model = StockPredictionModel("Logistic Regression")
    model.model = LogisticRegression(random_state=42, max_iter=1000)

    model.train(X_train, y_train)
    metrics, y_pred, y_pred_proba = model.evaluate(X_test, y_test)

    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return model, metrics, y_pred, y_pred_proba


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model"""
    print("\n" + "="*60)
    print("RANDOM FOREST")
    print("="*60)

    model = StockPredictionModel("Random Forest")
    model.model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    model.train(X_train, y_train)
    metrics, y_pred, y_pred_proba = model.evaluate(X_test, y_test)

    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Feature importance
    model.feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 5 Important Features:")
    print(model.feature_importance.head().to_string(index=False))

    return model, metrics, y_pred, y_pred_proba


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model"""
    if not XGBOOST_AVAILABLE:
        print("\nXGBoost not available - skipping")
        return None, None, None, None

    print("\n" + "="*60)
    print("XGBOOST")
    print("="*60)

    model = StockPredictionModel("XGBoost")
    model.model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.train(X_train, y_train)
    metrics, y_pred, y_pred_proba = model.evaluate(X_test, y_test)

    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Feature importance
    model.feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 5 Important Features:")
    print(model.feature_importance.head().to_string(index=False))

    return model, metrics, y_pred, y_pred_proba


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion_matrix_{model_name.replace(' ', '_')}.png")


def plot_roc_curve(y_test, y_pred_proba, model_name):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved roc_curve_{model_name.replace(' ', '_')}.png")


def plot_feature_importance(model):
    """Plot feature importance"""
    if model.feature_importance is None:
        return

    top_features = model.feature_importance.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importance - {model.name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model.name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved feature_importance_{model.name.replace(' ', '_')}.png")


def compare_models(results):
    """Create comparison of all models"""
    comparison_df = pd.DataFrame(results).T

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print("\n" + comparison_df.to_string())

    # Plot comparison
    comparison_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved model_comparison.png")

    return comparison_df


def main():
    """Main execution function"""
    print("="*60)
    print("Week 4: Machine Learning Models")
    print("="*60 + "\n")

    # Load ML-ready dataset
    try:
        df = pd.read_csv('ml_ready_dataset.csv')
        print(f"✓ Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        print("Error: ml_ready_dataset.csv not found!")
        print("Please run week4_sentiment_features.py first")
        return

    # Select features
    feature_cols = select_features(df)
    print(f"✓ Selected {len(feature_cols)} features")

    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['Target_Next_Day']

    # Split data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution: {dict(y.value_counts())}")

    # Train models
    results = {}

    # 1. Logistic Regression
    lr_model, lr_metrics, lr_pred, lr_proba = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    results['Logistic Regression'] = lr_metrics
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
    plot_roc_curve(y_test, lr_proba, "Logistic Regression")

    # 2. Random Forest
    rf_model, rf_metrics, rf_pred, rf_proba = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    results['Random Forest'] = rf_metrics
    plot_confusion_matrix(y_test, rf_pred, "Random Forest")
    plot_roc_curve(y_test, rf_proba, "Random Forest")
    plot_feature_importance(rf_model)

    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_metrics, xgb_pred, xgb_proba = train_xgboost(
            X_train, y_train, X_test, y_test
        )
        if xgb_metrics:
            results['XGBoost'] = xgb_metrics
            plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
            plot_roc_curve(y_test, xgb_proba, "XGBoost")
            plot_feature_importance(xgb_model)

    # Compare models
    comparison_df = compare_models(results)
    comparison_df.to_csv('model_comparison_results.csv')
    print("✓ Saved model_comparison_results.csv")

    # Save best model info
    best_model = comparison_df['Accuracy'].idxmax()
    best_accuracy = comparison_df['Accuracy'].max()

    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print("="*60)

    print("\n✓ Week 4 Modeling Complete!")
    print("\nGenerated files:")
    print("  - model_comparison_results.csv")
    print("  - confusion_matrix_*.png")
    print("  - roc_curve_*.png")
    print("  - feature_importance_*.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    # Install required packages if needed
    packages = {
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn'
    }

    import subprocess
    import sys

    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

    # Try to install XGBoost
    try:
        import xgboost
    except ImportError:
        print("Note: XGBoost not installed. Install with: pip install xgboost")

    main()
