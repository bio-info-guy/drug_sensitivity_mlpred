import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import os

def plot_roc_aupr_curves(best_model, X_test, y_test, drug, output_dir=".", threshold_type='best_precision'):
    """
    Plots ROC and AUPR curves and saves them to the specified output directory.

    Args:
        best_model: The trained model.
        X_test: Test features.
        y_test: Test labels.
        drug: Name of the drug for plot titles and filenames.
        output_dir: Directory to save the plots.
        threshold_type: Type of threshold to highlight on AUPR curve ('best_f1' or 'best_precision').
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get predicted probabilities for the positive class
    if hasattr(best_model, "predict_proba"):
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        y_pred_proba = best_model.decision_function(X_test)
    else:
        raise AttributeError("Model does not have predict_proba or decision_function method.")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {drug}')
    plt.legend(loc="lower right")

    # AUPR Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    aupr_score = average_precision_score(y_test, y_pred_proba)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'AUPR curve (area = {aupr_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve for {drug}')
    plt.legend(loc="lower left")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    # Find the point where recall is maximized before precision drops significantly
    # This is a heuristic, you might need to adjust the threshold for "significant drop"
    # For simplicity, let's find the point where precision is still high (e.g., > 0.8) and recall is maximized
    # Or, a common approach is to find the threshold that maximizes F1-score

    # Calculate F1-scores and find the best F1-score point
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # Add epsilon to avoid division by zero
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_f1_idx]
    best_recall_f1 = recall[best_f1_idx]
    best_precision_f1 = precision[best_f1_idx]

    # Find the point with highest precision and corresponding recall
    # Note: precision and recall arrays are sorted by increasing recall.
    # To find the highest precision before a drop, we can iterate backwards or find max.
    # The user's request "find highest precision level first, then find the highest recall level for that precision level"
    # implies finding the point on the curve where precision is maximal.
    best_precision_idx = np.argmax(precision)
    best_threshold_precision = thresholds[best_precision_idx]
    best_recall_precision = recall[best_precision_idx]
    best_precision_precision = precision[best_precision_idx]

    # Select which threshold to highlight based on threshold_type
    if threshold_type == 'best_f1':
        highlight_threshold = best_threshold_f1
        highlight_recall = best_recall_f1
        highlight_precision = best_precision_f1
        label_text = f'Best F1-score point (Thresh={highlight_threshold:.2f})'
    elif threshold_type == 'best_precision':
        highlight_threshold = best_threshold_precision
        highlight_recall = best_recall_precision
        highlight_precision = best_precision_precision
        label_text = f'Highest Precision point (Thresh={highlight_threshold:.2f})'
    else:
        raise ValueError("Invalid threshold_type. Must be 'best_f1' or 'best_precision'.")

    plt.plot(highlight_recall, highlight_precision, 'o', markersize=8, color='red',
             label=label_text)
    plt.axvline(x=highlight_recall, color='gray', linestyle='--', lw=1)
    plt.axhline(y=highlight_precision, color='gray', linestyle='--', lw=1)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{drug}_roc_aupr_curves.png'))
    plt.close()

if __name__ == '__main__':
    # Example usage (for testing purposes)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
                               n_classes=2, random_state=699)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Plot curves for best F1-score threshold
    plot_roc_aupr_curves(model, X_test, y_test, drug="ExampleDrug_BestF1", output_dir="./test_plots", threshold_type='best_f1')
    print("Plots with Best F1-score threshold saved to ./test_plots")

    # Plot curves for highest precision threshold
    plot_roc_aupr_curves(model, X_test, y_test, drug="ExampleDrug_BestPrecision", output_dir="./test_plots", threshold_type='best_precision')
    print("Plots with Highest Precision threshold saved to ./test_plots")
