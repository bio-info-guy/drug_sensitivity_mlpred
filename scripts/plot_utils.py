import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import os
from sklearn_evaluation.plot import confusion_matrix, ConfusionMatrix


def plot_roc_aupr_curves(best_model, X_test, y_test, drug, output_dir=".", threshold_type='best_precision'):
    """
    Plots ROC and AUPR curves and saves them to the specified output directory.

    Args:
        best_model: The trained model.
        X_test: Test features.
        y_test: Test labels.
        drug: Name of the drug for plot titles and filenames.
        output_dir: Directory to save the plots.
        threshold_type: Type of threshold to highlight on AUPR curve ('best_f1', 'best_precision', or 'both').
        show_plot: Whether to display the plot interactively.
    """
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(best_model, 'named_steps'):
        model_class= str(best_model['classifier']).split('(')[0]
    else:
        model_class= str(best_model).split('(')[0]
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

    # AUPR Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    aupr_score = average_precision_score(y_test, y_pred_proba)



    # Calculate predictions based on default and best precision threshold
    
    fig, axarr = plt.subplots(1, 3, figsize=(15,5))
    #fig = plt.figure(figsize=(10, 5)) # Adjusted figure size for three plots
    st = fig.suptitle(model_class, fontsize="large")

    plt.sca(axarr[0]) # First subplot: ROC Curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {drug}')
    plt.legend(loc="lower right")

    plt.sca(axarr[1]) # Second subplot: AUPR Curve
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
    best_f1_idx = np.argmax(f1_scores[:-1])
    best_threshold_f1 = thresholds[best_f1_idx]
    best_recall_f1 = recall[best_f1_idx]
    best_precision_f1 = precision[best_f1_idx]

    # Find the point with highest precision and corresponding recall
    # Note: precision and recall arrays are sorted by increasing recall.
    # To find the highest precision before a drop, we can iterate backwards or find max.
    # The user's request "find highest precision level first, then find the highest recall level for that precision level"
    # implies finding the point on the curve where precision is maximal.
    best_precision_idx = np.argmax(precision[:-1])
    best_threshold_precision = thresholds[best_precision_idx]
    best_recall_precision = recall[best_precision_idx]
    best_precision_precision = precision[best_precision_idx]

    y_pred_f1 = (y_pred_proba >= best_threshold_f1).astype(int)
    y_pred_threshold = (y_pred_proba >= best_threshold_precision).astype(int)
    # Select which threshold to highlight based on threshold_type
    if threshold_type == 'best_f1' or threshold_type == 'both':
        highlight_threshold = best_threshold_f1
        highlight_recall = best_recall_f1
        highlight_precision = best_precision_f1
        label_text = f'Max F1={np.max(f1_scores):.2f} (Thresh={highlight_threshold:.2f})'
        plt.plot(highlight_recall, highlight_precision, 'o', markersize=8, color='red',
             label=label_text)
        plt.axvline(x=highlight_recall, color='gray', linestyle='--', lw=1)
        plt.axhline(y=highlight_precision, color='gray', linestyle='--', lw=1)
    if threshold_type == 'best_precision' or threshold_type == 'both':
        highlight_threshold = best_threshold_precision
        highlight_recall = best_recall_precision
        highlight_precision = best_precision_precision
        label_text = f'Max Precision={np.max(precision):.2f} (Thresh={highlight_threshold:.2f})'
        plt.plot(highlight_recall, highlight_precision, 'o', markersize=8, color='green',
             label=label_text)
        plt.axvline(x=highlight_recall, color='gray', linestyle='--', lw=1)
        plt.axhline(y=highlight_precision, color='gray', linestyle='--', lw=1)
    
    if threshold_type not in ['best_f1', 'best_precision', 'both']:
        raise ValueError("Invalid threshold_type. Must be 'best_f1', 'best_precision', or 'both'.")
    plt.legend(loc="lower left")

    # Combined Confusion Matrix
    #plt.sca(axarr[2])  # Third subplot
    cm_plt = ConfusionMatrix.from_raw_data(y_test, y_pred_f1)+ConfusionMatrix.from_raw_data(y_test, y_pred_threshold)
    cm_plt.plot(ax=axarr[2])
    plt.sca(axarr[2])
    plt.title(f'Threshold Comparison CM')
    #(cm_default + cm_best_precision).plot(plt.sca(axarr[2]) )
    #plt.title(f'CM (Default + Best Precision Thresh) for {drug}')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, drug, f'{drug}_roc_aupr_cm.png')) # Updated filename
    
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

    os.makedirs("./test_plots/ExampleDrug_BestF1", exist_ok=True)
    # Plot curves for best F1-score threshold
    plot_roc_aupr_curves(model, X_test, y_test, drug="ExampleDrug_BestF1", output_dir="./test_plots", threshold_type='both')
    #print("Plots with Best F1-score threshold saved to ./test_plots")

    # Plot curves for highest precision threshold
    #plot_roc_aupr_curves(model, X_test, y_test, drug="ExampleDrug_BestPrecision", output_dir="./test_plots", threshold_type='best_precision')
    #print("Plots with Highest Precision threshold saved to ./test_plots")
