import os
import sys
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score, roc_auc_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse # Added argparse
from tqdm import tqdm # Added tqdm

# Add the parent directory to sys.path to import custom modules
# Assuming the script is run from the root of the drug_sensitivity_mlpred repository
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
sys.path.append(os.path.join(os.getcwd()))

from customio import read_data
from utils.config_loader import load_config

def summarize_drug_models(model_root_dir, data_file_path, main_config_file_path, num_drugs_to_process=None):
    """
    Summarizes nested cross-validation and test set results for models of a single type.

    Args:
        model_root_dir (str): Path to the root directory containing drug model subfolders.
        data_file_path (str): Path to the main input data file (X and Y).
        main_config_file_path (str): Path to the single config file for this model type.
        num_drugs_to_process (int, optional): If specified, only process this many drug folders. Defaults to None (all).

    Returns:
        pd.DataFrame: A DataFrame containing summarized results for all drugs of this model type.
    """
    all_results = []

    # Load the main data file
    X_all, Y_all, drugs_list = read_data(data_file_path)

    # Load the main config file once
    main_config = load_config(main_config_file_path, n_cores=1, device='cpu') # n_cores and device are placeholders
    model_type = main_config.get('model_type', 'Unknown')

    drug_folders = sorted([d for d in os.listdir(model_root_dir) if os.path.isdir(os.path.join(model_root_dir, d))])

    if num_drugs_to_process is not None:
        drug_folders = drug_folders[:num_drugs_to_process]

    train_test_split_seed = main_config.get('train_test_split_seed', 20) # Default to 20 if not found
    X_train, X_test, Y_train, Y_test = train_test_split(
                X_all, Y_all, test_size=0.2, random_state=train_test_split_seed
            )
    
    if main_config.get('pca',False):
        from sklearn.decomposition import PCA, TruncatedSVD
        n_components = min(X_train.shape[0], X_train.shape[1])
        # There is quite a difference between scaling before PCA
        #pca_model = Pipeline([('scaler', StandardScaler()), ('pca', TruncatedSVD(n_components=n_components))])
        pca_model = TruncatedSVD(n_components=n_components)
 
        X_train = pca_model.fit_transform(X_train)
        X_test = pca_model.transform(X_test)
        # Convert X_train and X_test back to DataFrame to maintain column names for feature importance
        # This is a simplification, as PCA transforms to a new feature space.
        # The feature importance calculation will need to handle this.
        X_train = pd.DataFrame(X_train, columns=[f'PC_{i}' for i in range(X_train.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f'PC_{i}' for i in range(X_test.shape[1])])

    for drug_folder_name in tqdm(drug_folders, desc=f"Processing {model_type} drug models"):
        drug_folder_path = os.path.join(model_root_dir, drug_folder_name)
        
        # Extract drug name from folder name (e.g., "BRD-A00077618-236-07-6::2.5::HTS")
        drug_name_full = drug_folder_name
        
        # Find cv_results.csv
        cv_results_file = None
        for f in os.listdir(drug_folder_path):
            if '_cv_results_' in f:
                cv_results_file = os.path.join(drug_folder_path, f)
                break
        
        if not cv_results_file:
            print(f"Warning: No cv_results.csv found in {drug_folder_path}. Skipping.")
            continue

        # Find joblib model file
        model_file = None
        for f in os.listdir(drug_folder_path):
            if f.endswith('.joblib'):
                model_file = os.path.join(drug_folder_path, f)
                break
        
        if not model_file:
            print(f"Warning: No joblib model found in {drug_folder_path}. Skipping.")
            continue

        try:
            # Load cv_results
            cv_df = pd.read_csv(cv_results_file)
            
            # Extract nested CV stats (mean of test scores)
            cv_stats = {
                'drug': drug_name_full,
                'model_type': model_type, # Use the model_type from the main config
                'cv_balanced_accuracy_mean': cv_df['test_balanced_accuracy'].mean(),
                'cv_precision_mean': cv_df['test_precision'].mean(),
                'cv_recall_mean': cv_df['test_recall'].mean(),
                'cv_f1_mean': cv_df['test_f1'].mean(),
                'cv_average_precision_mean': cv_df['test_average_precision'].mean(),
                'cv_roc_auc_mean': cv_df['test_roc_auc'].mean(),
            }

            # Load model
            model = joblib.load(model_file)

            # Prepare data for testing
            # Ensure the drug_name_full matches a column in Y_all
            if drug_name_full not in Y_all.columns:
                print(f"Warning: Drug '{drug_name_full}' not found in data file. Skipping.")
                continue
            
            y_test = Y_test[drug_name_full]
            
            # Use the train_test_split_seed from the main config
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get predicted probabilities for the positive class
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                raise AttributeError("Model does not have predict_proba or decision_function method.")

            # Calculate test metrics
            test_metrics = {
                'test_precision': precision_score(y_test, y_pred, zero_division=0),
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_f1': f1_score(y_test, y_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_pred, zero_division=0),
                'test_roc_auc': roc_auc_score(y_test, y_proba),
                'test_average_precision': average_precision_score(y_test, y_proba) # Added average precision score
            }

            # Calculate best F1 and best Precision from PR curve
            precision_pr, recall_pr, thresholds_pr = precision_recall_curve(y_test, y_proba)
            
            # Best F1 Score
            f1_scores_pr = 2 * (precision_pr * recall_pr) / (precision_pr + recall_pr + 1e-10)
            test_metrics['test_best_f1_score'] = np.max(f1_scores_pr)

            # Best Precision Score
            test_metrics['test_best_precision_score'] = np.max(precision_pr[:-1]) # Exclude the last point where recall is 0

            # Combine all results
            combined_entry = {**cv_stats, **test_metrics}

            # --- Feature Importance Plotting ---
            feature_importance_filename_pattern = f"{model_type}_feature_importance_{drug_name_full}.csv"
            feature_importance_file = None
            for f in os.listdir(drug_folder_path):
                if f == feature_importance_filename_pattern:
                    feature_importance_file = os.path.join(drug_folder_path, f)
                    break
            
            if False:
                try:
                    fi_df = pd.read_csv(feature_importance_file)
                    # Ensure 'Feature' and 'Importance' columns exist
                    if 'Feature' in fi_df.columns and 'Importance' in fi_df.columns:
                        fi_output_dir = os.path.join(drug_folder_path, 'feature_importance_plots')
                        plot_feature_importance(fi_df, drug_name_full, model_type, fi_output_dir)
                    else:
                        print(f"Warning: Feature importance file {feature_importance_file} missing 'Feature' or 'Importance' columns. Skipping plot.")
                except Exception as fi_e:
                    print(f"Error reading or plotting feature importance for {feature_importance_file}: {fi_e}")
            #else:
             #   print(f"No feature importance file found for {drug_name_full} with pattern {feature_importance_filename_pattern}. Skipping plot.")
            # --- End Feature Importance Plotting ---

            all_results.append(combined_entry)

        except Exception as e:
            print(f"Error processing {drug_folder_path}: {e}")
            continue

    return pd.DataFrame(all_results)

def compare_model_types(model_configs, data_file_path, model_summary = None, output_dir='model_comparison_plots', num_drugs_to_process=None, plot_type='boxplot'):
    """
    Compares results from multiple model types and generates box plots.

    Args:
        model_configs (dict): A dictionary where keys are model type names (e.g., 'SGDClassifier')
                              and values are dictionaries containing 'model_root_dir' and 'main_config_file_path'.
        data_file_path (str): Path to the main input data file (X and Y).
        output_dir (str): Directory to save the comparison plots.
        num_drugs_to_process (int, optional): If specified, only process this many drug folders per model type.
                                              Defaults to None (all).
        plot_type (str, optional): Type of plot to generate ('boxplot' or 'violinplot'). Defaults to 'boxplot'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_dfs = {}
    common_drugs = None
    if model_summary is None:
        for model_type_name, config_paths in tqdm(model_configs.items(), desc="Summarizing all model types"):
            print(f"Processing {model_type_name} models...")
            df = summarize_drug_models(
                config_paths['model_root_dir'],
                data_file_path,
                config_paths['main_config_file_path'],
                num_drugs_to_process=num_drugs_to_process
            )
            print(df)
            all_dfs[model_type_name] = df
        
            current_drugs = set(df['drug'].unique())
            if common_drugs is None:
                common_drugs = current_drugs
            else:
                common_drugs = common_drugs.intersection(current_drugs)
    
        if not common_drugs:
            print("No common drugs found across all model types. Cannot perform comparison. Exiting.")
            return
        print(len(common_drugs))
        all_models_df = pd.DataFrame()
        for model_type_name, df in all_dfs.items():
            filtered_df = df[df['drug'].isin(common_drugs)].copy()
            all_models_df = pd.concat([all_models_df, filtered_df], ignore_index=True)
    
        if all_models_df.empty:
            print("No data to compare after filtering for common drugs. Exiting comparison.")
            return
        
        # Save the summarized data if it was generated
        summary_output_path = os.path.join(output_dir, 'all_models_summary.csv')
        all_models_df.to_csv(summary_output_path, index=False)
        print(f"Summarized model data saved to {summary_output_path}")

    else:
        all_models_df = pd.read_csv(model_summary)
        print(f"Loaded model summary from {model_summary}")

    # Melt the DataFrame for easier plotting
    # CV metrics
    cv_metrics = [col for col in all_models_df.columns if col.startswith('cv_')]
    cv_melted_df = all_models_df.melt(id_vars=['drug', 'model_type'], value_vars=cv_metrics, 
                                      var_name='metric', value_name='score')

    # Test metrics
    test_metrics = [col for col in all_models_df.columns if col.startswith('test_') and not col.startswith('test_roc_auc')] # Exclude roc_auc for now if it's not directly comparable
    test_melted_df = all_models_df.melt(id_vars=['drug', 'model_type'], value_vars=test_metrics, 
                                        var_name='metric', value_name='score')

    # Determine the number of unique metrics for palette generation
    # Assuming cv_metrics and test_metrics have the same number of elements for hue
    num_metrics = len(cv_metrics)
    base_palette = sns.color_palette("tab10") # Changed to a more contrasting palette

    # Plot 1: Nested Cross-Validation Metrics
    plt.figure(figsize=(14, 9)) # Slightly larger figure size
    if plot_type == 'boxplot':
        sns.boxplot(data=cv_melted_df, x='model_type', y='score', hue='metric', palette=base_palette)
    elif plot_type == 'violinplot':
        sns.violinplot(data=cv_melted_df, x='model_type', y='score', hue='metric', palette=base_palette, inner="quartile", alpha=0.6) # Lighter violin plots, with quartile lines
        sns.stripplot(data=cv_melted_df, x='model_type', y='score', hue='metric', palette=base_palette, jitter=0.2, dodge=True, size=1.5, alpha=0.9, legend=False) # Stripplot, matching colors, slightly darker
    
    plt.title('Comparison of Nested Cross-Validation Metrics Across Model Types', fontsize=16) # Increased title font size
    plt.xlabel('Model Type', fontsize=14) # Increased x-label font size
    plt.ylabel('Score', fontsize=14) # Increased y-label font size
    plt.xticks(rotation=0, ha='center', fontsize=14) # Increased x-tick font size
    plt.yticks(fontsize=12) # Increased y-tick font size
    plt.legend(title='Metric', bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=3, fontsize=14, title_fontsize=16, columnspacing=1, handletextpad=0.5) # Moved legend to bottom, 2 rows of 3 columns
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cv_metrics_comparison_{plot_type}.png'), bbox_inches='tight') # Added bbox_inches='tight', dynamic filename
    plt.close()

    # Plot 2: Best Model Test Metrics
    plt.figure(figsize=(14, 9)) # Slightly larger figure size
    if plot_type == 'boxplot':
        sns.boxplot(data=test_melted_df, x='model_type', y='score', hue='metric', palette=base_palette)
    elif plot_type == 'violinplot':
        sns.violinplot(data=test_melted_df, x='model_type', y='score', hue='metric', palette=base_palette, inner="quartile", alpha=0.6) # Lighter violin plots, with quartile lines
        sns.stripplot(data=test_melted_df, x='model_type', y='score', hue='metric', palette=base_palette, jitter=0.2, dodge=True, size=1.5, alpha=0.9, legend=False) # Stripplot, matching colors, slightly darker
    
    plt.title('Comparison of Best Model Test Metrics Across Model Types', fontsize=16) # Increased title font size
    plt.xlabel('Model Type', fontsize=14) # Increased x-label font size
    plt.ylabel('Score', fontsize=14) # Increased y-label font size
    plt.xticks(rotation=0, ha='center', fontsize=14) # Removed rotation, increased x-tick font size
    plt.yticks(fontsize=12) # Increased y-tick font size
    plt.legend(title='Metric', bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=3, fontsize=14, title_fontsize=16, columnspacing=1, handletextpad=0.5) # Moved legend to bottom, 2 rows of 3 columns
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_metrics_comparison_{plot_type}.png'), bbox_inches='tight') # Added bbox_inches='tight', dynamic filename
    plt.close()

    print(f"Comparison plots saved to {output_dir}")

def plot_feature_importance(feature_importance_df, drug_name, model_type, output_dir):
    """
    Generates and saves a horizontal bar plot of feature importances.

    Args:
        feature_importance_df (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns.
        drug_name (str): Name of the drug for the plot title and filename.
        model_type (str): Type of the model for the plot title and filename.
        output_dir (str): Directory to save the plot.
    """
    if feature_importance_df.empty:
        print(f"No feature importance data to plot for {drug_name} ({model_type}).")
        return

    # Sort by importance and take top 10
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title(f'Top 10 Feature Importances for {drug_name} ({model_type})', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    plot_filename = f"{model_type}_feature_importance_{drug_name}.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"Saved feature importance plot to {os.path.join(output_dir, plot_filename)}")


def extract_and_copy_top_models_plots(all_models_df, model_configurations, output_base_dir, num_drugs, metric, select_bottom):
    """
    Extracts the top/bottom drug models based on a specified metric for each model type
    and copies their associated ROC PNGs from all model types to a new folder.

    Args:
        all_models_df (pd.DataFrame): DataFrame containing summarized results for all drugs and model types.
        model_configurations (dict): Dictionary with model type names and their root directories.
        output_base_dir (str): Base directory to save the top/bottom drug model plots.
        num_drugs (int): Number of top/bottom drugs to extract.
        metric (str): Metric to use for selecting top/bottom drugs.
        select_bottom (bool): If True, select bottom performing drugs; otherwise, select top.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Ensure the specified metric is available
    if metric not in all_models_df.columns:
        print(f"Error: '{metric}' column not found in all_models_df. Cannot extract top/bottom models.")
        return

    for model_type, config in tqdm(model_configurations.items(), desc=f"Extracting {'bottom' if select_bottom else 'top'} {num_drugs} drugs"):
        print(f"\nProcessing {'bottom' if select_bottom else 'top'} drugs for model type: {model_type}")
        
        # Filter for the current model type and sort by the specified metric
        model_type_df = all_models_df[all_models_df['model_type'] == model_type].copy()
        if model_type_df.empty:
            print(f"No data for model type {model_type}. Skipping.")
            continue
            
        # Sort based on metric and selection type
        ascending_sort = select_bottom # True for bottom (ascending), False for top (descending)
        selected_drugs_for_model_type = model_type_df.sort_values(by=metric, ascending=ascending_sort).head(num_drugs)
        
        if selected_drugs_for_model_type.empty:
            print(f"No {'bottom' if select_bottom else 'top'} drugs found for model type {model_type}. Skipping.")
            continue

        target_dir_name = f"{'bottom' if select_bottom else 'top'}_{num_drugs}_drugs_for_{model_type}"
        target_dir = os.path.join(output_base_dir, target_dir_name)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created target directory: {target_dir}")

        for _, row in tqdm(selected_drugs_for_model_type.iterrows(), desc=f"  Copying plots for {model_type}", leave=False):
            drug_name_full = row['drug']
            print(f"  Processing drug: {drug_name_full}")
            
            # Now, for this selected drug, find its ROC PNG in ALL model type folders
            for other_model_type, other_config in model_configurations.items():
                other_model_root_dir = other_config['model_root_dir']
                drug_folder_path = os.path.join(other_model_root_dir, drug_name_full)
                
                if not os.path.isdir(drug_folder_path):
                    print(f"    Warning: Drug folder not found for {drug_name_full} in {other_model_type}'s directory. Skipping.")
                    continue
                
                found_png = False
                for f in os.listdir(drug_folder_path):
                    if f.endswith('.png') and 'roc' in f.lower():
                        source_png_path = os.path.join(drug_folder_path, f)
                        # Rename the file to include the model type it came from
                        dest_png_name = f"{drug_name_full}_{other_model_type}_{f}"
                        dest_png_path = os.path.join(target_dir, dest_png_name)
                        
                        try:
                            import shutil
                            shutil.copy2(source_png_path, dest_png_path)
                            print(f"    Copied {f} from {other_model_type} to {target_dir}")
                            found_png = True
                        except Exception as e:
                            print(f"    Error copying {source_png_path} to {dest_png_path}: {e}")
                        break # Assuming only one ROC PNG per folder
                
                if not found_png:
                    print(f"    No ROC PNG found for {drug_name_full} in {other_model_type}'s directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize and compare drug sensitivity models.')

    parser.add_argument('--model_root_dirs', type=str, required=True,
                        help='Comma-separated list of model root directories. Each directory should contain drug model subfolders.')
    parser.add_argument('--data_file_in', type=str, default='./input/combined_DepMap_21Q3.csv',
                        help='Path to the main input data file (X and Y).')
    parser.add_argument('--num_drugs_to_process', type=int, default=None,
                        help='If specified, only process this many drug folders per model type. Defaults to None (all).')
    parser.add_argument('--outdir', type=str, default='model_comparison_results',
                        help='Directory to save the comparison plots and summary CSV.')
    parser.add_argument('--num_top_drugs', type=int, default=5,
                        help='Number of top/bottom drug PNGs to extract for each model type.')
    parser.add_argument('--metric_for_top_drugs', type=str, default='test_average_precision',
                        choices=['test_average_precision', 'test_f1', 'test_roc_auc', 'test_precision', 'test_recall'],
                        help='Metric to use for selecting top/bottom drugs.')
    parser.add_argument('--select_bottom_drugs', action='store_true',
                        help='If set, select bottom performing drug PNGs instead of top.')
    parser.add_argument('--plot_type', type=str, default='boxplot', choices=['boxplot', 'violinplot'],
                        help='Type of plot to generate for comparison (boxplot or violinplot).')

    args = parser.parse_args()

    # Parse model root directories and find their main config files
    model_root_dirs_list = [d.strip() for d in args.model_root_dirs.split(',')]
    model_configurations = {}

    for model_root_dir in model_root_dirs_list:
        main_config_file = None
        for f in os.listdir(model_root_dir):
            if f.endswith(('.yaml', '.json')):
                main_config_file = os.path.join(model_root_dir, f)
                break
        
        if not main_config_file:
            print(f"Error: No main config file (yaml or json) found in {model_root_dir}. Skipping.")
            continue
        
        # Infer model_type from the config file
        main_config = load_config(main_config_file, n_cores=1, device='cpu')
        model_type = main_config.get('model_type', os.path.basename(model_root_dir)) # Default to folder name if not found

        if model_type in model_configurations:
            print(f"Warning: Duplicate model_type '{model_type}' found. Using the first occurrence. Consider renaming model_type in configs or providing unique root directories.")
            continue

        model_configurations[model_type] = {
            'model_root_dir': model_root_dir,
            'main_config_file_path': main_config_file
        }
    
    if not model_configurations:
        print("No valid model configurations found. Exiting.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    summary_output_path = os.path.join(args.outdir, 'all_models_summary.csv')
    all_models_summary_df = None

    if os.path.exists(summary_output_path):
        print(f"Found existing summary file at {summary_output_path}. Loading it.")
        all_models_summary_df = pd.read_csv(summary_output_path)
    else:
        print("No existing summary file found. Generating new summary.")
        all_dfs = {}
        common_drugs = None
        for model_type_name, config_paths in tqdm(model_configurations.items(), desc="Summarizing all model types"):
            print(f"Processing {model_type_name} models...")
            df = summarize_drug_models(
                config_paths['model_root_dir'],
                args.data_file_in,
                config_paths['main_config_file_path'],
                num_drugs_to_process=args.num_drugs_to_process
            )
            all_dfs[model_type_name] = df
        
            current_drugs = set(df['drug'].unique())
            if common_drugs is None:
                common_drugs = current_drugs
            else:
                common_drugs = common_drugs.intersection(current_drugs)
    
        if not common_drugs:
            print("No common drugs found across all model types. Cannot perform comparison. Exiting.")
            sys.exit(1)
        
        all_models_summary_df = pd.DataFrame()
        for model_type_name, df in all_dfs.items():
            filtered_df = df[df['drug'].isin(common_drugs)].copy()
            all_models_summary_df = pd.concat([all_models_summary_df, filtered_df], ignore_index=True)
    
        if all_models_summary_df.empty:
            print("No data to compare after filtering for common drugs. Exiting comparison.")
            sys.exit(1)
        
        all_models_summary_df.to_csv(summary_output_path, index=False)
        print(f"Summarized model data saved to {summary_output_path}")

    # Generate comparison plots
    compare_model_types(model_configurations, args.data_file_in,
                        model_summary=summary_output_path, # Pass the path to the summary file
                        output_dir=args.outdir, 
                        num_drugs_to_process=args.num_drugs_to_process,
                        plot_type=args.plot_type)

    # Extract and copy top/bottom model plots
    extract_and_copy_top_models_plots(all_models_summary_df, model_configurations, 
                                      output_base_dir=os.path.join(args.outdir, 'top_bottom_drug_model_plots'),
                                      num_drugs=args.num_top_drugs,
                                      metric=args.metric_for_top_drugs,
                                      select_bottom=args.select_bottom_drugs)
