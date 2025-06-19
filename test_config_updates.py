#!/usr/bin/env python3
"""
Test script to verify the config updates work correctly.
"""

import sys
import os
sys.path.append('/home/yangqi.su/home/repos/drug_sensitivity_mlpred/')
sys.path.append('/home/yangqi.su/home/repos/drug_sensitivity_mlpred/scripts/')
def test_config_loading():
    """Test loading different search method configurations."""
    from utils.config_loader import load_config
    
    print("Testing config loading...")
    
    # Test loading the config
    config = load_config('configs/xgb_config.json')
    print(f"✓ Config loaded successfully!")
    print(f"  Search method: {config['search_method']}")
    print(f"  Available search params: {list(config['search_params'].keys())}")
    
    # Test different search methods
    search_methods = ['gridcv', 'halvingrandomsearch', 'optuna']
    
    for method in search_methods:
        print(f"\nTesting search method: {method}")
        
        # Create a temporary config with this search method
        test_config = config.copy()
        test_config['search_method'] = method
        
        # Test the search parameter selection logic
        if method == 'gridcv':
            search_params_key = 'grid_search_params'
        elif method == 'halvingrandomsearch':
            search_params_key = 'halving_search_params'
        elif method == 'optuna':
            search_params_key = 'optuna_search_params'
        
        if search_params_key in config:
            print(f"  ✓ {search_params_key} found in config")
            print(f"  Parameters: {list(config[search_params_key].keys())}")
            
            # Test optuna distribution parsing
            if method == 'optuna':
                print("  Testing optuna distributions:")
                for param, value in config[search_params_key].items():
                    if isinstance(value, str) and 'optuna.distributions.' in value:
                        print(f"    {param}: {value}")
        else:
            print(f"  ✗ {search_params_key} not found in config")

def test_optuna_distributions():
    """Test optuna distribution parsing."""
    print("\n" + "="*50)
    print("Testing optuna distribution parsing...")
    
    try:
        from utils.config_loader import _parse_optuna_distribution, OPTUNA_AVAILABLE
        
        if not OPTUNA_AVAILABLE:
            print("  ⚠ Optuna not available, skipping optuna-specific tests")
            return
        
        # Test different distribution types
        test_distributions = [
            "optuna.distributions.FloatDistribution(0.01, 0.1)",
            "optuna.distributions.IntDistribution(5, 20)",
            "optuna.distributions.CategoricalDistribution(['a', 'b', 'c'])"
        ]
        
        for dist_str in test_distributions:
            try:
                dist = _parse_optuna_distribution(dist_str)
                print(f"  ✓ Parsed: {dist_str} -> {type(dist).__name__}")
            except Exception as e:
                print(f"  ✗ Failed to parse: {dist_str} -> {e}")
                
    except ImportError as e:
        print(f"  ⚠ Could not import optuna functions: {e}")

def test_model_imports():
    """Test that all required model imports work."""
    print("\n" + "="*50)
    print("Testing model imports...")
    
    try:
        from scripts.skl_train_model import MODEL_TYPES, OVERSAMPLER_TYPES
        print("  ✓ Model types imported successfully")
        print(f"    Available models: {list(MODEL_TYPES.keys())}")
        print(f"    Available oversamplers: {list(OVERSAMPLER_TYPES.keys())}")
        
        # Test optuna import
        try:
            from optuna.integration import OptunaSearchCV
            print("  ✓ OptunaSearchCV imported successfully")
        except ImportError:
            print("  ⚠ OptunaSearchCV not available (optuna not installed)")
            
    except Exception as e:
        print(f"  ✗ Error importing model components: {e}")

if __name__ == "__main__":
    print("="*60)
    print("CONFIG UPDATE VERIFICATION TEST")
    print("="*60)
    
    try:
        test_config_loading()
        test_optuna_distributions()
        test_model_imports()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
