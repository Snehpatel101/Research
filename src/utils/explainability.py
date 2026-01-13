import numpy as np
from typing import Optional, Any

def compute_shap_values_safe(
    model: Any, 
    X_test: Any, 
    X_background: Any, 
    model_type: str = 'tree'
) -> Optional[np.ndarray]:
    """
    Safely compute SHAP values with error handling.
    
    Args:
        model: Trained model
        X_test: Test data to explain
        X_background: Background data for SHAP
        model_type: 'tree', 'deep', or 'linear'
    
    Returns:
        shap_values: SHAP values array or None if failed
    """
    try:
        import shap
        
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'deep':
            # For deep learning models
            explainer = shap.DeepExplainer(model, X_background)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_background)
        else:
            explainer = shap.KernelExplainer(model.predict, X_background)
            
        shap_values = explainer.shap_values(X_test)
        
        # Handle different output formats (e.g., list for multi-class)
        if isinstance(shap_values, list):
            # For multi-class, typically index 2 is Long (Class 2)
            # Check if we have 3 classes (Short, Hold, Long)
            if len(shap_values) == 3:
                return shap_values[2]
            else:
                return shap_values[-1] # Default to last class
        
        return shap_values
        
    except ImportError:
        print("SHAP library not installed. Skipping explainability.")
        return None
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return None
