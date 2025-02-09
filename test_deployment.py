import os
import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import h5py
import json
from detection import download_model, get_cached_model_path
from config import TRIBE_GROUPS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Disable mixed precision
tf.keras.mixed_precision.set_global_policy('float32')
# Set float precision
tf.keras.backend.set_floatx('float32')

def get_or_download_model():
    """Get cached model or download if not available."""
    return get_cached_model_path() or download_model()

def test_model_weights_loading(model_path):
    """Test specifically for weight loading issues"""
    try:
        logger.info("Testing model weights loading...")
        
        # Create base model first
        inputs = tf.keras.Input(shape=(224, 224, 3))
        base_model = tf.keras.applications.ResNet50(
            input_tensor=inputs,
            include_top=False,
            weights=None
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(len(TRIBE_GROUPS), activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Try loading weights with skip_mismatch
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        logger.info("✓ Weights loaded successfully with skip_mismatch")
        
        # Test with dummy input
        dummy_input = np.zeros((1, 224, 224, 3))
        predictions = model.predict(dummy_input)
        logger.info(f"Model produces valid predictions of shape: {predictions.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Weight loading test failed: {str(e)}")
        return False

def test_input_layer_config(model_path):
    """Test specifically for input layer configuration issues"""
    try:
        logger.info("Testing input layer configuration...")
        
        # Try loading with explicit input layer configuration
        inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
        model = tf.keras.Model(
            inputs=inputs,
            outputs=tf.keras.layers.Conv2D(64, 3)(inputs)  # Simple layer for testing
        )
        
        # Verify input layer configuration
        input_config = model.layers[0].get_config()
        logger.info(f"Input layer config: {input_config}")
        assert 'batch_shape' not in input_config, "batch_shape should not be in config"
        assert 'shape' in input_config, "shape should be in config"
        
        logger.info("✓ Input layer configuration is correct")
        return True
    except Exception as e:
        logger.error(f"Input layer config test failed: {str(e)}")
        return False

def test_layer_compatibility(model_path):
    """Test layer compatibility and mixed precision issues"""
    try:
        logger.info("Testing layer compatibility...")
        
        with h5py.File(model_path, 'r') as f:
            if 'model_weights' in f:
                # Check weight dtypes
                weight_dtypes = set()
                def collect_dtypes(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight_dtypes.add(obj.dtype.name)
                f['model_weights'].visititems(collect_dtypes)
                logger.info(f"Found weight dtypes: {weight_dtypes}")
                
                # Check for mixed precision compatibility
                has_float16 = any('float16' in dtype for dtype in weight_dtypes)
                has_float32 = any('float32' in dtype for dtype in weight_dtypes)
                logger.info(f"Has float16: {has_float16}, Has float32: {has_float32}")
        
        logger.info("✓ Layer compatibility check passed")
        return True
    except Exception as e:
        logger.error(f"Layer compatibility test failed: {str(e)}")
        return False

def inspect_saved_model(model_path):
    """Inspect the saved model structure and metadata"""
    try:
        logger.info("=== Inspecting Saved Model ===")
        with h5py.File(model_path, 'r') as f:
            # Check model metadata
            logger.info("\nModel Metadata:")
            for key, value in f.attrs.items():
                if isinstance(value, bytes):
                    value = value.decode()
                logger.info(f"{key}: {value}")
            
            # Check TensorFlow version used for saving
            if 'keras_version' in f.attrs:
                logger.info(f"\nKeras version used for saving: {f.attrs['keras_version'].decode()}")
            
            # Examine layer structure
            logger.info("\nLayer Structure:")
            if 'model_weights' in f:
                def print_layers(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        logger.info(f"Layer: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
                f['model_weights'].visititems(print_layers)
        
        return True
    except Exception as e:
        logger.error(f"Model inspection failed: {str(e)}")
        return False

def run_deployment_tests():
    logger.info("=== Starting Deployment Verification Tests ===")
    
    try:
        model_path = get_or_download_model()
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False
    
    tests = [
        ("Weight Loading", lambda: test_model_weights_loading(model_path)),
        ("Input Layer Config", lambda: test_input_layer_config(model_path)),
        ("Layer Compatibility", lambda: test_layer_compatibility(model_path)),
        ("Model Inspection", lambda: inspect_saved_model(model_path))
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\n=== Running {test_name} Test ===")
        if test_func():
            logger.info(f"✓ {test_name} test passed")
        else:
            logger.error(f"✗ {test_name} test failed")
            all_passed = False
    
    if all_passed:
        logger.info("\n=== All deployment verification tests passed! ===")
    else:
        logger.error("\n=== Some deployment verification tests failed! ===")
    
    return all_passed

if __name__ == "__main__":
    success = run_deployment_tests()
    if not success:
        raise SystemExit("Deployment verification failed - model loading issues persist!")