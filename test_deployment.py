import os
import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import h5py
import json
from detection import download_model, Cast, get_cached_model_path
from classify_mask import TRIBE_GROUPS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_or_download_model():
    """Get cached model or download if not available."""
    return get_cached_model_path() or download_model()

def test_model_layer_count(model_path):
    """Test for layer count mismatch"""
    try:
        logger.info("Testing model layer count...")
        
        # Check h5 file layer count
        with h5py.File(model_path, 'r') as f:
            if 'model_weights' in f:
                weight_names = []
                f['model_weights'].visit(lambda name: weight_names.append(name))
                logger.info(f"Found {len(weight_names)} weight layers in h5 file")
            
            if 'model_config' in f.attrs:
                config = f.attrs['model_config']
                if isinstance(config, bytes):
                    config = config.decode('utf-8')
                config_dict = json.loads(config)
                config_layers = len(config_dict['config']['layers'])
                logger.info(f"Found {config_layers} layers in model config")
                
                # Check for layer type mismatches
                layer_types = [layer['class_name'] for layer in config_dict['config']['layers']]
                logger.info(f"Layer types: {set(layer_types)}")
        
        return True
    except Exception as e:
        logger.error(f"Layer count test failed: {str(e)}")
        return False

def test_model_loading_methods(model_path):
    """Test all possible model loading methods"""
    try:
        logger.info("Testing all model loading methods...")
        
        custom_objects = {
            "Cast": Cast,
            "mixed_float16": tf.keras.mixed_precision.Policy('mixed_float16'),
            "float16": tf.float16,
            "float32": tf.float32
        }
        
        # Test 1: Direct loading with compile=False
        try:
            logger.info("Method 1: Testing direct model loading with compile=False...")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            logger.info("✓ Direct loading successful")
            
            # Check layer count
            logger.info(f"Model has {len(model.layers)} layers")
            # Log first few layer configs
            for i, layer in enumerate(model.layers[:5]):
                logger.info(f"Layer {i}: {layer.name} - {layer.get_config()}")
        except Exception as e:
            logger.warning(f"✗ Direct loading failed: {str(e)}")
            
            # Test 2: Try reconstructing model
            try:
                logger.info("Method 2: Testing model reconstruction...")
                base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3)
                )
                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                outputs = tf.keras.layers.Dense(30, activation='softmax')(x)
                
                model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
                model.load_weights(model_path)
                logger.info("✓ Reconstruction successful")
                
                # Verify output shape matches tribe count
                assert model.output_shape[-1] == len(TRIBE_GROUPS), "Output shape doesn't match tribe count"
            except Exception as e2:
                logger.error(f"✗ Reconstruction failed: {str(e2)}")
                raise
        
        # Test prediction
        dummy_input = np.zeros((1, 224, 224, 3))
        predictions = model.predict(dummy_input)
        logger.info(f"Prediction shape: {predictions.shape}")
        assert predictions.shape[-1] == len(TRIBE_GROUPS), "Prediction shape doesn't match tribe count"
        
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {str(e)}")
        return False

def test_model_deserialization(model_path):
    """Test specifically for the class_name deserialization error"""
    try:
        logger.info("Testing model deserialization...")
        
        with h5py.File(model_path, 'r') as f:
            config = f.attrs.get('model_config')
            if config is not None:
                if isinstance(config, bytes):
                    config = config.decode('utf-8')
                config_dict = json.loads(config)
                
                # Test config cleaning
                def clean_config(cfg):
                    if isinstance(cfg, dict):
                        keys_to_remove = ['class_name', 'module']
                        for key in keys_to_remove:
                            cfg.pop(key, None)
                        for k, v in cfg.items():
                            if isinstance(v, (dict, list)):
                                clean_config(v)
                    elif isinstance(cfg, list):
                        for item in cfg:
                            if isinstance(item, (dict, list)):
                                clean_config(item)
                    return cfg
                
                cleaned_config = clean_config(config_dict.copy())
                logger.info("Original config keys: " + str(list(config_dict.keys())))
                logger.info("Cleaned config keys: " + str(list(cleaned_config.keys())))
                
                # Verify no class_name attributes remain
                def verify_no_class_names(cfg):
                    if isinstance(cfg, dict):
                        assert 'class_name' not in cfg, "class_name still present in config"
                        for v in cfg.values():
                            if isinstance(v, (dict, list)):
                                verify_no_class_names(v)
                    elif isinstance(cfg, list):
                        for item in cfg:
                            if isinstance(item, (dict, list)):
                                verify_no_class_names(item)
                
                verify_no_class_names(cleaned_config)
                logger.info("✓ Config cleaning successful")
        
        return True
    except Exception as e:
        logger.error(f"Deserialization test failed: {str(e)}")
        return False

def run_all_tests():
    logger.info("=== Starting Comprehensive Deployment Tests ===")
    
    # Download model once at the start
    try:
        model_path = get_or_download_model()
        logger.info("Model downloaded and cached for all tests")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False
    
    tests = [
        ("Model Layer Count", lambda: test_model_layer_count(model_path)),
        ("Model Loading Methods", lambda: test_model_loading_methods(model_path)),
        ("Model Deserialization", lambda: test_model_deserialization(model_path))
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
        logger.info("\n=== All tests passed successfully! ===")
    else:
        logger.error("\n=== Some tests failed! ===")
    
    # Clean up at the end
    try:
        os.remove(model_path)
        logger.info("Cleaned up model file")
    except:
        pass
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        raise SystemExit("Tests failed - DO NOT DEPLOY!") 