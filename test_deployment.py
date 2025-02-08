import os
import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import h5py
import json
from detection import download_model, Cast, model as detection_model
from classify_mask import classify_mask, classification_model, TRIBE_GROUPS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_or_download_model():
    """Download model once and cache it for all tests"""
    cache_dir = os.path.join(tempfile.gettempdir(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached_model_path = os.path.join(cache_dir, "model.h5")
    
    if not os.path.exists(cached_model_path):
        temp_model_path = download_model()
        os.rename(temp_model_path, cached_model_path)
    
    return cached_model_path

def test_model_loading_methods():
    """Test all possible model loading methods"""
    try:
        temp_model_path = get_or_download_model()
        logger.info("Testing all model loading methods...")
        
        custom_objects = {
            "Cast": Cast,
            "mixed_float16": tf.keras.mixed_precision.Policy('mixed_float16'),
            "float16": tf.float16,
            "float32": tf.float32
        }
        
        # Method 1: Direct loading
        try:
            logger.info("Method 1: Testing direct model loading...")
            model = tf.keras.models.load_model(temp_model_path, custom_objects=custom_objects)
            logger.info("✓ Direct loading successful")
        except Exception as e:
            logger.warning(f"✗ Direct loading failed: {str(e)}")
        
        # Method 2: H5 loading with config extraction
        try:
            logger.info("Method 2: Testing H5 loading with config...")
            with h5py.File(temp_model_path, 'r') as f:
                # Test model_config attribute
                if 'model_config' in f.attrs:
                    config = f.attrs['model_config']
                    logger.info(f"Config type: {type(config)}")
                    if isinstance(config, bytes):
                        config = config.decode('utf-8')
                    config_dict = json.loads(config)
                    logger.info(f"Layer count in config: {len(config_dict['config']['layers'])}")
                    logger.info("✓ Config extraction successful")
                else:
                    logger.error("✗ No model_config found in h5 file")
        except Exception as e:
            logger.error(f"✗ H5 loading failed: {str(e)}")
        
        os.remove(temp_model_path)
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {str(e)}")
        return False

def test_model_prediction():
    """Test model prediction functionality"""
    try:
        logger.info("Testing model prediction...")
        
        # Test detection model
        test_input = np.zeros((1, 224, 224, 3))
        logger.info("Testing detection model prediction...")
        _ = detection_model.predict(test_input)
        logger.info("✓ Detection model prediction successful")
        
        # Test classification model
        logger.info("Testing classification model prediction...")
        test_image = Image.new('RGB', (224, 224), color='white')
        result_message, top_tribe, confidence = classify_mask(test_image)
        logger.info(f"Classification result: {result_message}")
        logger.info("✓ Classification model prediction successful")
        
        return True
    except Exception as e:
        logger.error(f"Prediction test failed: {str(e)}")
        return False

def test_model_attributes():
    """Test model attributes and shapes"""
    try:
        logger.info("Testing model attributes...")
        
        # Check detection model
        logger.info(f"Detection model input shape: {detection_model.input_shape}")
        logger.info(f"Detection model output shape: {detection_model.output_shape}")
        
        # Check classification model
        logger.info(f"Classification model input shape: {classification_model.input_shape}")
        logger.info(f"Classification model output shape: {classification_model.output_shape}")
        
        # Verify classification model output matches tribe count
        if classification_model.output_shape[-1] != len(TRIBE_GROUPS):
            raise ValueError("Mismatch between model outputs and tribe count")
        
        logger.info("✓ Model attributes test successful")
        return True
    except Exception as e:
        logger.error(f"Model attributes test failed: {str(e)}")
        return False

def test_with_timeout(func, timeout=300):
    """Run a test with timeout"""
    import signal
    
    def handler(signum, frame):
        raise TimeoutError(f"Test timed out after {timeout} seconds")
    
    signal.signal(signal.SIGALRM, signal.alarm(timeout))
    try:
        return func()
    finally:
        signal.alarm(0)

def run_all_tests():
    logger.info("=== Starting Comprehensive Deployment Tests ===")
    
    tests = [
        ("Model Loading Methods", test_model_loading_methods),
        ("Model Prediction", test_model_prediction),
        ("Model Attributes", test_model_attributes)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\n=== Running {test_name} Test ===")
        if test_with_timeout(test_func):
            logger.info(f"✓ {test_name} test passed")
        else:
            logger.error(f"✗ {test_name} test failed")
            all_passed = False
    
    if all_passed:
        logger.info("\n=== All tests passed successfully! ===")
    else:
        logger.error("\n=== Some tests failed! ===")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        raise SystemExit("Tests failed - DO NOT DEPLOY!") 