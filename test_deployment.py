import os
import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import h5py
import json
from detection import download_model, Cast
from classify_mask import classify_mask, TRIBE_GROUPS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_model_architecture():
    """Test if we can load and inspect the model architecture"""
    try:
        temp_model_path = download_model()
        logger.info("Checking model architecture...")
        
        with h5py.File(temp_model_path, 'r') as f:
            if 'model_config' in f.attrs:
                config = json.loads(f.attrs['model_config'].decode('utf-8'))
                logger.info(f"Found model config: {json.dumps(config, indent=2)}")
                logger.info(f"Layer count in config: {len(config['config']['layers'])}")
            else:
                logger.error("No model_config found in h5 file")
            
            # Check model weights
            if 'model_weights' in f:
                logger.info("Found model weights")
                weight_names = []
                f['model_weights'].visit(lambda name: weight_names.append(name))
                logger.info(f"Found {len(weight_names)} weight layers")
            else:
                logger.error("No model_weights found in h5 file")
        
        return True
    except Exception as e:
        logger.error(f"Architecture test failed: {e}")
        return False

def test_end_to_end():
    """Test the entire pipeline"""
    try:
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Test classification
        result_message, top_tribe, confidence = classify_mask(test_image)
        logger.info(f"Classification result: {result_message}")
        logger.info(f"Top tribe: {top_tribe}")
        logger.info(f"Confidence: {confidence}")
        
        return True
    except Exception as e:
        logger.error(f"End-to-end test failed: {e}")
        return False

def run_all_tests():
    logger.info("Starting deployment tests...")
    
    # Test 1: Model Architecture
    logger.info("\n=== Testing Model Architecture ===")
    if test_model_architecture():
        logger.info("✓ Model architecture test passed")
    else:
        logger.error("✗ Model architecture test failed")
    
    # Test 2: End-to-end Pipeline
    logger.info("\n=== Testing End-to-end Pipeline ===")
    if test_end_to_end():
        logger.info("✓ End-to-end test passed")
    else:
        logger.error("✗ End-to-end test failed")

if __name__ == "__main__":
    run_all_tests() 