import logging
import tensorflow as tf
from detection import download_model, Cast
import h5py
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_model_loading():
    try:
        logger.info("Starting model loading test...")
        temp_model_path = download_model()
        
        custom_objects = {
            "Cast": Cast,
            "mixed_float16": tf.keras.mixed_precision.Policy('mixed_float16'),
            "float16": tf.float16,
            "float32": tf.float32
        }
        
        # Try all loading methods
        try:
            logger.info("Method 1: Direct loading...")
            model = tf.keras.models.load_model(temp_model_path, custom_objects=custom_objects)
            logger.info("Direct loading successful!")
            return True
        except Exception as e1:
            logger.warning(f"Direct loading failed: {e1}")
            
            try:
                logger.info("Method 2: Experimental IO loading...")
                model = tf.keras.models.load_model(
                    temp_model_path, 
                    custom_objects=custom_objects,
                    compile=False,
                    options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                )
                logger.info("Experimental IO loading successful!")
                return True
            except Exception as e2:
                logger.warning(f"Experimental IO loading failed: {e2}")
                
                try:
                    logger.info("Method 3: H5 direct loading...")
                    with h5py.File(temp_model_path, 'r') as f:
                        model_config = f.attrs.get('model_config')
                        if model_config is not None:
                            model_config = json.loads(model_config.decode('utf-8'))
                            model = tf.keras.models.model_from_config(
                                model_config,
                                custom_objects=custom_objects
                            )
                            model.load_weights(temp_model_path)
                            logger.info("H5 direct loading successful!")
                            return True
                        else:
                            raise ValueError("No model config found in H5 file")
                except Exception as e3:
                    logger.error(f"All loading methods failed. Last error: {e3}")
                    return False
                
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("Model loading test passed!")
    else:
        print("Model loading test failed!") 