import os
import logging
import numpy as np
import gdown
from PIL import Image
import tensorflow as tf
from layers import Cast  # Add this import
import tempfile  # Add this import
import json
import h5py
from detection import get_cached_model_path, download_model

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

# Google Drive file ID for your model
MODEL_ID = "1foeIsWYYmvr2UAoVyFSRhKPagsAmTLrs"

# Add at top of file, after imports
tf.keras.backend.set_floatx('float32')
tf.keras.mixed_precision.set_global_policy('float32')

def download_model():
    """Download the model file from Google Drive to a temporary location."""
    logger.info("Downloading model from Google Drive...")
    temp_dir = tempfile.gettempdir()
    temp_model_path = os.path.join(temp_dir, "temp_model.h5")
    
    url = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
    gdown.download(url, temp_model_path, quiet=False)
    return temp_model_path

# Define the tribe mapping based on the class distribution graph
TRIBE_GROUPS = {
    0: "Benin/Yoruba", 1: "Burkina_Faso/Mossi", 2: "Cote d'Ivoire/Baoule",
    3: "Cote d'Ivoire/Bete", 4: "Cote d'Ivoire/Dan_Yacouba", 5: "Cote d'Ivoire/Djimini",
    6: "Cote d'Ivoire/Gouro", 7: "Cote d'Ivoire/Guro", 8: "Cote d'Ivoire/Kran",
    9: "Cote d'Ivoire/Senoufo", 10: "Cote d'Ivoire/Yohoure", 11: "Gabon/Punu",
    12: "Liberia/Grebo", 13: "Liberia/Guere", 14: "Mali/Bambara",
    15: "Mali/Dogon", 16: "Nigeria/Eket", 17: "Nigeria/Idoma",
    18: "Nigeria/Igbo", 19: "Nigeria/Yoruba", 20: "RDC/Chokwe",
    21: "RDC/Hemba", 22: "RDC/Kuba", 23: "RDC/Lega",
    24: "RDC/Luba", 25: "RDC/Mbala", 26: "RDC/Pende_Bapende",
    27: "RDC/Salampasu", 28: "RDC/Songye", 29: "Tanzanie/Makonde"
}

try:
    logger.info("Loading classification model...")
    model_path = get_cached_model_path() or download_model()
    
    try:
        # First attempt - direct loading with minimal custom objects
        classification_model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'dtype': tf.float32}
        )
        logger.info("Successfully loaded model directly")
    except Exception as e:
        logger.info(f"Direct loading failed: {str(e)}")
        logger.info("Attempting alternative loading method...")
        
        # Second attempt - manual construction with explicit input
        inputs = tf.keras.Input(shape=(224, 224, 3))
        base_model = tf.keras.applications.ResNet50(
            input_tensor=inputs,
            include_top=False,
            weights=None
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(len(TRIBE_GROUPS), activation='softmax')(x)
        
        classification_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Load weights with skip_mismatch
        classification_model.load_weights(
            model_path, 
            by_name=True, 
            skip_mismatch=True
        )
        logger.info("Successfully loaded model with manual construction")
    
    # Test model
    logger.info("Testing classification model...")
    dummy_input = np.zeros((1, 224, 224, 3))
    predictions = classification_model.predict(dummy_input)
    assert predictions.shape[-1] == len(TRIBE_GROUPS), "Model output shape mismatch"
    logger.info("Model test successful")
    
    # Clean up
    os.remove(model_path)
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def preprocess_image_for_classification(image):
    target_size = classification_model.input_shape[1:3]  # Get input size from model
    logger.info(f"Resizing image to {target_size}")
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert('RGB')
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    logger.info(f"Preprocessed image shape: {image_array.shape}")
    return image_array

def classify_mask(image):
    try:
        image_array = preprocess_image_for_classification(image)
        logger.info(f"Running prediction on image...")
        predictions = classification_model.predict(image_array)[0]
        logger.info(f"Raw predictions shape: {predictions.shape}")
        logger.info(f"Raw predictions: {predictions}")
        
        if len(predictions) != len(TRIBE_GROUPS):
            raise ValueError(f"Mismatch between number of predictions ({len(predictions)}) and number of tribes ({len(TRIBE_GROUPS)})")
        
        # Get top 3 predictions
        top_3_indices = predictions.argsort()[-3:][::-1]
        top_3_probabilities = predictions[top_3_indices]
        top_3_tribes = [TRIBE_GROUPS[i].split('/')[-1] for i in top_3_indices]  # Get only the tribe name
        
        # Format the result message
        result_message = f"Il y a {top_3_probabilities[0]*100:.2f}% de chance que votre masque appartienne au peuple {top_3_tribes[0]}.\n"
        result_message += f"Autres possibilités: {top_3_tribes[1]} et {top_3_tribes[2]}"

        logger.info(f"Classification result: {result_message}")
        return result_message, top_3_tribes[0], top_3_probabilities[0] * 100
    except Exception as e:
        logger.error(f"Error in classify_mask: {str(e)}")
        raise Exception(f"Error during mask classification: {e}")

# Test function to ensure everything is working
def test_classification():
    try:
        # Load a test image
        test_image_path = r"C:\path\to\test\image.jpg"  # Replace with an actual test image path
        test_image = Image.open(test_image_path)
        
        # Run classification
        result, top_tribe, confidence = classify_mask(test_image)
        
        print("Test Classification Result:")
        print(result)
        print(f"Top tribe: {top_tribe}")
        print(f"Confidence: {confidence}%")
    except Exception as e:
        print(f"Test classification failed: {e}")

if __name__ == "__main__":
    test_classification()
