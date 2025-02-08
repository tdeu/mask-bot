import os
import logging
import numpy as np
import gdown
from PIL import Image
import tensorflow as tf
from layers import Cast  # Add this import
import tempfile  # Add this import

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

# Google Drive file ID for your model
MODEL_ID = "1foeIsWYYmvr2UAoVyFSRhKPagsAmTLrs"

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
    temp_model_path = download_model()
    custom_objects = {
        "Cast": Cast,
        "mixed_float16": tf.keras.mixed_precision.Policy('mixed_float16'),
        "float16": tf.float16,
        "float32": tf.float32
    }
    
    # Add error handling and input layer workaround
    try:
        classification_model = tf.keras.models.load_model(temp_model_path, custom_objects=custom_objects)
    except TypeError as e:
        if "batch_shape" in str(e):
            logger.info("Attempting alternative model loading method...")
            try:
                # Try loading with experimental IO
                classification_model = tf.keras.models.load_model(
                    temp_model_path, 
                    custom_objects=custom_objects,
                    compile=False,
                    options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                )
            except:
                # If that fails, try reconstructing the model
                logger.info("Attempting model reconstruction...")
                base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3)
                )
                x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                outputs = tf.keras.layers.Dense(30, activation='softmax')(x)
                classification_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
                
                # Load weights only
                classification_model.load_weights(temp_model_path)
            
    # Clean up the temporary file
    os.remove(temp_model_path)
    logger.info("Classification model loaded successfully")
    logger.info(f"Model input shape: {classification_model.input_shape}")
    logger.info(f"Model output shape: {classification_model.output_shape}")
    
    # Determine the number of classes from the model's output shape
    num_classes = classification_model.output_shape[-1]
    logger.info(f"Number of classes detected from model output: {num_classes}")
    
    if num_classes != len(TRIBE_GROUPS):
        raise ValueError(f"Mismatch between number of classes in model ({num_classes}) and defined tribe mapping ({len(TRIBE_GROUPS)})")
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Tribe mapping loaded successfully")
    
    # Test the model with a dummy input
    logger.info("Testing classification model with dummy input...")
    dummy_input = np.zeros((1, 224, 224, 3))
    _ = classification_model.predict(dummy_input)
    logger.info("Classification model test successful")
        
except Exception as e:
    logger.error(f"Error during model loading or tribe mapping: {e}")
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
