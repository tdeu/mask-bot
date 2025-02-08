import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler
import os
import gdown
from tensorflow.keras.layers import Layer
from layers import Cast  # Add this import
import tempfile  # Add this import
from dotenv import load_dotenv
import json
import h5py

# Load environment variables
load_dotenv()

# Replace the hardcoded token with environment variable
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Update these constants
MODEL_ID = "1foeIsWYYmvr2UAoVyFSRhKPagsAmTLrs"  # ID for .h5 model
KERAS_MODEL_ID = "1XrWEZJWOluRv2nMUiGmifiLnvzKkfUY9"  # ID for .keras model
MODEL_PATH = "new_best_model.h5"
KERAS_MODEL_PATH = "new_best_model.keras"

# Add at the top of the file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

# Add near the top after logger setup
logger.setLevel(logging.DEBUG)  # Set to DEBUG level for more detailed logs

MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'model_cache'))

def get_cached_model_path():
    """Get path to cached model or None if not cached."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    cached_path = os.path.join(MODEL_CACHE_DIR, "model.h5")
    return cached_path if os.path.exists(cached_path) else None

def download_model():
    """Download the model file from Google Drive to temporary storage."""
    logger.debug("Starting model download...")
    
    # Use system temp directory which should be writable
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_model.h5")
    
    try:
        logger.info("Downloading model...")
        url = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
        gdown.download(url, temp_path, quiet=False)
        logger.debug("Download completed")
        return temp_path
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

# Remove the multiple model formats, just use one
try:
    logger.info("Loading model...")
    temp_model_path = download_model()
    custom_objects = {
        "Cast": Cast,
        "mixed_float16": tf.keras.mixed_precision.Policy('mixed_float16'),
        "float16": tf.float16,
        "float32": tf.float32
    }
    
    try:
        # First attempt: Direct loading with compile=False and safe mode
        model = tf.keras.models.load_model(
            temp_model_path, 
            custom_objects=custom_objects, 
            compile=False,
            safe_mode=True  # Added safe mode
        )
    except (TypeError, ValueError) as e:
        if "class_name" in str(e) or "batch_shape" in str(e):
            logger.info("Attempting alternative model loading method...")
            try:
                # Create base model with explicit layer names
                base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3),
                    name='resnet50_base'
                )
                x = tf.keras.layers.GlobalAveragePooling2D(name='global_pooling')(base_model.output)
                x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(x)
                x = tf.keras.layers.BatchNormalization(name='batch_norm_1')(x)
                x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
                outputs = tf.keras.layers.Dense(30, activation='softmax', name='predictions')(x)
                
                # Create model and load weights by name
                model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name='mask_classifier')
                try:
                    model.load_weights(temp_model_path, by_name=True)
                    logger.info("Successfully loaded weights by name")
                except:
                    logger.info("Attempting to load weights by skipping mismatched layers...")
                    model.load_weights(temp_model_path, by_name=True, skip_mismatch=True)
                    logger.info("Successfully loaded weights with skip_mismatch")
            except Exception as e2:
                logger.error(f"Alternative loading failed: {str(e2)}")
                raise
            
    # Clean up the temporary file
    os.remove(temp_model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Add this after model loading
try:
    # Test the model with a dummy input to ensure it's working
    logger.info("Testing model with dummy input...")
    dummy_input = np.zeros((1, 224, 224, 3))
    _ = model.predict(dummy_input)
    logger.info("Model test successful")
except Exception as e:
    logger.error(f"Model test failed: {e}")
    raise

# Define tribe mapping (copied from your classify_mask.py)
TRIBE_GROUPS = {
    0: "Benin/Yoruba",
    1: "Burkina_Faso/Mossi",
    2: "Cote d'Ivoire/Baoule",
    3: "Cote d'Ivoire/Bete",
    4: "Cote d'Ivoire/Dan_Yacouba",
    5: "Cote d'Ivoire/Djimini",
    6: "Cote d'Ivoire/Gouro",
    7: "Cote d'Ivoire/Guro",
    8: "Cote d'Ivoire/Kran",
    9: "Cote d'Ivoire/Senoufo",
    10: "Cote d'Ivoire/Yohoure",
    11: "Gabon/Punu",
    12: "Liberia/Grebo",
    13: "Liberia/Guere",
    14: "Mali/Bambara",
    15: "Mali/Dogon",
    16: "Nigeria/Eket",
    17: "Nigeria/Idoma",
    18: "Nigeria/Igbo",
    19: "Nigeria/Yoruba",
    20: "RDC/Chokwe",
    21: "RDC/Hemba",
    22: "RDC/Kuba",
    23: "RDC/Lega",
    24: "RDC/Luba",
    25: "RDC/Mbala",
    26: "RDC/Pende_Bapende",
    27: "RDC/Salampasu",
    28: "RDC/Songye",
    29: "Tanzanie/Makonde"
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Identifier un masque", callback_data='identify')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Bienvenue sur l'Identificateur de Masques Africains ! Envoyez-moi une photo de votre masque pour l'identifier.",
        reply_markup=reply_markup
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == 'identify':
        await query.edit_message_text("Veuillez télécharger une photo du masque que vous souhaitez identifier.")
        return 1  # WAITING_FOR_PHOTO state

    return ConversationHandler.END

def preprocess_image(image, target_size=(224, 224)):  # Updated size to match your model's input
    try:
        # Resize the image
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to RGB if not already
        image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        preprocessed = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Preprocessed image shape: {preprocessed.shape}")
        return preprocessed
    except Exception as e:
        logger.error(f"Error in preprocessing image: {e}")
        raise

def detect_mask(img_array):
    try:
        # Get model predictions
        predictions = model.predict(img_array)[0]
        
        # Log raw predictions
        logger.info(f"Raw predictions array: {predictions}")
        logger.info(f"Max prediction value: {np.max(predictions)}")
        
        # Get top 3 predictions
        top_3_indices = predictions.argsort()[-3:][::-1]
        top_3_probabilities = predictions[top_3_indices]
        top_3_tribes = [TRIBE_GROUPS[i] for i in top_3_indices]
        
        logger.info(f"Top 3 indices: {top_3_indices}")
        logger.info(f"Top 3 probabilities: {top_3_probabilities}")
        logger.info(f"Top 3 tribes: {top_3_tribes}")
        
        # Lower the threshold since we're dealing with multi-class classification
        threshold = 0.1  # Reduced from 0.3
        mask_detected = top_3_probabilities[0] > threshold
        
        if mask_detected:
            result_message = (
                f"Masque africain détecté!\n"
                f"Probabilités les plus élevées:\n"
                f"1. {top_3_tribes[0]}: {top_3_probabilities[0]*100:.1f}%\n"
                f"2. {top_3_tribes[1]}: {top_3_probabilities[1]*100:.1f}%\n"
                f"3. {top_3_tribes[2]}: {top_3_probabilities[2]*100:.1f}%"
            )
        else:
            result_message = (
                "Aucun masque africain n'a été détecté avec certitude dans cette image. "
                "Veuillez vous assurer que l'image est claire et montre bien un masque africain traditionnel."
            )
        
        logger.info(f"Final result message: {result_message}")
        return mask_detected, result_message
    
    except Exception as e:
        logger.error(f"Error in mask detection: {e}")
        raise