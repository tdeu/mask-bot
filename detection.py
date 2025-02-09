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
from config import TRIBE_GROUPS

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

# Force CPU usage to avoid GPU-related errors on Railway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

# Add near the top after logger setup
logger.setLevel(logging.DEBUG)  # Set to DEBUG level for more detailed logs

MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'model_cache'))

# Add at top of file, after imports
tf.keras.backend.set_floatx('float32')
tf.keras.mixed_precision.set_global_policy('float32')

def get_cached_model_path():
    """Get path to cached model or None if not cached."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    cached_path = os.path.join(MODEL_CACHE_DIR, "model.h5")
    return cached_path if os.path.exists(cached_path) else None

def download_model():
    """Download the model file from Google Drive to temporary storage."""
    logger.debug("Starting model download...")
    
    # Use system temp directory which should be writable on Railway
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

# Replace the model loading section with:
try:
    logger.info("Loading model...")
    temp_model_path = download_model()
    
    try:
        # First attempt - direct loading with Cast layer
        model = tf.keras.models.load_model(
            temp_model_path,
            custom_objects={'Cast': Cast, 'dtype': tf.float32},
            compile=False
        )
        logger.info("Successfully loaded model directly")
    except Exception as e:
        logger.info(f"Direct loading failed: {str(e)}")
        logger.info("Attempting alternative loading method...")
        
        # Second attempt - manual construction with Cast layer
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = Cast(dtype=tf.float32)(inputs)
        base_model = tf.keras.applications.ResNet50(
            input_tensor=x,
            include_top=False,
            weights=None
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(len(TRIBE_GROUPS), activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.load_weights(temp_model_path, by_name=True, skip_mismatch=True)
        logger.info("Successfully loaded model with manual construction")
        
    # Test model
    logger.info("Testing model...")
    dummy_input = np.zeros((1, 224, 224, 3))
    predictions = model.predict(dummy_input)
    assert predictions.shape[-1] == len(TRIBE_GROUPS), "Model output shape mismatch"
    logger.info("Model test successful")
    
    # Clean up
    os.remove(temp_model_path)
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

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