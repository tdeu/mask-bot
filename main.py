import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from detection import detect_mask, preprocess_image, start, button, BOT_TOKEN
from classify_mask import classify_mask
from PIL import Image
import io

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

# Define states
WAITING_FOR_PHOTO = 1

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Traitement de votre image...")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        img = Image.open(io.BytesIO(photo_bytes))

        logger.info(f"Original Image format: {img.format}")
        logger.info(f"Original Image size: {img.size}")
        logger.info(f"Original Image mode: {img.mode}")

        # Process the whole image
        logger.info("Processing entire image for mask detection.")
        preprocessed = preprocess_image(img)
        logger.info(f"Preprocessed image shape: {preprocessed.shape}")
        
        mask_detected, result_message = detect_mask(preprocessed)
        
        logger.info(f"Whole image - Mask detection result: {mask_detected}")
        logger.info(f"Result message: {result_message}")

        await update.message.reply_text(result_message)

        if mask_detected:
            # Mask classification
            classification_result, top_tribe, class_confidence = classify_mask(img)
            await update.message.reply_text("Classification du masque...")
            await update.message.reply_text(classification_result)

    except Exception as e:
        error_message = f"Erreur lors du traitement de l'image : {e}"
        logger.error(error_message)
        await update.message.reply_text(error_message)

    return ConversationHandler.END

def main() -> None:
    logger.info("Démarrage du bot")
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    
    mask_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(button, pattern='^identify$')],
        states={
            WAITING_FOR_PHOTO: [MessageHandler(filters.PHOTO, handle_photo)],
        },
        fallbacks=[],
    )
    
    application.add_handler(mask_conv_handler)
    application.add_handler(CallbackQueryHandler(button))

    logger.info("Début de l'écoute des mises à jour")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()