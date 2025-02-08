import logging
import pandas as pd
import random
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler, CallbackQueryHandler, CommandHandler

logger = logging.getLogger(__name__)

# Define conversation states
QUESTIONS = range(1)

# Load your datasets
df = pd.read_csv(r'C:\Users\thoma\Bureau\telegram-image-internal\data\Mask DB2.csv')
image_df = pd.read_csv(r'C:\Users\thoma\Bureau\telegram-image-internal\data\Mask DB.csv')

# Define the order of questions
question_order = [
    "Face_shape", "Eye_shape", "Nose", "Mouth_shape", "Scarification_style",
    "Decorative_elements", "Color_palette", "Symbolic_features", "Overall_size",
    "Material", "Surface_texture", "Additional_ornamentation"
]

# Define user-friendly questions
question_texts = {
    "Face_shape": "What is the face shape of the mask?",
    "Eye_shape": "What is the eye shape of the mask?",
    "Nose": "How would you describe the nose of the mask?",
    "Mouth_shape": "What is the mouth shape of the mask?",
    "Scarification_style": "What is the style of scarification or tattoos on the mask?",
    "Decorative_elements": "What decorative elements are present on the mask?",
    "Color_palette": "What is the color palette of the mask?",
    "Symbolic_features": "Are there any symbolic features or objects on the mask?",
    "Overall_size": "What is the overall size of the mask?",
    "Material": "What material is the mask made of?",
    "Surface_texture": "How would you describe the surface texture of the mask?",
    "Additional_ornamentation": "Is there any additional ornamentation on the mask?"
}

def get_random_image_url(tribe):
    tribe_images = image_df[image_df['Tribe'] == tribe]
    if not tribe_images.empty:
        return random.choice(tribe_images.iloc[:, 2].tolist())  # Assuming image URL is in the third column
    return None

async def start_identification(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.debug("Starting mask identification process")
    context.user_data['answers'] = {}
    context.user_data['current_question'] = 0
    context.user_data['matching_masks'] = df
    return await ask_question(update, context)

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.debug("Asking a question")
    
    if context.user_data['matching_masks'].empty or len(context.user_data['matching_masks']) == 1 or context.user_data['current_question'] >= len(question_order):
        return await provide_result(update, context)
    
    current_feature = question_order[context.user_data['current_question']]
    valid_options = context.user_data['matching_masks'][current_feature].unique()
    
    if len(valid_options) == 1:
        context.user_data['answers'][current_feature] = valid_options[0]
        context.user_data['current_question'] += 1
        return await ask_question(update, context)
    
    keyboard = [[InlineKeyboardButton(str(option), callback_data=f"answer_{option}")] for option in valid_options]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = update.callback_query.message if update.callback_query else update.message
    await message.reply_text(question_texts[current_feature], reply_markup=reply_markup)
    return QUESTIONS

async def handle_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.debug("Handling answer")
    query = update.callback_query
    await query.answer()
    
    if context.user_data['current_question'] >= len(question_order):
        return await provide_result(update, context)
    
    current_feature = question_order[context.user_data['current_question']]
    answer = query.data.split('_', 1)[1]
    
    context.user_data['answers'][current_feature] = answer
    context.user_data['matching_masks'] = context.user_data['matching_masks'][context.user_data['matching_masks'][current_feature] == answer]
    context.user_data['current_question'] += 1
    
    return await ask_question(update, context)

async def provide_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.debug("Providing result")
    matching_tribes = context.user_data['matching_masks']['Tribe'].value_counts()
    
    if len(matching_tribes) > 0:
        total_matches = matching_tribes.sum()
        result_message = "Based on your answers, the mask could be from the following tribe(s):\n\n"
        
        for tribe, count in matching_tribes.items():
            probability = (count / total_matches) * 100
            result_message += f"- {tribe}: {probability:.2f}%\n"
            
            # Fetch and send a random image for each tribe
            image_url = get_random_image_url(tribe)
            if image_url:
                try:
                    await update.callback_query.message.reply_photo(photo=image_url, caption=f"A sample mask from the {tribe} tribe")
                except Exception as e:
                    logger.error(f"Error sending image for {tribe}: {e}")
                    result_message += f"(Unable to load sample image for {tribe})\n"
            else:
                result_message += f"(No sample image available for {tribe})\n"
        
        if len(matching_tribes) == 1:
            result_message = f"There is a 100% chance your mask is from the {matching_tribes.index[0]} tribe."
    else:
        result_message = "Sorry, I couldn't identify a matching tribe based on your answers. The mask might have unique features not in our database."
    
    await update.callback_query.message.reply_text(result_message)
    return ConversationHandler.END

def get_mask_identification_handler():
    logger.debug("Creating mask identification conversation handler")
    return ConversationHandler(
        entry_points=[CallbackQueryHandler(start_identification, pattern='^identify$')],
        states={
            QUESTIONS: [CallbackQueryHandler(handle_answer, pattern='^answer_')]
        },
        fallbacks=[CommandHandler('cancel', lambda u, c: ConversationHandler.END)]
    )