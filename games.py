import random
import csv
import io
import requests
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Load CSV file
CSV_FILE_PATH = 'C:\\Users\\thoma\\Bureau\\telegram-image-internal\\data\\Mask DB.csv'

masks_data = []

try:
    with open(CSV_FILE_PATH, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        masks_data = list(csv_reader)
except Exception as e:
    print(f"Error reading CSV file: {str(e)}")
    exit(1)

# Create a dictionary of countries and their masks
country_masks = {}
for mask in masks_data:
    if mask['Country'] not in country_masks:
        country_masks[mask['Country']] = []
    country_masks[mask['Country']].append(mask)

# Define neighboring countries (simplified for this example)
neighboring_countries = {
    'Nigeria': ['Benin', 'Cameroon', 'Chad', 'Niger'],
    'Kenya': ['Tanzania', 'Uganda', 'Ethiopia', 'Somalia'],
    'Ghana': ['Ivory Coast', 'Burkina Faso', 'Togo'],
    # Add more countries and their neighbors as needed
}

def get_random_mask():
    return random.choice(masks_data)

def create_side_by_side_image(image_url1, image_url2):
    response1 = requests.get(image_url1)
    response2 = requests.get(image_url2)
    img1 = Image.open(io.BytesIO(response1.content))
    img2 = Image.open(io.BytesIO(response2.content))

    # Resize images to have the same height
    height = min(img1.size[1], img2.size[1])
    img1 = img1.resize((int(img1.size[0] * height / img1.size[1]), height))
    img2 = img2.resize((int(img2.size[0] * height / img2.size[1]), height))

    # Create a new image with the width of both images and the height of the tallest image
    total_width = img1.size[0] + img2.size[0]
    max_height = max(img1.size[1], img2.size[1])
    combined_img = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.size[0], 0))

    # Save the combined image to a bytes buffer
    img_byte_arr = io.BytesIO()
    combined_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return img_byte_arr

async def random_mask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mask = get_random_mask()
    image_url = mask.get('ImageURL') or mask.get('Image URL') or mask.get('URL') or mask.get('Image')
    caption = f"This is a mask from the {mask['Tribe']} tribe of {mask['Country']}."
    
    keyboard = [[InlineKeyboardButton("Get Another Random Mask", callback_data='random')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.message.reply_photo(photo=image_url, caption=caption, reply_markup=reply_markup)

async def play_games(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Devine", callback_data='game1')],
        [InlineKeyboardButton("Guess Which", callback_data='game2')],
        [InlineKeyboardButton("Mask Journey", callback_data='game3')],
        [InlineKeyboardButton("Tribal Tales", callback_data='game4')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text("Choose a game to play:", reply_markup=reply_markup)

async def play_devine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mask = get_random_mask()
    context.user_data['current_mask'] = mask
    image_url = mask.get('ImageURL') or mask.get('Image URL') or mask.get('URL') or mask.get('Image')
    
    # Send the image
    await update.callback_query.message.reply_photo(photo=image_url)
    
    # Prepare country options
    all_countries = list(set(m['Country'] for m in masks_data))
    wrong_countries = random.sample([c for c in all_countries if c != mask['Country']], 3)
    countries = wrong_countries + [mask['Country']]
    random.shuffle(countries)
    
    keyboard = [[InlineKeyboardButton(country, callback_data=f"country:{country}")] for country in countries]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Ask the country question
    await update.callback_query.message.reply_text(
        "Do you know where this mask is from?",
        reply_markup=reply_markup
    )

async def check_country(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chosen_country = query.data.split(':')[1]
    correct_country = context.user_data['current_mask']['Country']
    
    if chosen_country == correct_country:
        await query.answer("Well Done!")
        await guess_tribe(update, context)
    else:
        await query.edit_message_text(
            f"Too bad, this mask is actually from {correct_country}. Wanna try another one?",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Try Another", callback_data="play_devine")]])
        )

async def guess_tribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mask = context.user_data['current_mask']
    
    all_tribes = list(set(m['Tribe'] for m in masks_data if m['Country'] == mask['Country']))
    wrong_tribes = random.sample([t for t in all_tribes if t != mask['Tribe']], 3)
    tribes = wrong_tribes + [mask['Tribe']]
    random.shuffle(tribes)
    
    keyboard = [[InlineKeyboardButton(tribe, callback_data=f"tribe:{tribe}")] for tribe in tribes]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.edit_message_text(
        "Correct! What ethnic group does it come from?",
        reply_markup=reply_markup
    )

async def check_tribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chosen_tribe = query.data.split(':')[1]
    correct_tribe = context.user_data['current_mask']['Tribe']
    correct_country = context.user_data['current_mask']['Country']
    
    if chosen_tribe == correct_tribe:
        await query.edit_message_text(
            f"You are a pro. This mask is indeed from {correct_country} and belongs to the {correct_tribe} ethnic group.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Play Again", callback_data="play_devine")]])
        )
    else:
        await query.edit_message_text(
            f"Too bad. This mask is from {correct_country} and belongs to the {correct_tribe} ethnic group. Wanna try another one?",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Try Another", callback_data="play_devine")]])
        )

async def play_guess_which(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['guess_which_score'] = 0
    context.user_data['guess_which_round'] = 0
    await start_guess_which_round(update, context)

async def start_guess_which_round(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['guess_which_round'] += 1
    
    # Get two random masks from different ethnic groups
    mask1 = get_random_mask()
    mask2 = get_random_mask()
    while mask1['Tribe'] == mask2['Tribe']:
        mask2 = get_random_mask()
    
    context.user_data['current_masks'] = [mask1, mask2]
    correct_mask = random.choice([mask1, mask2])
    context.user_data['correct_mask'] = correct_mask
    
    image_url1 = mask1.get('ImageURL') or mask1.get('Image URL') or mask1.get('URL') or mask1.get('Image')
    image_url2 = mask2.get('ImageURL') or mask2.get('Image URL') or mask2.get('URL') or mask2.get('Image')
    
    # Create side-by-side image
    combined_image = create_side_by_side_image(image_url1, image_url2)
    
    caption = f"Which one belongs to the {correct_mask['Tribe']} ethnic group?"
    
    keyboard = [
        [InlineKeyboardButton("Left Image", callback_data="guess_which:0"),
         InlineKeyboardButton("Right Image", callback_data="guess_which:1")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send combined image and question
    message = await update.callback_query.message.reply_photo(
        photo=combined_image,
        caption=caption,
        reply_markup=reply_markup
    )
    
    # Store the message id for later editing
    context.user_data['guess_which_message_id'] = message.message_id

async def check_guess_which(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chosen_index = int(query.data.split(':')[1])
    correct_mask = context.user_data['correct_mask']
    chosen_mask = context.user_data['current_masks'][chosen_index]
    
    if chosen_mask['Tribe'] == correct_mask['Tribe']:
        context.user_data['guess_which_score'] += 1
        feedback = "Correct!"
    else:
        feedback = f"Wrong. The correct answer was the {correct_mask['Tribe']} mask."
    
    current_round = context.user_data['guess_which_round']
    current_score = context.user_data['guess_which_score']
    
    if current_round < 5:
        await query.edit_message_caption(
            caption=f"{feedback} Your current score: {current_score}/{current_round}\n\nNext question:",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Continue", callback_data="next_guess_which")]])
        )
    else:
        final_score = context.user_data['guess_which_score']
        keyboard = [
            [InlineKeyboardButton("Play Again", callback_data="play_guess_which")],
            [InlineKeyboardButton("Play Another Game", callback_data="play")],
            [InlineKeyboardButton("Return to Main Menu", callback_data="start")]
        ]
        await query.edit_message_caption(
            caption=f"{feedback}\n\nGame Over! Your final score: {final_score}/5",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def start_mask_journey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['journey'] = {
        'score': 0,
        'countries_visited': [],
        'current_country': random.choice(list(country_masks.keys())),
        'steps': 0
    }
    await present_country_challenge(update, context)

async def present_country_challenge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    journey = context.user_data['journey']
    current_country = journey['current_country']
    mask = random.choice(country_masks[current_country])
    
    journey['current_mask'] = mask
    journey['countries_visited'].append(current_country)
    journey['steps'] += 1
    
    image_url = mask.get('ImageURL') or mask.get('Image URL') or mask.get('URL') or mask.get('Image')
    caption = f"You are now in {current_country}. Can you identify which ethnic group this mask belongs to?"
    
    # Generate options for ethnic groups
    all_tribes = list(set(m['Tribe'] for m in country_masks[current_country]))
    options = random.sample(all_tribes, min(4, len(all_tribes)))
    if mask['Tribe'] not in options:
        options[0] = mask['Tribe']
    random.shuffle(options)
    
    keyboard = [[InlineKeyboardButton(tribe, callback_data=f"journey_tribe:{tribe}")] for tribe in options]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.message.reply_photo(photo=image_url, caption=caption, reply_markup=reply_markup)

async def check_journey_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chosen_tribe = query.data.split(':')[1]
    journey = context.user_data['journey']
    correct_tribe = journey['current_mask']['Tribe']
    
    if chosen_tribe == correct_tribe:
        journey['score'] += 1
        feedback = "Correct! "
    else:
        feedback = f"Sorry, that's incorrect. The mask belongs to the {correct_tribe} tribe. "
    
    cultural_fact = f"Did you know? The {correct_tribe} tribe is known for their {journey['current_mask']['Country']} masks."
    
    if journey['steps'] >= 7:  # End the journey after 7 steps
        final_score = journey['score']
        countries_visited = len(set(journey['countries_visited']))
        await query.edit_message_caption(
            caption=f"{feedback}\n\n{cultural_fact}\n\nJourney complete! Your final score: {final_score}/7\nYou visited {countries_visited} different countries!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Play Again", callback_data="start_mask_journey")],
                [InlineKeyboardButton("Play Another Game", callback_data="play")],
                [InlineKeyboardButton("Return to Main Menu", callback_data="start")]
            ])
        )
    else:
        # Continue the journey
        next_countries = neighboring_countries.get(journey['current_country'], list(country_masks.keys()))
        next_countries = [c for c in next_countries if c not in journey['countries_visited']]
        if not next_countries:
            next_countries = [c for c in country_masks.keys() if c not in journey['countries_visited']]
        
        journey['current_country'] = random.choice(next_countries) if next_countries else random.choice(list(country_masks.keys()))
        
        await query.edit_message_caption(
            caption=f"{feedback}\n\n{cultural_fact}\n\nReady to continue your journey?",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Next Country", callback_data="next_country")]])
        )

async def play_tribal_tales(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Select a random mask
    correct_mask = get_random_mask()
    image_url = correct_mask.get('ImageURL') or correct_mask.get('Image URL') or correct_mask.get('URL') or correct_mask.get('Image')
    
    # Get two other random masks for wrong descriptions
    wrong_masks = random.sample([m for m in masks_data if m['Tribe'] != correct_mask['Tribe']], 2)
    
    # Create description options
    descriptions = [correct_mask['Description']] + [m['Description'] for m in wrong_masks]
    random.shuffle(descriptions)
    
    # Send the image
    await update.callback_query.message.reply_photo(photo=image_url)
    
    # Create keyboard with actual descriptions
    keyboard = [[InlineKeyboardButton(desc[:50] + "...", callback_data=f"tribal_tales:{i}")] for i, desc in enumerate(descriptions)]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Store correct answer and descriptions in context
    context.user_data['tribal_tales_correct'] = descriptions.index(correct_mask['Description'])
    context.user_data['tribal_tales_descriptions'] = descriptions
    
    # Ask the question
    await update.callback_query.message.reply_text(
        "Which description matches this mask's tribe?",
        reply_markup=reply_markup
    )

async def check_tribal_tales(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chosen_index = int(query.data.split(':')[1])
    correct_index = context.user_data['tribal_tales_correct']
    descriptions = context.user_data['tribal_tales_descriptions']
    
    if chosen_index == correct_index:
        feedback = f"Correct! The full description is:\n\n{descriptions[correct_index]}"
    else:
        feedback = f"Sorry, that's incorrect. The correct description was:\n\n{descriptions[correct_index]}"
    
    keyboard = [
        [InlineKeyboardButton("Play Again", callback_data="game4")],
        [InlineKeyboardButton("Play Another Game", callback_data="play")],
        [InlineKeyboardButton("Return to Main Menu", callback_data="start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(feedback, reply_markup=reply_markup)

async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Implement the learn functionality
    await update.callback_query.message.reply_text("Learn functionality coming soon!")

async def handle_game_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query.data == 'game1':
        await play_devine(update, context)
    elif query.data == 'game2':
        await play_guess_which(update, context)
    elif query.data == 'game3':
        await start_mask_journey(update, context)
    elif query.data == 'game4':
        await play_tribal_tales(update, context)

async def handle_guess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query.data.startswith('country:'):
        await check_country(update, context)
    elif query.data.startswith('tribe:'):
        await check_tribe(update, context)

# Export all necessary functions
__all__ = ['random_mask', 'play_games', 'learn', 'handle_game_callback', 'handle_guess_callback',
           'check_guess_which', 'check_journey_answer', 'check_tribal_tales']