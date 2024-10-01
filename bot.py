import os
import sqlite3
import json
import ssl
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler
from dotenv import load_dotenv
from clicktime_ai_bot import NodeInstallationBot
from telegram.constants import ChatAction

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Define states for conversation handler
WAITING_FOR_URL = range(1)

# Initialize database
conn = sqlite3.connect('bot_data.db')
cursor = conn.cursor()

# Create table for state storage
cursor.execute('''
    CREATE TABLE IF NOT EXISTS states (
        chat_id INTEGER PRIMARY KEY,
        state TEXT
    )
''')
conn.commit()

# Initialize a single instance of the model
bot_instance = NodeInstallationBot()

# Function to save state
def save_state(chat_id, state):
    state_str = json.dumps(state)  # Serialize state to JSON string
    cursor.execute('REPLACE INTO states (chat_id, state) VALUES (?, ?)', (chat_id, state_str))
    conn.commit()

# Function to load state
def load_state(chat_id):
    cursor.execute('SELECT state FROM states WHERE chat_id = ?', (chat_id,))
    result = cursor.fetchone()
    return json.loads(result[0]) if result else None

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat_id = update.message.chat_id
    
    keyboard = [
        [InlineKeyboardButton("Load New Data", callback_data='train')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! Welcome to NodeRunner bot.",
        reply_markup=reply_markup,
    )

# Train command handler
async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    # Create cancel button
    keyboard = [
        [InlineKeyboardButton("Cancel", callback_data='cancel')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text="Please provide the URL of the website you want to train me on.",
        reply_markup=reply_markup
    )
    return WAITING_FOR_URL

# Handle URL input for training
async def url_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    url = update.message.text
    chat_id = update.message.chat_id

    # Save the URL in the state storage
    save_state(chat_id, {'url': url})

    await update.message.reply_text("I'm learning, please wait a moment...")
    success = bot_instance.load_and_store_data(url)
    if success:
        await update.message.reply_text("Great! I'm now smarter than you 👀")
    else:
        await update.message.reply_text("Failed to load data. Please try again with a valid URL.")

    # Clear state
    cursor.execute('DELETE FROM states WHERE chat_id = ?', (chat_id,))
    conn.commit()
    return ConversationHandler.END

# Handle incoming messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    text = update.message.text if update.message.text else ""
    image_text = ""
    if text and text.startswith('!'):
        # Send "typing" action
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        # Check for photo
        if update.message.photo:
            photo_file = await update.message.photo[-1].get_file()
            photo_path = os.path.join("images", f"{chat_id}_image.jpg")
            await photo_file.download_to_drive(custom_path=photo_path)
            image_text = bot_instance.extract_text_from_image(photo_path)
            
            # Remove image file after processing
            try:
                os.remove(photo_path)
            except OSError as e:
                print(f"Error: {photo_path} : {e.strerror}")
        
        # Combine text from message and image
        combined_text = text + "\n" + image_text

        # Get response from the model
        response = bot_instance.ask_question(combined_text)
        formatted_response = format_response(response)
        await update.message.reply_text(formatted_response, parse_mode='MarkdownV2')

# Format response
def format_response(response: str) -> str:
    # Escape backslashes first
    response = response.replace('\\', '\\\\')

    # Escape reserved characters in MarkdownV2
    reserved_chars = r'_[]()~>#+-=|{}.!*'
    for char in reserved_chars:
        response = response.replace(char, f'\\{char}')
    
    # Return formatted response
    return response

# Cancel conversation handler
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        text="Training cancelled. Returning to the main menu."
    )
    await start(update, context)
    return ConversationHandler.END

def main() -> None:
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Conversation handler for training
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(train, pattern='train')],
        states={
            WAITING_FOR_URL: [MessageHandler(filters.TEXT & ~filters.COMMAND, url_received)],
        },
        fallbacks=[CallbackQueryHandler(cancel, pattern='cancel')],
    )

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, handle_message))

    # Run the bot
    application.run_polling()

if __name__ == '__main__':
    main()
