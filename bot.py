import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import pipeline
import torch
import torchvision.transforms as transforms
from PIL import Image

# Telegram bot token (replace with your actual token)
TOKEN = "7411473634:AAGYbfeywKPq4YIQBcuHsNf0f_-u9NEEvzI"

# Logging setup (optional but recommended)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the text generation pipeline
text_generator = pipeline("text-generation")

# Initialize the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = torch.hub.load('openai/clip', 'ViT-B/32', pretrained=True)
clip_model.to(device)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Start command handler
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! I am your AI bot. Send me a message to get started.')

# Handle incoming messages
def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    
    # Example: Text generation using Hugging Face Transformers
    if user_input.startswith('/generate'):
        prompt = user_input[len('/generate'):].strip()
        response = text_generator(prompt, max_length=50, num_return_sequences=1)
        bot_response = response[0]['generated_text'].strip()
    
    # Example: Text to image generation using OpenAI's CLIP (not fully implemented)
    elif user_input.startswith('/image'):
        prompt = user_input[len('/image'):].strip()
        image_url = "https://example.com/image.jpg"  # Replace with an actual image URL
        image = Image.open(requests.get(image_url, stream=True).raw)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text = clip_model.encode_text([prompt]).to(device)
        image_features = clip_model.encode_image(image_tensor).to(device)
        logits_per_image, logits_per_text = clip_model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()[0]
        bot_response = f"Image classification result: {probs}"
    
    else:
        bot_response = "I'm sorry, I didn't understand that command."
    
    update.message.reply_text(bot_response)

def main() -> None:
    # Initialize the Telegram Bot
    updater = Updater(TOKEN, use_context=True)
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher
    
    # Register handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    
    # Start the Bot
    updater.start_polling()
    logger.info("Bot started. Press Ctrl+C to stop.")
    
    # Run the bot until you press Ctrl+C
    updater.idle()

if __name__ == '__main__':
    main()
