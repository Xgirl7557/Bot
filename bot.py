import os
import logging
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load GPT-Neo model
gpt_neo_generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

# Load Stable Diffusion model
stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# Commenting out the following line as PythonAnywhere might not have CUDA support
# stable_diffusion_pipe = stable_diffusion_pipe.to("cuda")

# GPT-Neo Text Generation
def generate_text(prompt: str) -> str:
    response = gpt_neo_generator(prompt, max_length=50)
    return response[0]['generated_text']

# Stable Diffusion Image Generation
def generate_image(prompt: str) -> str:
    image = stable_diffusion_pipe(prompt).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path

# Video Editing (concatenation of videos)
def edit_video(video_files: list, output_file="output.mp4") -> str:
    clips = [VideoFileClip(file) for file in video_files]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_file)
    return output_file

# Telegram bot handlers
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! Use /chat, /image, or /video to interact with the bot.')

def chat(update: Update, context: CallbackContext) -> None:
    prompt = ' '.join(context.args)
    response = generate_text(prompt)
    update.message.reply_text(response)

def image(update: Update, context: CallbackContext) -> None:
    prompt = ' '.join(context.args)
    image_path = generate_image(prompt)
    update.message.reply_photo(photo=open(image_path, 'rb'))

def video(update: Update, context: CallbackContext) -> None:
    if not context.args:
        update.message.reply_text("Please send the video files followed by the command.")
        return

    video_files = context.args
    output_file = edit_video(video_files)
    update.message.reply_video(video=open(output_file, 'rb'))

def video_handler(update: Update, context: CallbackContext) -> None:
    video_file = update.message.video.get_file()
    video_path = f"input_{video_file.file_id}.mp4"
    video_file.download(video_path)
    
    context.user_data['video_files'] = context.user_data.get('video_files', [])
    context.user_data['video_files'].append(video_path)

    update.message.reply_text("Video received. You can send more videos or use /video to process them.")

def main() -> None:
    # Set the Telegram bot token
    telegram_bot_token = '7411473634:AAGYbfeywKPq4YIQBcuHsNf0f_-u9NEEvzI'

    updater = Updater(telegram_bot_token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Register handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("chat", chat))
    dispatcher.add_handler(CommandHandler("image", image))
    dispatcher.add_handler(CommandHandler("video", video))
    dispatcher.add_handler(MessageHandler(Filters.video, video_handler))

    # Start the Bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
