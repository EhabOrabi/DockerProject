import json

import requests
import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
from img_proc import Img
import boto3

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Bot:
    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        """Bot Main message handler"""
        # logger.info(f'Incoming message: {msg}')
        if "text" in msg:
            self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')
        else:
            # if there is checkbox caption
            if "caption" in msg:
                try:
                    img_path = self.download_user_photo(msg)
                    if msg["caption"] == "Blur":
                        # Send message to telegram bot
                        self.send_text(msg['chat']['id'], "Blur filter in progress")
                        new_img = Img(img_path)
                        new_img.blur()
                        new_path = new_img.save_img()
                        self.send_photo(msg["chat"]["id"], new_path)
                        self.send_text(msg['chat']['id'], "Blur filter applied")
                    elif msg["caption"] == "Contour":
                        self.send_text(msg['chat']['id'], "Contour filter in progress")
                        new_img = Img(img_path)
                        new_img.contour()
                        new_path = new_img.save_img()
                        self.send_photo(msg["chat"]["id"], new_path)
                        self.send_text(msg['chat']['id'], "Contour filter applied")
                    elif msg["caption"] == "Salt and pepper":  # concat, segment
                        self.send_text(msg['chat']['id'], "salt_n_pepper filter in progress")
                        new_img = Img(img_path)
                        new_img.salt_n_pepper()
                        new_path = new_img.save_img()
                        self.send_photo(msg["chat"]["id"], new_path)
                        self.send_text(msg['chat']['id'], "salt_n_pepper filter applied")
                    elif msg["caption"] == "mix":
                        self.send_text(msg['chat']['id'], "mix filter in progress")
                        new_img = Img(img_path)
                        new_img.salt_n_pepper()
                        new_path = new_img.save_img()

                        new_img2 = Img(new_path)
                        new_img2.blur()
                        new_path = new_img2.save_img()

                        self.send_photo(msg["chat"]["id"], new_path)
                        self.send_text(msg['chat']['id'], "mix filter applied")
                    elif msg["caption"] == "predict":
                        # TODO upload the photo to S3
                        # TODO send an HTTP request to the `yolo5` service for prediction
                        # TODO send the returned results to the Telegram end-user
                        self.send_text(msg['chat']['id'], "yolo5 activated ! ")

                        logger.info(f'Photo downloaded to: {img_path}')
                        photo_s3_name = img_path.split("/")

                        # Get the bucket name from the environment variable
                        images_bucket = os.environ['BUCKET_NAME']
                        client = boto3.client('s3')
                        client.upload_file(img_path, images_bucket, photo_s3_name[1])
                        # Upload the image to S3
                        # Send an HTTP request to the YOLO5 service for prediction
                        yolo5_url = "http://yolo5_app:8081/predict"
                        headers = {'Content-Type': 'application/json'}
                        image_filename = img_path
                        json_data = {'imgName': image_filename}

                        try:
                            # Send an HTTP POST request to the YOLO5 service
                            response = requests.post(yolo5_url, headers=headers, json=json_data)
                            # Logging the status code for debugging
                            logger.info(f"Response status code: {response.status_code}")
                            # Check the response status code
                            if response.status_code == 200:
                                response_data = json.loads(response.text)
                                logger.info(response_data)
                                for key, value in response_data.items():
                                    message = f"{key}: {value}"
                                    self.send_text(msg['chat']['id'], message)
                            logger.info("Prediction request sent successfully.")
                        except Exception as e:
                            logger.info(f"Error {e}")
                            logger.error('Error sending prediction request:', exc_info=True)

                    else:
                        self.send_text(msg['chat']['id'],"Error invalid caption\n Available captions are :\n 1)Blur\n2)mix\n3)Salt and pepper\n 4)predict")
                except Exception as e:
                    logger.info(f"Error {e}")
                    self.send_text(msg['chat']['id'], f'failed - try again later')
            else:
                self.send_text(msg['chat']['id'], "please provide caption")
