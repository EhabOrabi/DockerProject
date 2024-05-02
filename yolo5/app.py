import json
import time
from pathlib import Path
import boto3
import pymongo
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os

from pymongo import MongoClient

# Get the bucket name from the environment variable
images_bucket = os.environ['BUCKET_NAME']


# Open the YAML file in read mode
with open("data/coco128.yaml", "r") as stream:
    # Load the YAML content safely, Extract the value associated with the key 'names'
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.json.get('imgName')
    logger.info(f'img_name is received is {img_name}')
    photo_s3_name = img_name.split("/")

    file_path_pic_download = os.getcwd() + "/" + str(photo_s3_name[1])
    logger.info(file_path_pic_download)
    client = boto3.client('s3')
    client.download_file(images_bucket, str(photo_s3_name[1]), file_path_pic_download)


    # TODO download img_name from S3, store the local image path in the original_img_path variable.
    #  The bucket name is provided as an env var BUCKET_NAME.
    original_img_path = file_path_pic_download
    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    path = Path(f'static/data/{prediction_id}/{str(photo_s3_name[1])}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        pass
    predicted_img_path = Path(f'static/data/{prediction_id}/{str(photo_s3_name[1])}')
    path_str = str(predicted_img_path)
    json_str = json.dumps({"path": path_str})
    json_data = json.loads(json_str)
    unique_filename = str(uuid.uuid4()) + '.jpeg'
    client.upload_file(json_data["path"], images_bucket, unique_filename)

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).

    # Parse prediction labels and create a summary
    path = Path(f'static/data/{prediction_id}/labels/{photo_s3_name[1].split(".")[0]}.txt')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        pass
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{photo_s3_name[1].split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]
            logger.info(f'prediction: {prediction_id}/{photo_s3_name[1]}. prediction summary:\n\n{labels}')
            prediction_summary = {
                'prediction_id': prediction_id,
                'original_img_path': photo_s3_name[1],
                'predicted_img_path': json_data["path"],
                'labels': labels,
                'time': time.time()
            }

            logger.info(prediction_summary)

            client = MongoClient("mongodb://mongo_primary:27017/")
            db = client['mydatabase']
            collection = db['mycollection']
            collection.insert_one(prediction_summary)

        # TODO store the prediction_summary in MongoDB
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["mongo_primary"]
        # Select or create a collection for predictions
        collection = db["predictions"]
        # Insert JSON data into MongoDB
        collection.insert_one(prediction_summary)
        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
