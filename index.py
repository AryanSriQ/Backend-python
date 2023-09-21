from flask import request, Flask, send_file, jsonify
from flask_cors import CORS
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from diffusers import DiffusionPipeline
import scipy
import os
from dotenv import find_dotenv, load_dotenv
import pymongo
import random
import gridfs

load_dotenv(find_dotenv())
MONGODB_URL = os.getenv("MONGODB_URL")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
app = Flask(__name__)

CORS(app)


def mongo_conn():
    try:
        conn_str = f"{MONGODB_URL}"
        client = pymongo.MongoClient(conn_str)
        return client
    except Exception as e:
        print(e)


client = mongo_conn()


@app.route('/text_to_music', methods=['POST'])
def getdata():
    data = request.get_json()
    prompt = data.get('prompt')

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )

    random_suffix = random.randint(1, 100000)
    wav_filename = f"{prompt}_{random_suffix}.wav"
    wav_filename = wav_filename.replace(" ", "_")

    print(wav_filename)

    audio_values = model.generate(**inputs, max_new_tokens=256)
    sampling_rate = model.config.audio_encoder.sampling_rate

    scipy.io.wavfile.write(wav_filename, rate=sampling_rate, data=audio_values[0, 0].numpy())

    db = client.music
    fs = gridfs.GridFS(db)

    with open(wav_filename, "rb") as filedata:
        fs.put(filedata, filename=wav_filename)

    print("file uploaded")

    saved_file = fs.get_last_version(filename=wav_filename)

    if saved_file:
        return send_file(saved_file, mimetype='audio/wav')
    else:
        return "File not found", 404


@app.route('/text_to_image', methods=['POST'])
def text_to_image():
    data = request.get_json()
    prompt = data.get('prompt')

    pipeline = DiffusionPipeline.from_pretrained("Envvi/Inkpunk-Diffusion")
    img = pipeline(prompt).images[0]

    random_suffix = random.randint(1, 100000)
    img_filename = f"{prompt}_{random_suffix}.wav"
    img_filename = img_filename.replace(" ", "_")

    db = client["image"]
    fs = gridfs.GridFS(db)

    image_id = fs.put(img, filename=f"{img_filename}.png")

    return jsonify({"image_id": str(image_id)})


@app.route('/get_image/<image_id>', methods=['GET'])
def get_image(image_id):
    try:
        db = client["image"]  # Use the same database name where you stored the image
        fs = gridfs.GridFS(db)
        image = fs.get(image_id)

        if image:
            # Set the response headers to indicate the image content type
            response_headers = {
                'Content-Type': 'image/png',  # Adjust content type as needed
                'Content-Disposition': f'attachment; filename={image.filename}'
            }
            return send_file(image, as_attachment=True, download_name=image.filename, headers=response_headers)
        else:
            return "Image not found", 404
    except Exception as e:
        print("Error:", str(e))
        return "Internal Server Error", 500


if __name__ == "__main__":
    app.run(debug=True)
