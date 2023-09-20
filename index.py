from flask import request, Flask, send_file
from flask_cors import CORS
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import os
from dotenv import find_dotenv, load_dotenv
import pymongo
import random
import gridfs

load_dotenv(find_dotenv())
MONGODB_URL = os.getenv("MONGODB_URL")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/warp-ai/wuerstchen"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
app = Flask(__name__)

CORS(app, supports_credentials=True)


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


if __name__ == "__main__":
    app.run()
