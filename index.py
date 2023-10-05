from flask import request, Flask, send_file, jsonify
from flask_cors import CORS
from transformers import AutoProcessor, MusicgenForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from diffusers import DiffusionPipeline
import scipy
import os
from dotenv import find_dotenv, load_dotenv
import pymongo
import random
import gridfs
import re
import cohere

load_dotenv(find_dotenv())
MONGODB_URL = os.getenv("MONGODB_URL")
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
    time = data.get('time')
    print("time: ", time)

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

    audio_values = model.generate(**inputs, max_new_tokens=time)
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


@app.route('/prompt_to_text', methods=['POST'])
def post_data():
    req = request.get_json()
    prompt = req.get('prompt')
    co = cohere.Client('sDeY1e2YtCt3XOdGOxZgDecF2H9I108rwdLy6Emw')
    response = co.generate(
        model='e6366bc6-735b-4654-a319-5d4dd1fea947-ft',
        prompt=prompt,
        max_tokens=300)
    print(response)
    gen = 'Prediction: {}'.format(response.generations[0].text)
    response_data = {
        'output': gen
    }
    return jsonify(response_data)


@app.route('/translate', methods=['POST'])
def get_translation():

    data = request.get_json()
    prompt = data.get('prompt')
    from_lang = data.get('fromLanguage')
    to_lang = data.get('toLanguage')

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    input_text = f"translate from {from_lang} to {to_lang}: {prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    output = tokenizer.decode(outputs[0])

    cleaned_text = re.sub(r'<[^>]+>', '', output).lstrip()
    response_data = {
        'output': cleaned_text
    }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
