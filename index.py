from flask import request, Flask, send_file
from flask_cors import CORS
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import requests
import torch
from diffusers import AutoPipelineForText2Image
import io
from PIL import Image

# from random import randint

app = Flask(__name__)
CORS(app)
API_URL = "https://api-inference.huggingface.co/models/warp-ai/wuerstchen"
headers = {"Authorization": "Bearer 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"}


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

    audio_values = model.generate(**inputs, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(f"{prompt}.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

    # wav_file_path = f"{prompt}.wav".replace(" ", "_")
    # wav_file_path += randint(1, 100)
    return send_file(f"{prompt}.wav", as_attachment=True)


# @app.route('/text_to_image', methods=['POST'])
# def query():
#     device = "cuda"
#     dtype = torch.float16
#
#     pipeline = AutoPipelineForText2Image.from_pretrained(
#         "warp-diffusion/wuerstchen", torch_dtype=dtype
#     ).to(device)
#
#     caption = "Anthropomorphic cat dressed as a fire fighter"
#
#     output = pipeline(
#         prompt=caption,
#         height=1024,
#         width=1024,
#         prior_guidance_scale=4.0,
#         decoder_guidance_scale=0.0,
#     ).images
#
#     return output
    # payload = request.get_json()
    # response = requests.post(API_URL, headers=headers, json=payload)
    # print(response.content)
    # return response.content


# query()

if __name__ == "__main__":
    app.run()
