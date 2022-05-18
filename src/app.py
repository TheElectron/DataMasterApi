import os
import random
import librosa
import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify, abort

app = Flask(__name__)

HOP_LENGTH = 512
SAMPLE_RATE = 22050
MODEL_PATH = os.path.join(os.getcwd(), "model")
model = keras.models.load_model(MODEL_PATH)


def get_audio_len(path):
    """ Get Audio Duration"""
    audio_data, sr = librosa.load(path, sr=SAMPLE_RATE)
    audio_len = librosa.get_duration(y=audio_data, sr=SAMPLE_RATE)
    return round(audio_len, 2)


def get_features(path):
    """ Get MFCC Features For Audio Data"""
    features = []
    audio_data, sr = librosa.load(path, sr=SAMPLE_RATE)
    audio_data = librosa.util.fix_length(audio_data, size=5 * SAMPLE_RATE)
    features.append(librosa.feature.mfcc(audio_data, n_mfcc=16,
                                         sr=SAMPLE_RATE, hop_length=HOP_LENGTH))
    return np.array(features)


@app.route('/')
def hello_world():
    return '<h1> Hello World From Heroku! </h2>'


@app.route('/api/predict', methods=['POST'])
def predict_by_audio_post():
    """
    METHOD: POST
    DESCRIPTION: Receive audio file, processing and make a emotion prediction
    """
    try:
        file_name = f"audio_file_{random.randint(1, 1000)}"
        audio_file = request.files["file"]
        audio_file.save(file_name)

        # TODO: REFATORAR OA CHAMADA PARA O MODELO
        mfcc_data = get_features(file_name)
        predicted_labels = model.predict(mfcc_data)
        predicted_labels = predicted_labels.argmax()
        target_names = ["Neutra", "Calma", "Felicidade", "Tristeza", "Raiva", "Medo", "Nojo", "Surpresa"]

        response_data = {"predicted_label": target_names[predicted_labels],
                         "duration": get_audio_len(file_name)}
        os.remove(file_name)
        return jsonify(response_data), 200
    except IndexError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
