# backend.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import librosa
import tensorflow as tf
import logging
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for request and response
class AudioConversionRequest(BaseModel):
    audio_file: bytes

class AudioConversionResponse(BaseModel):
    converted_audio: bytes

# Load the generator model
model_path = r"C:\Users\SID\Documents\Speech\voice_conversion_gan\models\G_NAM2WHSP.keras"
try:
    G_NAM2WHSP = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model from {model_path}")

# Method to pad audio and extract MFCC features
def process_audio(file_bytes):
    # Decode audio file and pad if necessary
    signal, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    signal = pad_audio(signal)
    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return mfccs

# Method to pad audio
def pad_audio(y, max_samples=1589449):
    padding_length = max_samples - len(y)
    if padding_length > 0:
        y = np.pad(y, (0, padding_length), 'constant')
    elif padding_length < 0:
        y = y[:max_samples]
    return y

@app.post("/convert_audio/", response_model=AudioConversionResponse)
async def convert_audio(request: AudioConversionRequest):
    try:
        # Process uploaded audio file
        mfccs = process_audio(request.audio_file)
        # Reshape and convert to tensor
        mfcc_tensor = tf.convert_to_tensor(mfccs, dtype=tf.float32)
        mfcc_tensor = tf.reshape(mfcc_tensor, (1, *mfccs.shape))
        # Generate converted audio using the model
        converted_mfccs = G_NAM2WHSP(mfcc_tensor)
        converted_mfccs = tf.squeeze(converted_mfccs).numpy()
        # Reshape to original shape for inverse transformation
        converted_audio = librosa.feature.inverse.mfcc_to_audio(converted_mfccs)
        # Convert to bytes
        converted_audio_bytes = converted_audio.tobytes()
        return AudioConversionResponse(converted_audio=converted_audio_bytes)
    except Exception as e:
        logger.error(f"Error in audio conversion: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Method to load and process audio file for testing
@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    try:
        file_bytes = await file.read()
        request = AudioConversionRequest(audio_file=file_bytes)
        response = await convert_audio(request)
        return response
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Invalid file upload")
