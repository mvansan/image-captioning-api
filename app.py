from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from model import CNN_Encoder, TransformerEncoderLayer, TransformerDecoderLayer, ImageCaptioningModel, Embeddings
from PIL import Image
import io
import os
import boto3
import tempfile
from botocore import UNSIGNED
from botocore.client import Config

S3_BUCKET = "ic-model"
S3_REGION = "ap-southeast-1"
MODEL_FILES = {
    'cnn_model': 'cnn_model.h5',
    'encoder_weights': 'encoder_weights.weights.h5',
    'decoder_weights': 'decoder_weights.weights.h5'
}

TEMP_MODEL_DIR = tempfile.mkdtemp()

def download_model_from_s3():
    print("Đang tải models từ S3...")
    
    s3 = boto3.client('s3', region_name=S3_REGION, config=Config(signature_version=UNSIGNED))
    
    # s3 = boto3.client('s3', region_name=S3_REGION,
    #                 aws_access_key_id='YOUR_ACCESS_KEY',
    #                 aws_secret_access_key='YOUR_SECRET_KEY')
    
    for model_key, file_name in MODEL_FILES.items():
        local_path = os.path.join(TEMP_MODEL_DIR, file_name)
        
        if not os.path.exists(local_path):
            print(f"Đang tải {file_name}...")
            s3.download_file(S3_BUCKET, file_name, local_path)
            print(f"Đã tải {file_name} thành công!")
    
    return {
        'cnn_model_path': os.path.join(TEMP_MODEL_DIR, MODEL_FILES['cnn_model']),
        'encoder_weights_path': os.path.join(TEMP_MODEL_DIR, MODEL_FILES['encoder_weights']),
        'decoder_weights_path': os.path.join(TEMP_MODEL_DIR, MODEL_FILES['decoder_weights'])
    }

model_paths = download_model_from_s3()

app = Flask(__name__)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

tf.keras.utils.get_custom_objects().update({
    'CNN_Encoder': CNN_Encoder,
    'TransformerEncoderLayer': TransformerEncoderLayer,
    'TransformerDecoderLayer': TransformerDecoderLayer,
    'ImageCaptioningModel': ImageCaptioningModel,
    'Embeddings': Embeddings
})

MAX_LENGTH = 40
EMBEDDING_DIM = 512
UNITS = 512

encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 16, tokenizer)
cnn_model = CNN_Encoder()
model = ImageCaptioningModel(cnn_model, encoder, decoder)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss_fn)

try:
    if os.path.exists(model_paths['cnn_model_path']):
        model.cnn_model = tf.keras.models.load_model(model_paths['cnn_model_path'])
        print("Đã tải CNN model thành công!")
    
    dummy_image = tf.zeros((1, 299, 299, 3))
    features = model.cnn_model(dummy_image)
    enc_output = model.encoder(features, training=False)
    dec_input = tf.zeros((1, 1), dtype=tf.int32)
    _ = model.decoder(dec_input, enc_output, training=False)
    
    try:
        if os.path.exists(model_paths['encoder_weights_path']):
            model.encoder.load_weights(model_paths['encoder_weights_path'])
            print("Đã tải encoder weights thành công")
            
        if os.path.exists(model_paths['decoder_weights_path']):
            model.decoder.load_weights(model_paths['decoder_weights_path'])
            print("Đã tải decoder weights thành công")
    except Exception as e:
        print(f"Không thể tải encoder/decoder weights: {e}")
    
    print("Đã hoàn thành việc tải mô hình")
    
except Exception as e:
    print(f"Lỗi khi tải các phần mô hình: {e}")
    import traceback
    traceback.print_exc()

def load_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(np.array(img))
    return tf.expand_dims(img, 0)

def generate_caption(image_tensor):
    features = model.cnn_model(image_tensor)
    
    enc_output = model.encoder(features, training=False)
        
    if hasattr(tokenizer, 'get_vocabulary'):
        vocab = tokenizer.get_vocabulary()
        try:
            start_index = next((i for i, word in enumerate(vocab) if word == '[start]'), 1)
            end_index = next((i for i, word in enumerate(vocab) if word == '[end]'), 2)
        except:
            start_index = 1  
            end_index = 2    
        
        dec_input = tf.expand_dims([start_index], 0)
        
    else:
        start_index = tokenizer.word_index.get('[start]', 1)
        end_index = tokenizer.word_index.get('[end]', 2)
        dec_input = tf.expand_dims([start_index], 0)
    
    result = []
    
    for i in range(MAX_LENGTH):
        preds = model.decoder(dec_input, enc_output, training=False)
        pred_id = tf.argmax(preds[:, -1, :], axis=-1).numpy()[0]
        
        if hasattr(tokenizer, 'get_vocabulary'):
            word = vocab[pred_id] if pred_id < len(vocab) else ""
        else:
            word = next((w for w, idx in tokenizer.word_index.items() if idx == pred_id), "")
        
        if word != '[start]' and word != '[end]' and word != '<pad>' and word != '':
            result.append(word)
            
        if word == '[end]' or pred_id == end_index:
            break
            
        dec_input = tf.concat([dec_input, [[pred_id]]], axis=-1)

    return ' '.join(result)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    image_tensor = load_image(image)
    caption = generate_caption(image_tensor)

    return jsonify({'caption': caption})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
