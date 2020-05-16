import numpy
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from flask import Flask, jsonify, make_response, request
import google.cloud.logging
import logging

client = google.cloud.logging.Client()

client.setup_logging()

app = Flask(__name__)

encoder_dict = imdb.get_word_index(path="imdb_word_index.json")
def encode(sent):
  lst = []
  for i in sent.lower().split():
    if i in encoder_dict.keys():
      if encoder_dict[i]<50000:
        lst.append(encoder_dict[i])
  return lst

dec_d = {v:k for k,v in encoder_dict.items()}
def decode(sent):
  out = ''
  for i in sent:
    if i in dec_d.keys():
      out = out + " " + dec_d[i]
  return out

print("Loading Model")
model = tf.keras.models.load_model("./imdb_model.hdf5")

logging.info("Model Loaded......")


def predicts(sent):
    X = encode(sent)
    X = sequence.pad_sequences([X], maxlen=300)
    X = model.predict(X)
    return X

@app.route('/seclassifer', methods = ['POST'])
def predict_sentiment():
    text = request.get_json()['text']
    print(text)
    prediction = predicts(text)
    sentiment = 'positive' if float(' '.join(map(str,prediction[0]))) > 0 else 'negetive'
    app.logger.info("prediction :" + str(prediction[0]) + "sentiment :" + sentiment)
    return jsonify({'predictions': prediction[0].tolist(), 'sentiment': sentiment})

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port = '5000', debug=True)