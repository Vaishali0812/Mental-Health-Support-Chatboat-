import re
import random
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle


def load_model(file_path):
    model = tf.keras.models.load_model(file_path)
    return model

def load_tokenizer_and_encoder(tokenizer_File_path, encoder_file_path):
    with open(tokenizer_File_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(encoder_file_path, 'rb') as f:
        encoder = pickle.load(f)
    return tokenizer, encoder

def generate_answer(tokenizer, model, lbl_enc, pattern, df):
    print(pattern)
    while True:
        # Check if user wants to quit the session
        if pattern.lower() in ['quit', 'exit']:
            return "Ending the session. Goodbye!"
            break

        # Preprocess the input pattern
        text = []
        txt = re.sub('[^a-zA-Z\']', ' ', pattern)
        txt = txt.lower()
        txt = txt.split()
        txt = " ".join(txt)
        text.append(txt)

        # Convert text to sequence and pad it
        x_test = tokenizer.texts_to_sequences(text)
        x_test = np.array(x_test).squeeze()
        x_test = pad_sequences([x_test], padding='post', maxlen=18)

        # Predict and get response
        y_pred = model.predict(x_test)
        y_pred = y_pred.argmax()
        tag = lbl_enc.inverse_transform([y_pred])[0]
        print("Predicted Tag:", tag)

        # Fetch responses for the predicted tag
        response_string = df[df['tag'] == tag]['responses'].iloc[0]
        responses = response_string.split(',')  # Split string into list assuming comma separation
        print("Possible responses:", responses)

        # Output a random response
        bot_response = random.choice(responses)
        print("Selected response:", bot_response)
        return bot_response

