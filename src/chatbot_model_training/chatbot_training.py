from dotenv import load_dotenv
import json
import os
import pickle
import random

from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
import tensorflow as tf

load_dotenv()
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def train_chatbot_model():
    lemmatizer = WordNetLemmatizer()

    # intents = json.loads(open(f'{SCRIPT_DIR_PATH}/chatbot_intents.json').read())

    # changed to 'with' because of a warning about not clising opened files after use
    with open(f'{SCRIPT_DIR_PATH}/chatbot_intents.json', 'r') as f:
        intents = json.load(f)
        
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
                
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))

    classes = sorted(set(classes))

    # pickle.dump(words, open(f'{SCRIPT_DIR_PATH}/chatbot_words.pkl', 'wb'))
    # pickle.dump(classes, open(f'{SCRIPT_DIR_PATH}/chatbot_classes.pkl', 'wb'))

    # Use 'with' to ensure files are closed after writing
    with open(f'{SCRIPT_DIR_PATH}/chatbot_words.pkl', 'wb') as f:
        pickle.dump(words, f)

    with open(f'{SCRIPT_DIR_PATH}/chatbot_classes.pkl', 'wb') as f:
        pickle.dump(classes, f)
        
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
            
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        
        training.append([bag, output_row])

    # Shuffle the training data
    random.shuffle(training)

    # Split the training data into X and Y
    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    model.save(f'{SCRIPT_DIR_PATH}/chatbot_model.keras', hist)
    # removed the hist argument from model.save() because it was causing an error
    # model.save(f'{SCRIPT_DIR_PATH}/chatbot_model.keras')
    # check if the model is saved
    if os.path.exists(f'{SCRIPT_DIR_PATH}/chatbot_model.keras'):
        print("Model training complete!")
    else:
        print("Model training failed!")

if __name__ == '__main__':
    train_chatbot_model()