# standard imports
import json
import os
import pickle
import random
import time
# third-party imports
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
import tensorflow as tf
# local imports
from chatbot.chatbot_speech import SpeechToTextTextToSpeechIO
import chatbot.chatbot_global_state

# check if nltk punkt_tab is installed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
# ENVIRONMENT VARIABLES ###################################################################################################################################
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
PROJECT_TOOL_DIRECTORY = os.getenv('PROJECT_TOOL_DIRECTORY')

# CONSTANTS ###################################################################################################################################

# # Set the default SSL context for the entire script
# def create_ssl_context():
#     return ssl.create_default_context(cafile=certifi.where())

# ssl._create_default_https_context = create_ssl_context
# context = create_ssl_context()
# print(f"""SSL Context Details: 
#     CA Certs File: {context.cert_store_stats()} 
#     Protocol: {context.protocol} 
#     Options: {context.options} 
#     Verify Mode: {context.verify_mode}
#     Verify Flags: {context.verify_flags}
#     Check Hostname: {context.check_hostname}
#     CA Certs Path: {certifi.where()}
#     """)

# Establish the TTS bot's wake/activation word and script-specific global constants
# mic_on = True
# mic_on = False
# conversation_history = []
activation_word = os.getenv('ACTIVATION_WORD', 'robot')
username = os.getenv('USERNAME', 'None')
password = os.getenv('PASSWORD', 'None')
exit_words = os.getenv('EXIT_WORDS', 'None').split(',')
print(f'Activation word is {activation_word}\n\n')
lemmmatizer = WordNetLemmatizer()
intents = json.loads(open(f'{PROJECT_ROOT_DIRECTORY}/src/chatbot_model_training/chatbot_intents.json').read())
words = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/chatbot_model_training/chatbot_words.pkl', 'rb'))
classes = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/chatbot_model_training/chatbot_classes.pkl', 'rb'))
chatbot_model = tf.keras.models.load_model(f'{PROJECT_ROOT_DIRECTORY}/src/chatbot_model_training/chatbot_model.keras')
unrecognized_file_path = f'{PROJECT_ROOT_DIRECTORY}/src/chatbot_model_training/chatbot_unrecognized_message_intents.json'
print('Language models loaded.\n\n')

# CLASS DEFINITIONS ###################################################################################################################################

class ChatBotBrain:
    '''the ChatBotBrain class contains the app's entry point chatbot_model.keras model which operates as the "cns" of the chatbot and handles most of the routing for the app. 
    it is a very simple neural network model trained on the intents.json file of question/response pairs, some of which have callable functions. 
    shout out and credit to @neuralnine for the tutorual that taught me how to build this section of the app.
    '''
    def __init__(self):
        self.lemmatizer = lemmmatizer
        self.intents = intents
        self.words = words
        self.classes = classes
        self.chatbot_model = chatbot_model
        
    def clean_up_sentence(self, sentence):
        '''clean_up_sentence pre-processes words in the user input for use in the bag_of_words function'''
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        '''bag_of_words creates a bag of words from the user input for use in the predict_class function'''
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        '''predict_class predicts the class (tag) of the user input based on the bag of words'''
        bow = self.bag_of_words(sentence)
        res = self.chatbot_model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, chatbot_tools):
        '''takes user_input and uses the model to predict the most most likely class (tag) of the user input. 
        from there it will return a response from the chatbot and trigger a method if there's one attached to the JSON intent.'''
        if not intents_list:  # Check if intents_list is empty
            return "Sorry, what?"
        tag = intents_list[0]['intent']
        list_of_intents = self.intents['intents']
        result = None
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                if 'action' in i and i['action']:
                    action_method_name = i['action']
                    action_method = getattr(chatbot_tools, action_method_name, None)
                    if action_method:
                        # Call the method with only user_input as it's the only expected argument
                        action_method()
                break
        return result

    def chat(self, chatbot_tools):
        '''chat is the main chatbot entry point function. it takes user input, predicts the class (subject tag) of the user input, 
        and returns a response from the chatbot with the get_response function based on the most likely match from the predict_class function. 
        if the JSON intent for the matched response contains a function name in it's 'action' key, the function is called. 
        the function name is then used to call the function from the ChatBotTools class.'''
        # global conversation_history
        print('Start talking with the bot (type quit to stop)!')
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Online.')
        
        while True:
            # global chatbot_global_state.mic_on
            if not SpeechToTextTextToSpeechIO.is_speaking and chatbot.chatbot_global_state.mic_on:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue
                
                if user_input:
                    # conversation_history.append("User: " + user_input)
                    chatbot_tools.set_user_input(user_input)
                    query = user_input.lower().split()
                    if not query:
                        continue
                    
                    if len(query) > 1 and query[0] == activation_word and query[1] in exit_words:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Shutting down.')
                        time.sleep(.5)
                        break
                    
                    if len(query) > 1 and query[0] == activation_word:
                        query.pop(0)
                        user_input = ' '.join(query)
                        ints = self.predict_class(user_input)
                        res = self.get_response(ints, chatbot_tools)  
                        print(f'Bot: {res}')
                        SpeechToTextTextToSpeechIO.speak_mainframe(res)
                    
                    time.sleep(.1)
