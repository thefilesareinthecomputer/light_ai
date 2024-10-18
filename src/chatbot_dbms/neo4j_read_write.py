'''
this is a testing module that's essentially a replica of the main chatbot module
but pared down to only have the ability to write to and read from the neo4j graph database.
'''

from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import os
import pickle
import queue
import random
import re
import ssl
import sys
import subprocess
import threading
import time
import traceback
from neo4j import GraphDatabase
from nltk.stem import WordNetLemmatizer
import certifi
import google.generativeai as genai
import numpy as np
import nltk
import pandas as pd
import speech_recognition as sr
import tensorflow as tf
import yfinance as yf

# CONSTANTS

load_dotenv()
JAVA_HOME = os.getenv('JAVA_HOME')
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
NEO4J_PATH = os.getenv("NEO4J_PATH")
USER_PREFERRED_LANGUAGE = os.getenv('USER_PREFERRED_LANGUAGE', 'en')  # 2-letter lowercase
USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Evan')  # Daniel
USER_PREFERRED_NAME = os.getenv('USER_PREFERRED_NAME', 'User')  # Title case
USER_SELECTED_HOME_CITY = os.getenv('USER_SELECTED_HOME_CITY', 'None')  # Title case
USER_SELECTED_HOME_STATE = os.getenv('USER_SELECTED_HOME_STATE', 'None')  # Title case
USER_SELECTED_HOME_LAT = os.getenv('USER_SELECTED_HOME_LAT', 'None')  # Float with 6 decimal places
USER_SELECTED_HOME_LON = os.getenv('USER_SELECTED_HOME_LON', 'None')  # Float with 6 decimal places 
USER_SELECTED_TIMEZONE = os.getenv('USER_SELECTED_TIMEZONE', 'America/Chicago')  # Country/State format
USER_STOCK_WATCH_LIST = os.getenv('USER_STOCK_WATCH_LIST', 'None').split(',')  # Comma separated list of stock symbols
USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATABASES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_databases')
FILE_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_generated_files')
LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_local_models')
BASE_KNOWLEDGE_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_base_knowledge')
SECRETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_secrets')
SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_source_data')
SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_tests')
UTILITIES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'utilities')

# Set the default SSL context for the entire script
def create_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = create_ssl_context
context = create_ssl_context()

# Set API keys and other sensitive information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
google_documentation_search_engine_id = os.getenv('GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID')
print('API keys and other sensitive information loaded from environment variables.\n\n')

# Establish the TTS bot's wake/activation word and script-specific global constants
activation_word = os.getenv('ACTIVATION_WORD', 'robot')
password = os.getenv('PASSWORD', 'None')
exit_words = os.getenv('EXIT_WORDS', 'None').split(',')
print(f'Activation word is {activation_word}\n\n')

# Initialize the language models
# pocket_sphinx_model_files = os.path.join(LOCAL_LLMS_DIR, "sphinx4-5prealpha-src")
genai.configure(api_key=google_gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-pro')  
gemini_vision_model = genai.GenerativeModel('gemini-pro-vision')
print('Google Gemini LLM initialized.\n\n') 

lemmmatizer = WordNetLemmatizer()
intents = json.loads(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_intents.json').read())
words = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_words.pkl', 'rb'))
classes = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_classes.pkl', 'rb'))
chatbot_model = tf.keras.models.load_model(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_model.keras')
unrecognized_file_path = f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_unrecognized_message_intents.json'
print('Local chatbot model loaded.\n\n')

# CLASS DEFINITIONS

class SpeechToTextTextToSpeechIO:
    '''SpeechToTextTextToSpeechIO handles the speech to text and text to speech functionality of the chatbot. It also handles the speech output queue.
    the speech output queue places all text chunks output from the bot and plays them in order so they don't overlap. The speech manager thread is constantly checking the queue for new items. 
    the speech manager runs on its own thread so that the bot can still recieve input while speaking.'''
    speech_queue = queue.Queue()
    queue_lock = threading.Lock()
    is_speaking = False

    @classmethod
    def parse_user_speech(cls):
        '''parse_user_speech is the main speech recognition function. 
        it uses the google speech recognition API to parse user speech from the microphone into text'''
        listener = sr.Recognizer()
        while True:
            if cls.is_speaking == True:
                continue
            if cls.is_speaking == False:
                print('Listening...')
                try:
                    with sr.Microphone() as source:
                        listener.pause_threshold = 2
                        input_speech = listener.listen(source, timeout=10, phrase_time_limit=10) 
                    print('Processing...')
                    query = listener.recognize_google(input_speech, language='en_US') 
                    print(f'You said: {query}\n')
                    return query

                except sr.WaitTimeoutError:
                    print('Listening timed out. Please try again.')
                    return None

                except sr.UnknownValueError:
                    print('Speech not recognized. Please try again.')
                    return None
                
    @classmethod
    def speech_manager(cls):
        '''speech_manager handles the flow of the speech output queue in a first in first out order, 
        ensuring that only one speech output is running at a time.'''
        while True:
            cls.queue_lock.acquire()
            try:
                if not cls.speech_queue.empty():
                    item = cls.speech_queue.get()
                    if item is not None:
                        cls.is_speaking = True
                        text, rate, chunk_size, voice = item
                        if text:
                            chunks = [' '.join(text.split()[i:i + chunk_size]) for i in range(0, len(text.split()), chunk_size)]
                            for chunk in chunks:
                                subprocess.call(['say', '-v', voice, '-r', str(rate), chunk])
                        cls.speech_queue.task_done()
            finally:
                cls.queue_lock.release()
            cls.is_speaking = False
            time.sleep(0.2)

    @classmethod
    def calculate_speech_duration(cls, text, rate):
        '''calculate_speech_duration calculates the duration of the speech based on text length and speech rate.'''
        words = text.split() if text else []
        number_of_words = len(words)
        minutes = number_of_words / rate
        seconds = minutes * 60
        return seconds + 1
    
    @classmethod
    def speak_mainframe(cls, text, rate=190, chunk_size=1000, voice=USER_PREFERRED_VOICE):
        '''speak_mainframe contains the bot's speech output voice settings, and it puts each chunk of text output from the bot or the LLM 
        into the speech output queue to be processed in sequential order.'''
        cls.queue_lock.acquire()
        try:
            cls.speech_queue.put((text, rate, chunk_size, voice))
            speech_duration = cls.calculate_speech_duration(text, rate)
        finally:
            cls.queue_lock.release()
        return speech_duration
            
class ChatBotApp:
    '''the ChatBotApp class contains the app's entry point chatbot_model.keras model which operates as the central chatbot brain and routing system for the app. '''
    def __init__(self):
        self.project_root_directory = PROJECT_ROOT_DIRECTORY
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
        '''chat is the main chatbot entry point function.'''
        print('Start talking with the bot (type quit to stop)!')
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Online.')
        
        while True:
            if not SpeechToTextTextToSpeechIO.is_speaking:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue
                
                if user_input:
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
                                            
class ChatBotTools:
    '''ChatBotTools contains all of the functions that are called by the chatbot_model, including larger llms, system commands, utilities, and api connections 
    to various services. it contains all of the methods that are called by the JSON intents in the chatbot_intents.json file in response to user input. '''
    data_store = {}
    
    def __init__(self):
        self.user_input = None  

    def set_user_input(self, input_text):
        '''Sets the user input for use in other methods.'''
        self.user_input = input_text
    
# ###################################################################################################################

    @staticmethod
    def save_note():
        SpeechToTextTextToSpeechIO.speak_mainframe('What is the subject of the note?')
        time.sleep(1.5)
        while True:
            subject_response = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not subject_response:
                continue  # Wait for valid input

            subject_query = subject_response.lower().split()
            if not subject_query or subject_query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Note saving cancelled.')
                return
            else:
                break  # Valid input received

        subject = subject_response.strip().lower()

        SpeechToTextTextToSpeechIO.speak_mainframe('Please say the content of the note.')
        time.sleep(1.5)
        while True:
            content_response = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not content_response:
                continue  # Wait for valid input

            content_query = content_response.lower().split()
            if not content_query or content_query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Note saving cancelled.')
                return
            else:
                break  # Valid input received

        content = content_response.strip()

        try:
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                with driver.session() as session:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    session.run("""
                        CREATE (n:UserVoiceNotes:Note:UserChatBotInteractions {subject: $subject, content: $content, timestamp: $timestamp})
                    """, subject=subject, content=content, timestamp=timestamp)
                SpeechToTextTextToSpeechIO.speak_mainframe('Note saved successfully.')
        except Exception as e:
            SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while saving the note.')
            print(e)
        
# ###################################################################################################################                     

    @staticmethod
    def recall_notes():
        SpeechToTextTextToSpeechIO.speak_mainframe('Say "list", "statistics", or "recall".')
        while True:
            user_choice = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_choice:
                continue

            choice_query = user_choice.lower().split()
            if choice_query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Operation cancelled.')
                return

            # Listing available subjects
            if 'list' in choice_query:
                try:
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                        with driver.session() as session:
                            result = session.run("MATCH (n:Note) RETURN DISTINCT n.subject ORDER BY n.subject")
                            subjects = [record['n.subject'] for record in result]
                        if subjects:
                            subject_list = ', '.join(subjects)
                            SpeechToTextTextToSpeechIO.speak_mainframe(f"Available subjects: {subject_list}")
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe('No subjects found.')
                except Exception as e:
                    SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while retrieving subjects.')
                    print(e)
                return

            # Getting database statistics
            elif 'statistics' in choice_query:
                try:
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                        with driver.session() as session:
                            # Count nodes by label
                            label_counts = session.run("MATCH (n) UNWIND labels(n) AS label RETURN label, COUNT(*) AS count")
                            labels_info = [f"Label {record['label']}: {record['count']} nodes" for record in label_counts]

                            # Add more statistics as needed

                        if labels_info:
                            stats_info = '\n'.join(labels_info)
                            SpeechToTextTextToSpeechIO.speak_mainframe(f"Database statistics:\n{stats_info}")
                            print(f"Database statistics:\n{stats_info}")
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe('No statistics found.')
                except Exception as e:
                    SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while retrieving database statistics.')
                    print(e)
                return

            # Recalling specific notes
            elif 'recall' in choice_query:
                SpeechToTextTextToSpeechIO.speak_mainframe('Which subject notes would you like to recall?')
                subject_response = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not subject_response or subject_response.lower().split()[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Note recall cancelled.')
                    return

                subject = subject_response.strip().lower()
                try:
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                        with driver.session() as session:
                            result = session.run("""
                                MATCH (n:Note {subject: $subject})
                                RETURN n.content, n.timestamp
                                ORDER BY n.timestamp DESC
                            """, subject=subject)
                            notes = [f"Date: {record['n.timestamp']}, Note: {record['n.content']}" for record in result]
                        if notes:
                            SpeechToTextTextToSpeechIO.speak_mainframe(" ".join(notes))
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe('No notes found for the subject.')
                except Exception as e:
                    SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred during note recall.')
                    print(e)
                return

            else:
                SpeechToTextTextToSpeechIO.speak_mainframe('Please specify "list", "statistics", or "recall".')
                                
# ###################################################################################################################      

if __name__ == '__main__':
    threading.Thread(target=SpeechToTextTextToSpeechIO.speech_manager, daemon=True).start()
    chatbot_app = ChatBotApp()
    chatbot_tools = ChatBotTools()
    chatbot_app.chat(chatbot_tools)