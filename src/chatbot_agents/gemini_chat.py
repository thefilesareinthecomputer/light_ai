# standard imports
from datetime import datetime, timedelta
import asyncio
import inspect
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
import webbrowser
# third party imports
from neo4j import GraphDatabase
from nltk.stem import WordNetLemmatizer
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
import certifi
import flet as ft
import google.generativeai as genai
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import numpy as np
import nltk
nltk.download('punkt_tab')
import pandas as pd
import PIL.Image
import pyautogui
import pytz
import requests
import speech_recognition as sr
import tensorflow as tf
import wikipedia
import wolframalpha
import yfinance as yf

from app_secrets.user_persona import (
    user_demographics, 
    user_skills_and_experience,
    user_interests, 
    user_favorite_quotes,
    )

# ENVIRONMENT VARIABLES ###################################################################################################################################
from dotenv import load_dotenv
load_dotenv()
JAVA_HOME = os.getenv('JAVA_HOME')
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
NEO4J_PATH = os.getenv("NEO4J_PATH")
USER_PREFERRED_LANGUAGE = os.getenv('USER_PREFERRED_LANGUAGE', 'en')  # 2-letter lowercase
USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Evan')
USER_PREFERRED_NAME = os.getenv('USER_PREFERRED_NAME', 'User')  # Title case
USER_SELECTED_HOME_CITY = os.getenv('USER_SELECTED_HOME_CITY', 'None')  # Title case
USER_SELECTED_HOME_COUNTY = os.getenv('USER_SELECTED_HOME_COUNTY', 'None')  # Title case
USER_SELECTED_HOME_STATE = os.getenv('USER_SELECTED_HOME_STATE', 'None')  # Title case
USER_SELECTED_HOME_COUNTRY = os.getenv('USER_SELECTED_HOME_COUNTRY', 'None')  # 2-letter country code
USER_SELECTED_HOME_LAT = os.getenv('USER_SELECTED_HOME_LAT', 'None')  # Float with 6 decimal places
USER_SELECTED_HOME_LON = os.getenv('USER_SELECTED_HOME_LON', 'None')  # Float with 6 decimal places 
USER_SELECTED_TIMEZONE = os.getenv('USER_SELECTED_TIMEZONE', 'America/Chicago')  # Country/State format
USER_STOCK_WATCH_LIST = os.getenv('USER_STOCK_WATCH_LIST', 'None').split(',')  # Comma separated list of stock symbols
USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
PROJECT_CODE_DIRECTORY = os.getenv('PROJECT_CODE_DIRECTORY')
PROJECT_TOOL_DIRECTORY = os.getenv('PROJECT_TOOL_DIRECTORY')
DATABASES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_databases')
FILE_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_generated_files')
LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_local_models')
BASE_KNOWLEDGE_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_base_knowledge')
SECRETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_secrets')
SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_source_data')
SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_tests')
UTILITIES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'utilities')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
folders_to_create = [DATABASES_DIR_PATH, 
                     FILE_DROP_DIR_PATH, 
                     LOCAL_LLMS_DIR, 
                     BASE_KNOWLEDGE_DIR_PATH, 
                     SECRETS_DIR_PATH, 
                     SOURCE_DATA_DIR_PATH, 
                     SRC_DIR_PATH, 
                     TESTS_DIR_PATH, 
                     UTILITIES_DIR_PATH]
for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)

# CONSTANTS ###################################################################################################################################

# Set the default SSL context for the entire script
def create_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = create_ssl_context
context = create_ssl_context()
print(f"""SSL Context Details: 
    CA Certs File: {context.cert_store_stats()} 
    Protocol: {context.protocol} 
    Options: {context.options} 
    Verify Mode: {context.verify_mode}
    Verify Flags: {context.verify_flags}
    Check Hostname: {context.check_hostname}
    CA Certs Path: {certifi.where()}
    """)

# Set API keys and other sensitive information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
openai_api_key=os.getenv('OPENAI_API_KEY')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
google_documentation_search_engine_id = os.getenv('GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID')
google_job_search_search_engine_id = os.getenv('GOOGLE_JOB_SEARCH_SEARCH_ENGINE_ID')
google_health_search_engine_id = os.getenv('GOOGLE_HEALTH_SEARCH_ENGINE_ID')
google_research_search_engine_id = os.getenv('GOOGLE_RESEARCH_SEARCH_ENGINE_ID')
google_restaurant_search_engine_id = os.getenv('GOOGLE_RESTAURANT_SEARCH_ENGINE_ID')
print('API keys and other sensitive information loaded from environment variables.\n\n')

# Establish the TTS bot's wake/activation word and script-specific global constants
# mic_on = True
mic_on = False
conversation_history = []
activation_word = os.getenv('ACTIVATION_WORD', 'robot')
username = os.getenv('USERNAME', 'None')
password = os.getenv('PASSWORD', 'None')
exit_words = os.getenv('EXIT_WORDS', 'None').split(',')
print(f'Activation word is {activation_word}\n\n')

# Initialize the language models
print('Available language models:')
# pocket_sphinx_model_files = os.path.join(LOCAL_LLMS_DIR, "sphinx4-5prealpha-src")  # for offline speech recognition (not good)
genai.configure(api_key=google_gemini_api_key)
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
# gemini_model = genai.GenerativeModel('gemini-pro')  
gemini_model = genai.GenerativeModel('gemini-1.0-pro-latest')  
gemini_vision_model = genai.GenerativeModel('gemini-pro-vision')
lemmmatizer = WordNetLemmatizer()
intents = json.loads(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_intents.json').read())
words = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_words.pkl', 'rb'))
classes = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_classes.pkl', 'rb'))
chatbot_model = tf.keras.models.load_model(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_model.keras')
unrecognized_file_path = f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_unrecognized_message_intents.json'
print('Language models loaded.\n\n')

# CLASS DEFINITIONS ###################################################################################################################################


@staticmethod
def gemini_chat():
    '''gemini_chat is a general purpose chat thread with the Gemini model, with optional branches for 
    running thorough diagnostics of the app codebase, calling Gemini as a pair programmer, and accessing data 
    stored in the data_store variable which is housed within the ChatBotTools class.'''
    SpeechToTextTextToSpeechIO.speak_mainframe('Initializing...')
    chat = gemini_model.start_chat(history=[])
    
    prompt_template = '''### <* SYSTEM MESSAGE BEGIN *> ### 
    ### <* these are your instructions *> ### 
    Gemini, you are in a verbal chat with the user via a 
    STT / TTS AI companion agent application. You must generate text that is structured like natural spoken language, 
    not long written text. you must avoid monologuing or including anything in the output that will 
    not sound like natural spoken language. 
    you will refer to the user directly in the second person tense. You are talking to the user directly. 
    You are a trusted advisor to the user. The user will come to you for help solving problems and answering questions. 
    you must confirm your understanding of these instructions by simply saying "Chat loop is open", 
    and then await another prompt from the user. 
    you must not generate markdown. you must not generate paragraphs. You are engaging in real-time conversational dialogue with a human. 
    you must not generate text that will take a long time to play back or read. 
    Each message back and forth must be only a few sentences at maximum. Some may only require a few words. you must direct and to the point. 
    you will emulate a human conversationalist. 
    After you confirm you understand this message, the chat will proceed. 
    ### <* wait for user input after you acknowledge this message *> ### 
    ### <* SYSTEM MESSAGE END *> ### 
    '''

    intro_response = chat.send_message(f'{prompt_template}', stream=True)
    
    if intro_response:
        for chunk in intro_response:
            SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            time.sleep(0.1)
    
    while True:
        global mic_on
        if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            if query[0] == 'stoic' and query[1] == 'lesson':
                SpeechToTextTextToSpeechIO.speak_mainframe('OK - thinking.')
                url = "https://stoic-quotes.com/api/quote"  # Fetch a single quote
                response = requests.get(url)
                # Check the status code and handle any issues
                if response.status_code == 200:
                    try:
                        quotes_data = response.json()

                        # Handle case where it's a single quote
                        if isinstance(quotes_data, dict):
                            quote_text = quotes_data['text']
                            quote_author = quotes_data['author']
                            print(f"From {quote_author}: {quote_text}")
                            SpeechToTextTextToSpeechIO.speak_mainframe(f"From {quote_author}: {quote_text}")
                            stoic_response = chat.send_message(f'''You are a stoic philosophy expert. You are teaching the user about the following quote from 
                                                                {quote_author}: {quote_text}. Mimic the philosopher who authored this quote, and give the user a deep 
                                                                analysis of the quote. Teach the user the deeper lessons underlying this quote, and teach the user how to apply these 
                                                                principles within the context of modern life. Provide references to similar stoic principles, and connect these principles 
                                                                with common themes from the daily realities of modern living. Give examples of how these stoic principles apply to 
                                                                modern issues like technology and money and health, and how the user as an individual can apply these principles to achieve 
                                                                greater happiness and quality of life. 
                                                                To repeat, the quote you're currently examining is from {quote_author}. 
                                                                The quote is: {quote_text}. Provide your insights on this quote now.''', stream=True)
                            if stoic_response:
                                for chunk in stoic_response:
                                    if hasattr(chunk, 'parts'):
                                        # Concatenate the text from each part
                                        full_text = ''.join(part.text for part in chunk.parts)
                                        SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                        print(full_text)
                                    else:
                                        # If it's a simple response, just speak and print the text
                                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                        print(chunk.text)
                                    time.sleep(0.1)
                                time.sleep(1)
                            continue
                        
                        # Handle case where it's multiple quotes
                        elif isinstance(quotes_data, list) and quotes_data:
                            for quote in quotes_data:
                                SpeechToTextTextToSpeechIO.speak_mainframe(f"From {quote['author']}: {quote['text']}")
                        
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe("No quotes available at the moment.")
                    except ValueError:
                        # Handle JSON decoding error
                        SpeechToTextTextToSpeechIO.speak_mainframe("Error: Unable to decode the response from the Stoic API.")
                        print(f"Failed to decode JSON: {response.text}")
                else:
                    # Handle non-200 status codes
                    SpeechToTextTextToSpeechIO.speak_mainframe(f"Error: Received a {response.status_code} status code from the Stoic API.")
                    print(f"Error: Status Code {response.status_code}, Response: {response.text}")
                
            if query[0] == 'access' and query [1] == 'data':
                SpeechToTextTextToSpeechIO.speak_mainframe('Accessing global memory.')
                data_store = ChatBotTools.data_store
                print(ChatBotTools.data_store)
                data_prompt = f'''### "*SYSTEM MESSAGE*" ### Gemini, the user is currently speaking to you from within their TTS / STT app. 
                Here is the data they've pulled into the conversation so far. The user is going to ask you to discuss this data: 
                \n {data_store}\n
                ### "*SYSTEM MESSAGE*" ### Gemini, read and deeply understand all the nuances of the data and metadata in 
                this dictionary - examine this data and place it all together in context. 
                Read the data, and then say "I've read the data. What do you want to discuss first?", and then await further instructions. 
                ### "*wait for user input after you acknowledge this message*" ###'''
                data_response = chat.send_message(f'{data_prompt}', stream=True)
                if data_response:
                    for chunk in data_response:
                        if hasattr(chunk, 'parts'):
                            # Concatenate the text from each part
                            full_text = ''.join(part.text for part in chunk.parts)
                            SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                            print(full_text)
                        else:
                            # If it's a simple response, just speak and print the text
                            SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                            print(chunk.text)
                        time.sleep(0.1)
                    time.sleep(1)
                continue

            if query[0] in ['sous', 'sue', 'soo', 'su', 'tsu', 'sew', 'shoe', 'shoo'] and query [1] in ['chef', 'shef', 'chefs', 'shefs']:
                SpeechToTextTextToSpeechIO.speak_mainframe('Yes chef.')
                egg_mix_recipe = {
                    "recipe": "Egg Mix",
                    "ingredients": [
                        {"quantity": "6000", "unit": "g", "ingredient": "Egg"},
                        {"quantity": "240", "unit": "g", "ingredient": "Creme Fraiche"},
                        {"quantity": "30", "unit": "g", "ingredient": "Kosher Salt"}
                    ],
                    "yield": {"quantity": "6", "unit_of_measure": "liter"},
                    "source_document": "recipes_brunch.docx",
                    "instructions": {"step_1": "Crack eggs into a large plastic Cambro.",
                                    "step_2": "Add creme fraiche and salt.",
                                    "step_3": "Mix with an immersion blender until smooth",
                                    "step_4": "Pass through a mesh sieve, then label and store in the walk-in until needed.",},
                    "shelf_life": "2 days",
                    "tools": ["Immersion Blender", "Mesh Sieve", "Scale", "Cambro", "Label Maker"],
                }
                print(egg_mix_recipe)
                chef_prompt = f'''### "*SYSTEM MESSAGE*" ### Gemini, the user is currently speaking to you from within their TTS / STT app. 
                The user's role is Executive Chef at a restaurant. Your role is Sous Chef.
                Here is the recipe data they've pulled into the conversation so far. The user is going to ask you to discuss this data: 
                \n {egg_mix_recipe}\n
                ### "*SYSTEM MESSAGE*" ### Gemini, read and understand all the nuances of the recipe data and metadata in 
                this dictionary - examine this data and place it all together in context. 
                Read the data, and then say "Yes Chef. What do you want to discuss first?", and then await further instructions. 
                ### "*wait for user input after you acknowledge this message*" ###'''
                print(chef_prompt)
                chef_response = chat.send_message(f'{chef_prompt}', stream=True)
                if chef_response:
                    for chunk in chef_response:
                        if hasattr(chunk, 'parts'):
                            # Concatenate the text from each part
                            full_text = ''.join(part.text for part in chunk.parts)
                            SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                            print(full_text)
                        else:
                            # If it's a simple response, just speak and print the text
                            SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                            print(chunk.text)
                        time.sleep(0.1)
                    time.sleep(1)
                continue

            if query[0] in ['pair', 'pear', 'pare', 'payer', 'prayer', 'hair', 'tare', 'tear', 'air'] and query[1] == 'programmer':
                number_words = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                inspection_code = {}
                SpeechToTextTextToSpeechIO.speak_mainframe('Reading the code.')
                
                # Summarize and list all classes
                diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__])
                class_names = list(diagnostic_summary.keys())
                
                # Present the classes to the user
                class_list_text = "Here are the available classes: "
                for i, cls in enumerate(class_names, 1):
                    class_list_text += f"{i}. {cls}. "
                
                SpeechToTextTextToSpeechIO.speak_mainframe(class_list_text)
                print(f"Available classes: {class_names}")
                
                # Let the user pick a class by number
                SpeechToTextTextToSpeechIO.speak_mainframe('Please choose a class by number.')
                while True:
                    user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                    if not user_input:
                        continue
                    
                    # Convert spoken number words to integers
                    user_input = user_input.strip().lower()
                    selected_number = number_words.get(user_input, user_input)

                    try:
                        selected_number = int(selected_number)
                        if selected_number < 1 or selected_number > len(class_names):
                            SpeechToTextTextToSpeechIO.speak_mainframe('Invalid number. Please choose a valid class number.')
                            continue
                    except ValueError:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Please say a valid number.')
                        continue
                    
                    # Get the selected class
                    selected_class = class_names[selected_number - 1]
                    
                    # Drill down into the selected class and check for source code
                    class_details = ChatBotTools.summarize_module(sys.modules[__name__], class_name=selected_class)
                    print(f"Class details: {class_details['classes'][selected_class]}")
                    if 'source_code' in class_details['classes'][selected_class]:
                        # Add class source code to inspection_code
                        inspection_code[selected_class] = class_details['classes'][selected_class]['source_code']
                        print(f"Inspection code: {inspection_code}")
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"Code for {selected_class} added. Do you have any questions?")
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"Source code not available for {selected_class}.")
                    
                    # Proceed to follow-up conversation or exit
                    while True:
                        follow_up = SpeechToTextTextToSpeechIO.parse_user_speech()
                        if follow_up.lower() in exit_words:
                            SpeechToTextTextToSpeechIO.speak_mainframe("Ending conversation.")
                            break
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe(f"Let's discuss the {selected_class} class.")
                            response = chat.send_message(f'You are a pair programmer. Read this code and the user will ask questions: {inspection_code}', stream=True)
                            if response:
                                for chunk in response:
                                    if hasattr(chunk, 'parts'):
                                        # Concatenate the text from each part
                                        full_text = ''.join(part.text for part in chunk.parts)
                                        SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                        print(full_text)
                                    else:
                                        # If it's a simple response, just speak and print the text
                                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                        print(chunk.text)
                                    time.sleep(0.1)
                            if not response:
                                attempt_count = 1  # Initialize re-try attempt count
                                while attempt_count < 5:
                                    response = chat.send_message(f'{user_input}', stream=True)
                                    attempt_count += 1  # Increment attempt count
                                    if response:
                                        for chunk in response:
                                            if hasattr(chunk, 'parts'):
                                                # Concatenate the text from each part
                                                full_text = ''.join(part.text for part in chunk.parts)
                                                SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                                print(full_text)
                                            else:
                                                # If it's a simple response, just speak and print the text
                                                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                                print(chunk.text)
                                            time.sleep(0.1)
                                    else:
                                        SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')
                                        
                    # break  # Exit after class processing


            # if query[0] in ['pair', 'pear', 'pare', 'payer', 'prayer', 'hair', 'tare', 'tear', 'air'] and query [1] == 'programmer':
            #     number_words = {
            #         'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
            #         'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            #     }
            #     inspection_code = {}
            #     SpeechToTextTextToSpeechIO.speak_mainframe('Reading the code.')
            #     diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__])
            #     print(f'DIAGNOSTIC SUMMARY: \n\n{diagnostic_summary}\n\n')
            #     class_names = list(diagnostic_summary.keys())
            #     # Number the classes and present them to the user
            #     class_list_text = "Here are the available classes: "
            #     for i, cls in enumerate(class_names, 1):
            #         class_list_text += f"{i}. {cls}. "
                
            #     SpeechToTextTextToSpeechIO.speak_mainframe(class_list_text)
            #     print(f"Available classes: {class_names}")
                
            #     # Let the user pick a class by number
            #     SpeechToTextTextToSpeechIO.speak_mainframe('Please choose a class by number.')
            #     while True:
            #         user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            #         if not user_input:
            #             continue
                    
            #         # Convert spoken number words to integers
            #         user_input = user_input.strip().lower()
            #         selected_number = number_words.get(user_input, user_input)  # Use the failsafe conversion

            #         try:
            #             selected_number = int(selected_number)  # Convert to int if it's a numeric string
            #             if selected_number < 1 or selected_number > len(class_names):
            #                 SpeechToTextTextToSpeechIO.speak_mainframe('Invalid number. Please choose a valid class number.')
            #                 continue
            #         except ValueError:
            #             SpeechToTextTextToSpeechIO.speak_mainframe('Please say a valid number.')
            #             continue
                    
            #         # Get the selected class by its number
            #         selected_class = class_names[selected_number - 1]
                    
            #         # Drill down into the selected class
            #         class_details = ChatBotTools.summarize_module(sys.modules[__name__], class_name=selected_class)
            #         method_list = list(class_details['classes'][selected_class]['methods'].keys())
                    
            #         # Add class source code to inspection_code
            #         inspection_code[selected_class] = class_details['classes'][selected_class]['source_code']
                    
            #         # Step 2: List methods of the selected class in a numbered list
            #         method_list_text = f"The {selected_class} class has the following methods: "
            #         for i, method in enumerate(method_list, 1):
            #             method_list_text += f"{i}. {method}. "
                    
            #         SpeechToTextTextToSpeechIO.speak_mainframe(method_list_text)
            #         print(f"Methods in {selected_class}: {method_list}")
                    
            #         # Let the user pick a method by number
            #         SpeechToTextTextToSpeechIO.speak_mainframe('Please choose a method by number.')
            #         while True:
            #             method_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            #             if not method_input:
            #                 continue
                        
            #             # Convert spoken number words to integers for methods as well
            #             method_input = method_input.strip().lower()
            #             method_number = number_words.get(method_input, method_input)

            #             try:
            #                 method_number = int(method_number)
            #                 if method_number < 1 or method_number > len(method_list):
            #                     SpeechToTextTextToSpeechIO.speak_mainframe('Invalid number. Please choose a valid method number.')
            #                     continue
            #             except ValueError:
            #                 SpeechToTextTextToSpeechIO.speak_mainframe('Please say a valid number.')
            #                 continue
                        
            #             # Get the selected method by its number
            #             selected_method = method_list[method_number - 1]
                        
            #             # Retrieve method details
            #             method_details = ChatBotTools.summarize_module(sys.modules[__name__], class_name=selected_class, method_name=selected_method)
                        
            #             # Add method source code to inspection_code
            #             inspection_code[selected_method] = method_details[selected_method]['source_code']
                        
            #             # Proceed to a new conversational loop
            #             SpeechToTextTextToSpeechIO.speak_mainframe("Code found. Do you have any questions about this method?")
            #             print(f"Details of {selected_method}: {method_details[selected_method]}")
                        
            #             # Start a follow-up loop for dynamic conversation
            #             while True:
            #                 follow_up = SpeechToTextTextToSpeechIO.parse_user_speech()
            #                 if follow_up.lower() in exit_words:
            #                     SpeechToTextTextToSpeechIO.speak_mainframe("Ending conversation.")
            #                     break
            #                 else:
            #                     # Provide answers based on the content of the inspection_code
            #                     SpeechToTextTextToSpeechIO.speak_mainframe(f"Let's discuss the {selected_method} method.")
            #                     response = chat.send_message(f'You are a pair programmer. Read this code and then the user will ask questions: {inspection_code}', stream=True)
            #                     if response:
            #                         for chunk in response:
            #                             if hasattr(chunk, 'parts'):
            #                                 # Concatenate the text from each part
            #                                 full_text = ''.join(part.text for part in chunk.parts)
            #                                 SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                                 print(full_text)
            #                             else:
            #                                 # If it's a simple response, just speak and print the text
            #                                 SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                                 print(chunk.text)
            #                             time.sleep(0.1)
            #                     if not response:
            #                         attempt_count = 1  # Initialize re-try attempt count
            #                         while attempt_count < 5:
            #                             response = chat.send_message(f'{user_input}', stream=True)
            #                             attempt_count += 1  # Increment attempt count
            #                             if response:
            #                                 for chunk in response:
            #                                     if hasattr(chunk, 'parts'):
            #                                         # Concatenate the text from each part
            #                                         full_text = ''.join(part.text for part in chunk.parts)
            #                                         SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                                         print(full_text)
            #                                     else:
            #                                         # If it's a simple response, just speak and print the text
            #                                         SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                                         print(chunk.text)
            #                                     time.sleep(0.1)
            #                             else:
            #                                 SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')
            
            
            #                     # Here you could extend to more dynamic interactions, such as:
            #                     # - Answer specific questions about the source code.
            #                     # - Provide additional details or explanations.
            #                     # - Offer code recommendations or refactoring advice.
            #                     continue
            #             break  # Exit after processing method
            #         break  # Exit after processing class







                # while True:
                #     user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                #     if not user_input:
                #         continue

                #     query = user_input.lower().split()
                #     if not query:
                #         continue

                #     if query[0] in exit_words:
                #         SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                #         break
                    
                #     if query[0] in ['all', 'any', 'full', 'module', 'everything']:
                #         diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__])
                #     else:
                #         diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__], detail_level='medium')
                        
            #         SpeechToTextTextToSpeechIO.speak_mainframe('Examining the code...')
            #         print(f'DIAGNOSTIC SUMMARY: \n\n{diagnostic_summary}\n\n')
            #         prompt = f'''### SYSTEM MESSAGE ### Gemini, Here is a summary of the current Python codebase for the app. 
            #         Once you've read the code we're going to discuss and plan modifications to the codebase.: 
            #         \n {diagnostic_summary}\n
            #         ### SYSTEM MESSAGE ### Gemini, you must read and understand all the nuances of this codebase. 
            #         you will read the code, and then you will say "I've read the code. What do you want to discuss first?", 
            #         and then you will await further instructions. 
            #         You are a trusted advisor for the user who owns the data above. 
            #         Your objective is to help the user meet goals, solve problems, and fulfill stated requirements. 
            #         you will help the user decide what to focus on to achieve their goals and requirements most effectively. 
            #         you will guide implementation of stated requirements beginning from the current state of things. 
            #         You will review all the information in context and you will think your task through step by step. 
            #         ## wait for user input after you acknowledge this message ##'''
            #         diagnostic_response = chat.send_message(f'{prompt}', stream=True)
            #         if diagnostic_response:
            #             for chunk in diagnostic_response:
            #                 if hasattr(chunk, 'parts'):
            #                     # Concatenate the text from each part
            #                     full_text = ''.join(part.text for part in chunk.parts)
            #                     SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                     print(full_text)
            #                 else:
            #                     # If it's a simple response, just speak and print the text
            #                     SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                     print(chunk.text)
            #                 time.sleep(0.1)
            #             time.sleep(1)
            #         break
            #     continue
                                            
            # else:
            #     response = chat.send_message(f'{user_input}', stream=True)
            #     if response:
            #         for chunk in response:
            #             if hasattr(chunk, 'parts'):
            #                 # Concatenate the text from each part
            #                 full_text = ''.join(part.text for part in chunk.parts)
            #                 SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                 print(full_text)
            #             else:
            #                 # If it's a simple response, just speak and print the text
            #                 SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                 print(chunk.text)
            #             time.sleep(0.1)
            #     if not response:
            #         attempt_count = 1  # Initialize re-try attempt count
            #         while attempt_count < 5:
            #             response = chat.send_message(f'{user_input}', stream=True)
            #             attempt_count += 1  # Increment attempt count
            #             if response:
            #                 for chunk in response:
            #                     if hasattr(chunk, 'parts'):
            #                         # Concatenate the text from each part
            #                         full_text = ''.join(part.text for part in chunk.parts)
            #                         SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                         print(full_text)
            #                     else:
            #                         # If it's a simple response, just speak and print the text
            #                         SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                         print(chunk.text)
            #                     time.sleep(0.1)
            #             else:
            #                 SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')









                
            # if query[0] in ['pair', 'pear', 'pare', 'payer', 'prayer', 'hair', 'tare', 'tear', 'air'] and query [1] == 'programmer':
            #     diagnostic_summary = ""
            #     SpeechToTextTextToSpeechIO.speak_mainframe('What level of detail?')
            #     while True:
            #         user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            #         if not user_input:
            #             continue

            #         query = user_input.lower().split()
            #         if not query:
            #             continue

            #         if query[0] in exit_words:
            #             SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
            #             break
                    
            #         if query[0] in ['high', 'hi', 'full']:
            #             diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__], detail_level='high')
            #         if query[0] in ['medium', 'mid', 'middle']:
            #             diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__], detail_level='medium')
            #         if query[0] in ['low', 'lo', 'little', 'small']:
            #             diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__], detail_level='low')
                        
            #         SpeechToTextTextToSpeechIO.speak_mainframe('Examining the code...')
            #         print(f'DIAGNOSTIC SUMMARY: \n\n{diagnostic_summary}\n\n')
            #         prompt = f'''### SYSTEM MESSAGE ### Gemini, Here is a summary of the current Python codebase for the app. 
            #         Once you've read the code we're going to discuss and plan modifications to the codebase.: 
            #         \n {diagnostic_summary}\n
            #         ### SYSTEM MESSAGE ### Gemini, you must read and understand all the nuances of this codebase. 
            #         you will read the code, and then you will say "I've read the code. What do you want to discuss first?", 
            #         and then you will await further instructions. 
            #         You are a trusted advisor for the user who owns the data above. 
            #         Your objective is to help the user meet goals, solve problems, and fulfill stated requirements. 
            #         you will help the user decide what to focus on to achieve their goals and requirements most effectively. 
            #         you will guide implementation of stated requirements beginning from the current state of things. 
            #         You will review all the information in context and you will think your task through step by step. 
            #         ## wait for user input after you acknowledge this message ##'''
            #         diagnostic_response = chat.send_message(f'{prompt}', stream=True)
            #         if diagnostic_response:
            #             for chunk in diagnostic_response:
            #                 if hasattr(chunk, 'parts'):
            #                     # Concatenate the text from each part
            #                     full_text = ''.join(part.text for part in chunk.parts)
            #                     SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                     print(full_text)
            #                 else:
            #                     # If it's a simple response, just speak and print the text
            #                     SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                     print(chunk.text)
            #                 time.sleep(0.1)
            #             time.sleep(1)
            #         break
            #     continue
                                            
            # else:
            #     response = chat.send_message(f'{user_input}', stream=True)
            #     if response:
            #         for chunk in response:
            #             if hasattr(chunk, 'parts'):
            #                 # Concatenate the text from each part
            #                 full_text = ''.join(part.text for part in chunk.parts)
            #                 SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                 print(full_text)
            #             else:
            #                 # If it's a simple response, just speak and print the text
            #                 SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                 print(chunk.text)
            #             time.sleep(0.1)
            #     if not response:
            #         attempt_count = 1  # Initialize re-try attempt count
            #         while attempt_count < 5:
            #             response = chat.send_message(f'{user_input}', stream=True)
            #             attempt_count += 1  # Increment attempt count
            #             if response:
            #                 for chunk in response:
            #                     if hasattr(chunk, 'parts'):
            #                         # Concatenate the text from each part
            #                         full_text = ''.join(part.text for part in chunk.parts)
            #                         SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
            #                         print(full_text)
            #                     else:
            #                         # If it's a simple response, just speak and print the text
            #                         SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
            #                         print(chunk.text)
            #                     time.sleep(0.1)
            #             else:
            #                 SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')