# standard imports
from datetime import datetime, timedelta
import inspect
import json
import os
import re
# import ssl
import sys
import subprocess
# import threading
import time
import traceback
import webbrowser
# third-party imports
from neo4j import GraphDatabase
# from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
# import certifi
# import flet as ft
import google.generativeai as genai
# from langchain_google_genai.llms import GoogleGenerativeAI
# from langchain_openai import ChatOpenAI
# from langchain.agents import tool
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
# from langchain.schema import HumanMessage, SystemMessage
# import numpy as np
import nltk
# nltk.download('punkt_tab')
import pandas as pd
import PIL.Image
import pyautogui
import pytz
import requests
# import speech_recognition as sr
# import tensorflow as tf
import wikipedia
import wolframalpha
# import yfinance as yf
# local imports
from chatbot.chatbot_speech import SpeechToTextTextToSpeechIO
from chatbot.chatbot_brain import ChatBotBrain
import chatbot.chatbot_global_state

from app_secrets.user_persona import (
    user_demographics, 
    user_skills_and_experience,
    user_interests, 
    user_favorite_quotes,
    )

# check if nltk punkt_tab is installed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
# ENVIRONMENT VARIABLES ###################################################################################################################################
from dotenv import load_dotenv
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
USER_PREFERRED_LANGUAGE = os.getenv('USER_PREFERRED_LANGUAGE', 'en')  # 2-letter lowercase
# USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Evan')
# USER_PREFERRED_NAME = os.getenv('USER_PREFERRED_NAME', 'User')  # Title case
# USER_SELECTED_HOME_CITY = os.getenv('USER_SELECTED_HOME_CITY', 'None')  # Title case
# USER_SELECTED_HOME_COUNTY = os.getenv('USER_SELECTED_HOME_COUNTY', 'None')  # Title case
# USER_SELECTED_HOME_STATE = os.getenv('USER_SELECTED_HOME_STATE', 'None')  # Title case
# USER_SELECTED_HOME_COUNTRY = os.getenv('USER_SELECTED_HOME_COUNTRY', 'None')  # 2-letter country code
USER_SELECTED_HOME_LAT = os.getenv('USER_SELECTED_HOME_LAT', 'None')  # Float with 6 decimal places
USER_SELECTED_HOME_LON = os.getenv('USER_SELECTED_HOME_LON', 'None')  # Float with 6 decimal places 
USER_SELECTED_TIMEZONE = os.getenv('USER_SELECTED_TIMEZONE', 'America/Chicago')  # Country/State format
USER_STOCK_WATCH_LIST = os.getenv('USER_STOCK_WATCH_LIST', 'None').split(',')  # Comma separated list of stock symbols
# USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
# PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
# PROJECT_CODE_DIRECTORY = os.getenv('PROJECT_CODE_DIRECTORY')
# PROJECT_TOOL_DIRECTORY = os.getenv('PROJECT_TOOL_DIRECTORY')
# DATABASES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_databases')
FILE_DROP_DIR_PATH = os.getenv('FILE_DROP_DIR_PATH', os.path.join(PROJECT_ROOT_DIRECTORY, 'app_generated_files'))
# LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_local_models')
# BASE_KNOWLEDGE_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_base_knowledge')
# SECRETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_secrets')
# SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_source_data')
# SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
# TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_tests')
# UTILITIES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'utilities')
# SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
folders_to_create = [FILE_DROP_DIR_PATH,]
for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)

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

# Set API keys and other sensitive information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
# openai_api_key=os.getenv('OPENAI_API_KEY')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
# google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
google_documentation_search_engine_id = os.getenv('GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID')
google_job_search_search_engine_id = os.getenv('GOOGLE_JOB_SEARCH_SEARCH_ENGINE_ID')
google_health_search_engine_id = os.getenv('GOOGLE_HEALTH_SEARCH_ENGINE_ID')
google_research_search_engine_id = os.getenv('GOOGLE_RESEARCH_SEARCH_ENGINE_ID')
google_restaurant_search_engine_id = os.getenv('GOOGLE_RESTAURANT_SEARCH_ENGINE_ID')
print('API keys and other sensitive information loaded from environment variables.\n\n')

# Establish the TTS bot's wake/activation word and script-specific global constants
# mic_on = True
# mic_on = False
# conversation_history = []
activation_word = os.getenv('ACTIVATION_WORD', 'robot')
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
# lemmmatizer = WordNetLemmatizer()
# intents = json.loads(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_intents.json').read())
# words = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_words.pkl', 'rb'))
# classes = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_classes.pkl', 'rb'))
# chatbot_model = tf.keras.models.load_model(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_model.keras')
unrecognized_inputs_file_path = f'{PROJECT_ROOT_DIRECTORY}/src/chatbot/chatbot_model_training/chatbot_unrecognized_message_intents.json'
print('Language models loaded.\n\n')

# CLASS DEFINITIONS ###################################################################################################################################
                        
class ChatBotTools:
    '''ChatBotTools contains all of the functions that are called by the chatbot_model, including larger llms, system commands, utilities, and api connections 
    to various services. it contains all of the methods that are called by the JSON intents in the chatbot_intents.json file in response to user input. '''
    data_store = {}
    
    def __init__(self):
        self.user_input = None  

    def set_user_input(self, input_text):
        '''Sets the user input for use in other methods.'''
        self.user_input = input_text
        
    # @staticmethod
    def gemini_chat(self):
        '''gemini_chat is a general purpose chat thread with the Gemini model, with optional branches for 
        running thorough diagnostics of the app codebase, calling Gemini as a pair programmer, and accessing data 
        stored in the data_store variable which is housed within the ChatBotTools class.'''
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Initializing the chat model...')
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
            # global mic_on
            if not SpeechToTextTextToSpeechIO.is_speaking and chatbot.chatbot_global_state.mic_on:
                self.user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not self.user_input:
                    print('No user input detected.')
                    continue

                query = self.user_input.lower().split()
                
                if not query:
                    print('No query was detected for parsing.')
                    continue

                if query[0] in exit_words:
                    print('Ending chat.')
                    SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                    break
                
                elif query[0] == 'stoic' and query[1] == 'lesson':
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
                    
                elif query[0] == 'access' and query [1] == 'data':
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

                elif query[0] in ['sous', 'sue', 'soo', 'su', 'tsu', 'sew', 'shoe', 'shoo'] and query [1] in ['chef', 'shef', 'chefs', 'shefs']:
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
                                                
                else:
                    response = chat.send_message(f'{self.user_input}', stream=True)
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
                            response = chat.send_message(f'{self.user_input}', stream=True)
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
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                                
    # @staticmethod
    # def alfred_chat():
    #     '''alfred_chat is a purpose built chat thread with the Gemini model, which focuses on multi action chains to help 
    #     the user work through career questions and form paths toward goals.'''
    #     chat = gemini_model.start_chat(history=[])
    #     SpeechToTextTextToSpeechIO.speak_alfred('Alfred has entered the chat. Calibrating')
    #     all_dicts = [
    #         user_demographics, 
    #         user_skills_and_experience,
    #         user_interests, 
    #         user_favorite_quotes,
    #         ]

    #     formatted_info = []
    #     for dictionary in all_dicts:
    #         # Check if the item is actually a dictionary
    #         if isinstance(dictionary, dict):
    #             formatted_dict = ", ".join([f"{k}={v}" for k, v in dictionary.items()])
    #             formatted_info.append(formatted_dict)
    #         else:
    #             print(f"Expected a dictionary, but found: {type(dictionary)}")

    #     formatted_user_info = " | ".join(formatted_info)
        
    #     print(f"\n\n##########\n##########\n##########\n\nUSER INFO:")
    #     print(f'User info: \n{formatted_user_info}\n')
        
    #     alfred_prompt = f"""
    #     ### <SYSTEM MESSAGE> <1/4> <START> ### 
    #     Gemini, you are being re-calibrated as a tailored personal assistant for the user. 
    #     You are about to recieve a host of data about your user along with instructions about your tasks and conduct. 
    #     The user data contains much of what you'll need to know to deeply understand the user. 
    #     In this data, there is a list of the user's main traits, goals, interests, hobbies, work experience, role models, etc. 
    #     You will emulate these traits and these people, and you will align yourself with their philosophies, characteristics, traits, and ways of thinking and acting. 
    #     You are learning about the user to be their ideal mentor and coach. 
    #     You are a trusted advisor. You must help the user find the best direction in all situations. 
    #     you will provide your insight and you will assist the user in finding answers with your knowledge and computational resources. 
    #     you must ensure that all of your advice is tailored to the user's persona. 
    #     you must be critical of the user, help them improve upon their weaknesses, identify their strengths, build upon their strengths, and challenge them with new ideas. 
    #     you must be very frank and matter of fact with the user. be like the user's listed role models. 
    #     you will help the user decide what to study and focus on and how to spend their time to meet their goals most effectively. 
    #     you must consider all factors and provide a holistic approach to the user's career path, including things that are not directly related to work but affect well-being and performance and health and cognition. 
    #     The ultimate goal is to help the user find the most effective ways to take swift action toward their goals. 
    #     We are here to combine theory and then take action. You will not talk about vague nebulous concepts. You will always respond with tangible, concrete, actionable advice. 
    #     All of your advice must also be followed by a recommended next best action. you must not be pedantic. 
    #     you will always think about the next best action and focus on this with the user. You must help keep things moving along. 
    #     Considering the user's profile: \n{formatted_user_info}\n what are your thoughts about this person? What would be the best actions for them to take to meet their goals? 
    #     you will think this through step by step. 
    #     you will keep it simple and concise. 
    #     You are a trusted advisor. 
    #     ### <SYSTEM MESSAGE> <1/4> <END> ###
    #     """
        
    #     print(f"\n\n##########\n##########\n##########\n\nALFRED PROMPT:")
    #     print(alfred_prompt)
        
    #     alfred_response = chat.send_message(f'{alfred_prompt}', stream=False)
    #     alfred_response.resolve()
        
    #     if not alfred_response:
    #         attempt_count = 1  # Initialize re-try attempt count
    #         while attempt_count < 5:
    #             alfred_response = chat.send_message(f'{alfred_prompt}', stream=False)
    #             attempt_count += 1  # Increment attempt count
    #             if alfred_response:
    #                 alfred_response.resolve()
    #             else:
    #                 print('Failed.')
        
    #     print(f"\n\n##########\n##########\n##########\n\nALFRED RESPONSE:")
    #     print(alfred_response)
        
    #     alfred_web_search_prompt = f"""
    #     ### <SYSTEM MESSAGE> <2/4> <START> ### 
    #     New information - you have access to tools. You will be using a search engine on behalf of the user. 
    #     You are an AI agent that can take actions on behalf of the user. you will help the user search for open job roles that align with their experience, requirements, and goals. 
    #     The next step in this assignment is for you to use your search tool - a search engine of popular job search websites. 
    #     you will review our messages to this point and ensure you are still on track and you are aligned with the requirements. 
    #     you must keep in mind the real-world circumstances of the user's current situation, and you will provide advice that takes this into consideration. 
    #     you must ensure you're recommending things that are attainable and realistic for the user. Ambitious is ok, but don't be unrealistic.  
    #     Considering the user's profile: \n{formatted_user_info}\n which job title is the most appropriate for the user based on their priorities and experience and goals and personality?  
    #     Your output for this step must be in the form of a job title - this will be the search phrase passed to the search engine. 
    #     your output must be formatted similarly to these examples: 
    #     python developer, technical program manager, platform developer, systems architect, systems engineer, business systems analyst, technical project manager, software engineer, etc. 
    #     Your output will be passed to the search engine and the results will be added to your memory so you can discuss them with the user. 
    #     You are being asked to search a search engine for jobs for the user. 
    #     You will be searching indeed, linkedin, monster, ziprecruiter, all at once with this custom search engine. 
    #     You will search for the most appropriate job title you can think of for the user. 
    #     The search phrase must be just a few words, no more. it must be a real job title. 
    #     DO NOT PROVIDE A LONG FORM RESPONSE. 
    #     DO NOT APPEND YOUR SEARCH PHRASE WITH ANY OTHER TEXT OR EXPLANADION OR DEFINITIONS. 
    #     PROVIDE YOUR JOB TITLE SEARCH PHRASE NOW. 
    #     ### <SYSTEM MESSAGE> <2/4> <END> ###
    #     """
        
    #     print(f"\n\n##########\n##########\n##########\n\nALFRED WEB SEARCH PROMPT:")
    #     print(alfred_web_search_prompt)
        
    #     alfred_web_search = chat.send_message(f'{alfred_web_search_prompt}', stream=False)
    #     alfred_web_search.resolve()
        
    #     search_phrase = ""
        
    #     if alfred_web_search:
    #         for chunk in alfred_web_search:
    #             if hasattr(chunk, 'parts'):
    #                 # Concatenate the text from each part
    #                 search_phrase += ''.join(part.text for part in chunk.parts)
    #             else:
    #                 # If it's a simple response, just concatenate the text
    #                 search_phrase += chunk.text
    #     if not alfred_web_search:
    #         attempt_count = 1  # Initialize re-try attempt count
    #         while attempt_count < 5:
    #             alfred_web_search = chat.send_message(f'{alfred_web_search_prompt}', stream=False)
    #             attempt_count += 1  # Increment attempt count
    #             if alfred_web_search:
    #                 for chunk in alfred_web_search:
    #                     if hasattr(chunk, 'parts'):
    #                         # Concatenate the text from each part
    #                         search_phrase += ''.join(part.text for part in chunk.parts)
    #                     else:
    #                         # If it's a simple response, just concatenate the text
    #                         search_phrase += chunk.text
    #             else:
    #                 print('ERROR.')
        
    #     print(f"\n\n##########\n##########\n##########\n\nALFRED WEB SEARCH RESPONSE:")
    #     print(search_phrase)
        
    #     search_url = f"https://www.googleapis.com/customsearch/v1?key={google_cloud_api_key}&cx={google_job_search_search_engine_id}&q={search_phrase}"

    #     response = requests.get(search_url)
    #     if response.status_code == 200:
    #         search_results = response.json().get('items', [])
    #         print(f"Search results: \n{search_results}\n")
    #         ChatBotTools.data_store['last_search'] = search_results
    #         print('Search results added to memory.')
    #     else:
    #         print('Search unsuccessful.')
            
    #     data_store = ChatBotTools.data_store
    #     print(ChatBotTools.data_store)
        
    #     alfred_web_search_review_prompt = f"""
    #     ### <SYSTEM MESSAGE> <3/4> <START> ### 
    #     You are receiving new information - you are now going to review the results of your search tool. You are an AI agent that is taking actions for the user. 
    #     The next step in this assignment is for you to review the results from your google programmable search for job titles. 
    #     You must review your conversation to this point, and ensure you are on the right track and you are aligned with the user requirements. 
    #     You must keep in mind the real-world circumstances of the user's current situation, and you will provide advice that takes this into consideration. 
    #     You will ensure you're recommending things that are attainable and realistic for the user. Ambitious is ok, but you must not be unrealistic.  
    #     you will provide your insight aftet you think it through step by step, and you will assist the user in finding answers by leveraging your knowledge and computational resources. 
    #     you must ensure that all of your advice is tailored to the user's persona. 
    #     you must be critical of the user, help them improve upon their weaknesses, identify their strengths, build upon their strengths, and challenge them with new ideas. 
    #     you will be very frank and matter of fact with the user. you iwll be like the listed personalities in the user's persona data. 
    #     you will help the user decide what to study and focus on and how to spend their time to meet these goals most effectively. 
    #     you will consider all factors and provide a holistic approach to the user's career path, including things that are not directly related to work but affect well-being and performance and health and cognition. 
    #     Considering the search results: \n{data_store}\n which job titles are most attainable and suitable for the user based on their priorities and experience?  Think thie through step by step and form an intentional point of view. 
    #     How should the user go about working toward these positions from where they currently are? You must provide simple, actionable, concrete, steps to implement this plan.
    #     You will think this through step by step. 
    #     ### <SYSTEM MESSAGE> <3/4> <END> ###
    #     """
        
    #     print(alfred_web_search_review_prompt)
        
    #     alfred_web_search_review = chat.send_message(f'{alfred_web_search_review_prompt}', stream=False)
    #     alfred_web_search_review.resolve()
        
    #     if not alfred_web_search_review:
    #         attempt_count = 1  # Initialize re-try attempt count
    #         while attempt_count < 5:
    #             alfred_web_search_review = chat.send_message(f'{alfred_web_search_review_prompt}', stream=False)
    #             attempt_count += 1  # Increment attempt count
    #             if alfred_web_search_review:
    #                 alfred_web_search_review.resolve()
    #             else:
    #                 print('Failed.')
                    
    #     print(alfred_web_search_review)

    #     alfred_prompt_2 = f""" 
    #     \n### USER DATA ### 
    #     ### <SYSTEM MESSAGE> <4/4> <START> ###
    #     *AI AGENT ROLE*
    #     You are a trusted advisor for the user who owns the data above. You will act as a trusted advidor for the user. 
    #     Your objective is to help the user meet their goals and solve the problems they present to you. 
    #     You will review the user persona information and you will think your task through step by step. 
    #     Draw insightful conclusions about the user - understand what they're interested in, how they think, what their aptitudes are, what direction they should take, etc.  
    #     # Read the user data: \n\n{formatted_user_info}\n\n
    #     *AI AGENT INSTRUCTIONS*
    #     Use your critical thinking skills to challenge and refine your own thought process - make your conclusions more accurate and more insightful. 
    #     you will help the user decide what to focus on to meet their goals most effectively. 
    #     you must provide advice on how the user can work toward their goals from their current position. 
    #     you will help the user learn where they exist within the current market environment, their hiring value in the current market, and how to improve their position in the market. 
    #     you must help the user identify their own potential biases or self-limiting thoughts and beliefs and help them work through them and call them out if you observe them when speaking to the user. 
    #     you must be a critical advisor to the user - do not accept what they say at face value. you must help them improve. 
    #     you will help recommend cool new things to the user. you will help the user learn new things. 
    #     you will act as a sounding board for the user and help them identify the things they can not see for themselves. 
    #     *AI AGENT CONDUCT*
    #     You must ensure that all of your output is well-calibrated and tailored to the user's persona and requirements. 
    #     you must be critical of the user, help them learn of new things, and challenge them with new ideas and concepts and interesting things to explore. 
    #     You must be very frank and matter of fact with the user. You must emulate the personalities and characteristics listed in the prompt requirements and users data. 
    #     You must NOT EMULATE BOTH SIDES OF THE CONVERSATION - you will only respond as the advisor - you are in a real-time verbal conversation with a human. 
    #     You must fact check yourself and you must make your statements very clear and simple. 
    #     You must reply concisely. Your output must sound like natural speech for the entirety of your communication with the user.  
    #     You must not generate long text. You will not write paragraphs. You will speak in sentences like humans do in conversation. You are in a conversation with a human.  
    #     You must not act stiff and robotic. You will maintain a natural conversational tone and flow throughout the conversation. 
    #     You must not ramble. You will not monologue. You will not generate long responses. 
    #     now, you will begin chatting with the user directly. you will prompt the user with thought provoking statements and questions. 
    #     you won't say too many things at once without user involvement. you won't ask too many questions at once without user involvement. 
    #     don't say too many things in a row without user involvement. don't ask too many questions in a row without user involvement. 
    #     you must optimize your text output to sound like speech when played in audio. punctuation and spacing is important. 
    #     for example: the text "AWS", when played in audio, sounds like "aaaahhhhzzzz", not "A. W. S. ". 
    #     you must "write" your text in a way that the user should "hear it" when played in audio - such as A W S or A.W.S. instead of AWS or aws. You will apply this same principle to all of your output. 
    #     the user is trying to decide which job they should pursue from their list of top choices. you will help them decide. 
    #     *AI AGENT WHAT TO DO AFTER YOU FORM YOUR PLAN*
    #     # YOU WILL THINK THIS THROUGH STEP BY STEP AND THEN PROVIDE YOUR REFINED SUMMARY OF YOUR INTRODUCTORY THOUGHTS TO THE USER AND THEN YOU WILL AWAIT THE USER'S REPLY TO BEGIN THE CONVERSATION DIALOGUE. 
    #     ### <SYSTEM MESSAGE> <4/4> <END> ###
    #     """
        
    #     print(alfred_prompt_2)
        
    #     alfred_response_2 = chat.send_message(f'{alfred_prompt_2}', stream=True)
        
    #     if alfred_response_2:
    #         for chunk in alfred_response_2:
    #             if hasattr(chunk, 'parts'):
    #                 # Concatenate the text from each part
    #                 full_text = ''.join(part.text for part in chunk.parts)
    #                 SpeechToTextTextToSpeechIO.speak_alfred(full_text)
    #                 print(full_text)
    #             else:
    #                 # If it's a simple response, just speak and print the text
    #                 SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
    #                 print(chunk.text)
    #             time.sleep(0.1)
    #         time.sleep(1)
        
    #     if not alfred_response_2:
    #         attempt_count = 1  # Initialize re-try attempt count
    #         while attempt_count < 5:
    #             alfred_response_2 = chat.send_message(f'{alfred_prompt_2}', stream=True)
    #             attempt_count += 1  # Increment attempt count
    #             if alfred_response_2:
    #                 for chunk in alfred_response_2:
    #                     if hasattr(chunk, 'parts'):
    #                         # Concatenate the text from each part
    #                         full_text = ''.join(part.text for part in chunk.parts)
    #                         SpeechToTextTextToSpeechIO.speak_alfred(full_text)
    #                         print(full_text)
    #                     else:
    #                         # If it's a simple response, just speak and print the text
    #                         SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
    #                         print(chunk.text)
    #                     time.sleep(0.1)
    #             else:
    #                 SpeechToTextTextToSpeechIO.speak_alfred('Chat failed.')
            
    #     while True:
    #         # global mic_on
    #         if not SpeechToTextTextToSpeechIO.is_speaking and chatbot.chatbot_global_state.mic_on:
    #             user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
    #             if not user_input:
    #                 continue

    #             query = user_input.lower().split()
    #             if not query:
    #                 continue

    #             if query[0] in exit_words:
    #                 SpeechToTextTextToSpeechIO.speak_alfred('Ending chat.')
    #                 break

    #             else:
    #                 response = chat.send_message(f'{user_input}', stream=True)
    #                 if response:
    #                     for chunk in response:
    #                         if hasattr(chunk, 'parts'):
    #                             # Concatenate the text from each part
    #                             full_text = ''.join(part.text for part in chunk.parts)
    #                             SpeechToTextTextToSpeechIO.speak_alfred(full_text)
    #                             print(full_text)
    #                         else:
    #                             # If it's a simple response, just speak and print the text
    #                             SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
    #                             print(chunk.text)
    #                         time.sleep(0.1)
    #                 if not response:
    #                     attempt_count = 1  # Initialize re-try attempt count
    #                     while attempt_count < 5:
    #                         response = chat.send_message(f'{user_input}', stream=True)
    #                         attempt_count += 1  # Increment attempt count
    #                         if response:
    #                             for chunk in response:
    #                                 if hasattr(chunk, 'parts'):
    #                                     # Concatenate the text from each part
    #                                     full_text = ''.join(part.text for part in chunk.parts)
    #                                     SpeechToTextTextToSpeechIO.speak_alfred(full_text)
    #                                     print(full_text)
    #                                 else:
    #                                     # If it's a simple response, just speak and print the text
    #                                     SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
    #                                     print(chunk.text)
    #                                 time.sleep(0.1)
    #                         else:
    #                             SpeechToTextTextToSpeechIO.speak_alfred('Chat failed.')

    # @staticmethod
    # def ideas_chat():
    #     '''ideas_chat is a brainstorming chat thread with the Gemini model, which focuses on multi action chains to help 
    #     the user work through questions and form implementation plans for goals.'''
    #     chat = gemini_model.start_chat(history=[])
    #     SpeechToTextTextToSpeechIO.speak_mainframe('Brainstorm has entered the chat. Calibrating')
    #     all_dicts = [
    #         user_demographics, 
    #         user_skills_and_experience,
    #         user_interests, 
    #         user_favorite_quotes,
    #         ]

    #     formatted_info = []
    #     for dictionary in all_dicts:
    #         # Check if the item is actually a dictionary
    #         if isinstance(dictionary, dict):
    #             formatted_dict = ", ".join([f"{k}={v}" for k, v in dictionary.items()])
    #             formatted_info.append(formatted_dict)
    #         else:
    #             print(f"Expected a dictionary, but found: {type(dictionary)}")

    #     formatted_user_info = " | ".join(formatted_info)
        
    #     print(f"\n\n##########\n##########\n##########\n\nUSER INFO:")
    #     print(f'User info: \n{formatted_user_info}\n')
        
    #     ideas_prompt = f""" 
    #     \n### USER PERSONA DATA ### 
    #     \n{formatted_user_info}\n\n 
    #     ### <SYSTEM MESSAGE> <1/1> <START> ###
    #     You are a trusted advisor for the user who owns the data above. You must act as a trusted advidor for the user. 
    #     The goal of this conversation is a brainstorming conversation with the user, to help the user work through questions and form implementation plans for goals. 
    #     Your objective is to help the user meet their goals or solve the problems they present to you. 
    #     You will review the user persona information and you will think your task through step by step. 
    #     Draw insightful conclusions about the user - understand what they're interested in, how they think, what advice they need, etc.  
    #     Use your critical thinking skills to challenge and refine your own thought process - make your conclusions more accurate and more insightful. 
    #     You must fact check yourself and you must make your statements very clear and simple. 
    #     You must reply concisely. Your output must sound like natural speech for the entirety of your communication with the user.  
    #     Do not generate long text. Do not write paragraphs. Speak in sentences like humans do in conversation. You are in a conversation with a human. 
    #     You must ensure that all of your output is calibrated and tailored to the user's persona and requirements. 
    #     you must be critical of the user, help them learn of new things, and challenge them with new ideas and concepts and interesting things to explore. 
    #     You must be very frank and matter of fact with the user. You must emulate the personalities and characteristics listed in the prompt requirements and users data. 
    #     Help recommend cool new things to the user. Help the user learn new things. 
    #     DO NOT EMULATE BOTH SIDES OF THE CONVERSATION - only respond as the advisor - you are in a real-time verbal conversation with a human.  
    #     You must not act stiff and robotic. You will maintain a natural conversational tone and flow throughout the conversation. 
    #     do not ramble. do not monologue. do not generate long responses. 
    #     act as a sounding board for the user and help them identify the things they can not see for themselves. 
    #     now you will begin chatting with the user directly. prompt the user with thought provoking statements and questions. 
    #     don't say too many things at once. don't ask too many questions at once. don't say too many things in a row. don't ask too many questions in a row. 
    #     THINK THIS THROUGH STEP BY STEP AND THEN PROVIDE YOUR REFINED INTRODUCTORY THOUGHTS TO THE USER AND THEN AWAIT THE USER'S REPLY TO BEGIN THE CONVERSATION DIALOGUE. 
    #     ### <SYSTEM MESSAGE> <1/1> <END> ### 
    #     """
        
    #     print(ideas_prompt)
        
    #     ideas_response = chat.send_message(f'{ideas_prompt}', stream=True)
        
    #     if ideas_response:
    #         for chunk in ideas_response:
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
    #         time.sleep(1)
    #     if not ideas_response:
    #         attempt_count = 1  # Initialize re-try attempt count
    #         while attempt_count < 5:
    #             ideas_response = chat.send_message(f'{ideas_prompt}', stream=True)
    #             attempt_count += 1  # Increment attempt count
    #             if ideas_response:
    #                 for chunk in ideas_response:
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
            
    #     while True:
    #         # global mic_on
    #         if not SpeechToTextTextToSpeechIO.is_speaking and chatbot.chatbot_global_state.mic_on:
    #             user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
    #             if not user_input:
    #                 continue

    #             query = user_input.lower().split()
    #             if not query:
    #                 continue

    #             if query[0] in exit_words:
    #                 SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
    #                 break

    #             else:
    #                 response = chat.send_message(f'{user_input}', stream=True)
    #                 if response:
    #                     for chunk in response:
    #                         if hasattr(chunk, 'parts'):
    #                             # Concatenate the text from each part
    #                             full_text = ''.join(part.text for part in chunk.parts)
    #                             SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
    #                             print(full_text)
    #                         else:
    #                             # If it's a simple response, just speak and print the text
    #                             SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
    #                             print(chunk.text)
    #                         time.sleep(0.1)
    #                 if not response:
    #                     attempt_count = 1  # Initialize re-try attempt count
    #                     while attempt_count < 5:
    #                         response = chat.send_message(f'{user_input}', stream=True)
    #                         attempt_count += 1  # Increment attempt count
    #                         if response:
    #                             for chunk in response:
    #                                 if hasattr(chunk, 'parts'):
    #                                     # Concatenate the text from each part
    #                                     full_text = ''.join(part.text for part in chunk.parts)
    #                                     SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
    #                                     print(full_text)
    #                                 else:
    #                                     # If it's a simple response, just speak and print the text
    #                                     SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
    #                                     print(chunk.text)
    #                                 time.sleep(0.1)
    #                         else:
    #                             SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def run_greeting_code(self):
        '''This is a placeholder test function that will be called by the chatbot when the user says hello'''
        print('### TEST ### You said:', self.user_input)
     
    def generate_json_intent(self):
        '''generate_json_intent is called by the chatbot when the user input is not recognized. it "works" but the content is not very intelligent yet.'''
        print("UNRECOGNIZED INPUT: writing new intent to chatbot_unrecognized_message_intents.json")
        json_gen_prompt = '''# System Message Start # - Gemini, ONLY GENERATE ONE SHORT SENTENCE FOR EACH PROMPT ACCORDING TO THE USER INSTRUCTIONS. KEEP EACH SENTENCE TO UNDER 10 WORDS, IDEALLY CLOSER TO 5. - # System Message End #'''
        # Generate an initial response using Gemini
        initial_reply = gemini_model.generate_content(f"{json_gen_prompt}. // Please provide a response to: {self.user_input}")
        initial_reply.resolve()
        bot_reply = initial_reply.text
        json_function_name = re.sub(r'\W+', '', self.user_input).lower() + '_function'
        new_intent = {
            "tag": f"unrecognized_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patterns": [self.user_input],
            "responses": [bot_reply],
            "action": json_function_name
        }

        print(f"\nAttempting to write to:\n", unrecognized_inputs_file_path)
        print(f"\nNew Intent:\n", new_intent)

        try:
            with open(unrecognized_inputs_file_path, 'r+') as file:
                data = json.load(file)
                data["intents"].append(new_intent)
                file.seek(0)
                json.dump(data, file, indent=4)
                print('New intent written to chatbot_unrecognized_message_intents.json')
        except FileNotFoundError:
            try:
                with open(unrecognized_inputs_file_path, 'w') as file:
                    json.dump({"intents": [new_intent]}, file, indent=4)
                    print('New file created and intent written to chatbot_unrecognized_message_intents.json')
            except Exception as e:
                print(f"Error creating new file: {e}")
        except Exception as e:
            print(f"Error updating existing file: {e}")

        print('Intent update attempted. Check the file for changes.')

    @classmethod
    def control_mouse(cls):
        '''control_mouse is a simple mouse control function that allows the user to control the mouse with their voice by 
        saying "{activation_word}, mouse control" or "{activation_word}, control the mouse". this will activate the mouse control 
        which the user can trigger by saying "mouse click" or "mouse up 200" (pixels), etc.'''
        SpeechToTextTextToSpeechIO.speak_mainframe('Mouse control activated.')
        direction_map = {
            'north': (0, -1),
            'south': (0, 1),
            'west': (-1, 0),
            'east': (1, 0),
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue
            query = user_input.lower().split()
            if not query:
                continue
            if len(query) > 0 and query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Exiting mouse control.')
                break
            if query[0] == 'click':
                pyautogui.click()
            elif query[0] in ['move', 'mouse', 'go'] and len(query) > 2 and query[1] in direction_map and query[2].isdigit():
                move_distance = int(query[2])  # Convert to integer
                direction_vector = direction_map[query[1]]
                pyautogui.move(direction_vector[0] * move_distance, direction_vector[1] * move_distance, duration=0.1)

    @staticmethod
    def take_screenshot():
        '''takes a screenshot of the current screen, saves it to the file drop folder, and asks the user if they want a summary of the image. 
        the summary is spoken and also saved as a .txt file alongside the screenshot.'''
        today = datetime.today().strftime('%Y%m%d %H%M%S')       
        file_name = f'{FILE_DROP_DIR_PATH}/screenshot_{today}.png'
        subprocess.call(['screencapture', 'screenshot.png'])
        # Save the screenshot to the file drop folder
        subprocess.call(['mv', 'screenshot.png', file_name])
        SpeechToTextTextToSpeechIO.speak_mainframe('Saved. Do you want a summary?')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            if query[0] in ['no', 'nope', 'nah', 'not', 'not yet']:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            if query[0] in ['yeah', 'yes', 'yep', 'sure', 'ok']:
                img = PIL.Image.open(file_name)
                response = gemini_vision_model.generate_content(["### SYSTEM MESSAGE ### Gemini, you are a computer vision photo-to-text parser. DO NOT HALLUCINATE ANY FALSE FACTS. Create a succinct but incredibly descriptive and informative summary of all important details in this image (fact check yourself before you finalize your response) (fact check yourself before you finalize your response) (fact check yourself before you finalize your response):", img])
                response.resolve()
                response_1 = response.text
                print(f'RESPONSE 1 \n\n {response_1}\n')
                # Convert the content of the response to a .txt file and save it
                with open(f'{FILE_DROP_DIR_PATH}/screenshot_{today}_description.txt', 'w') as f:
                    f.write(response_1)
                SpeechToTextTextToSpeechIO.speak_mainframe(f'{response_1}')

    @staticmethod
    def google_search():
        SpeechToTextTextToSpeechIO.speak_mainframe('What do you want to search?')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue
            query = user_input.lower().split()
            if not query:
                continue
            if len(query) > 0 and query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Exiting mouse control.')
                break
            else:
                url = f'https://www.google.com/search?q={user_input}'
                webbrowser.open(url, new=1)
                break
        
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

    @staticmethod
    def summarize_module(module, class_name=None, method_name=None):
        '''Summarize the classes and methods in a module. Allows drilling down to specific class or method level.'''
        print('### BEGINNING SUMMARIZE MODULE ###')
        summary = {'classes': {}}
        
        # Get all classes in the module that are defined in the current file
        classes = inspect.getmembers(module, inspect.isclass)
        
        if not class_name:
            # Return list of all classes
            return {cls_name: {'docstring': inspect.getdoc(cls_obj)} 
                    for cls_name, cls_obj in classes if cls_obj.__module__ == module.__name__}
        
        # If class_name is provided, get selected class details
        selected_class = next((cls_obj for cls_name, cls_obj in classes 
                            if cls_name == class_name and cls_obj.__module__ == module.__name__), None)
        
        if not selected_class:
            return f"Class '{class_name}' not found."
        
        cls_summary = {
            'docstring': inspect.getdoc(selected_class),
            'methods': {}
        }
        
        # Get methods for the class (only from the current file)
        methods = inspect.getmembers(selected_class, inspect.isfunction)
        cls_summary['methods'] = {method_name_: {'docstring': inspect.getdoc(method_obj), 
                                                'source_code': inspect.getsource(method_obj)} 
                                for method_name_, method_obj in methods 
                                if method_obj.__module__ == module.__name__}
        
        summary['classes'][class_name] = cls_summary
        
        if method_name:
            # Get details for a specific method
            selected_method = cls_summary['methods'].get(method_name)
            if not selected_method:
                return f"Method '{method_name}' not found in class '{class_name}'."
            return {method_name: selected_method}
        
        print('### ENDING SUMMARIZE MODULE ###')
        print(f"Summary: \n{summary}")
        return summary

    # @staticmethod
    # def summarize_module(module, class_name=None, method_name=None):
    #     '''Summarize the classes and methods in a module. Allows drilling down to specific class or method level.'''
    #     summary = {'classes': {}}
        
    #     # Get all classes in the module
    #     classes = inspect.getmembers(module, inspect.isclass)
    #     if not class_name:
    #         # Return list of all classes
    #         return {cls_name: {'docstring': inspect.getdoc(cls_obj)} for cls_name, cls_obj in classes if cls_obj.__module__ == module.__name__}
        
    #     # If class_name is provided, get selected class details
    #     selected_class = next((cls_obj for cls_name, cls_obj in classes if cls_name == class_name), None)
    #     if not selected_class:
    #         return f"Class '{class_name}' not found."
        
    #     cls_summary = {
    #         'docstring': inspect.getdoc(selected_class),
    #         'methods': {}
    #     }
        
    #     # Get methods for the class
    #     methods = inspect.getmembers(selected_class, inspect.isfunction)
    #     cls_summary['methods'] = {method_name_: {'docstring': inspect.getdoc(method_obj), 'source_code': inspect.getsource(method_obj)} for method_name_, method_obj in methods}
        
    #     summary['classes'][class_name] = cls_summary
        
    #     if method_name:
    #         # Get details for a specific method
    #         selected_method = cls_summary['methods'].get(method_name)
    #         if not selected_method:
    #             return f"Method '{method_name}' not found in class '{class_name}'."
    #         return {method_name: selected_method}
        
    #     return summary


    # @staticmethod
    # def summarize_module(module, class_name=None, method_name=None):
    #     '''Summarize the classes and methods in a module. Allows drilling down to specific class or method level.'''
    #     summary = {'classes': {}, 'functions': {}}
        
    #     # Get all classes in the module
    #     classes = inspect.getmembers(module, inspect.isclass)
    #     if not class_name:
    #         # If no class is selected, return list of all classes
    #         return {cls_name: {'docstring': inspect.getdoc(cls_obj)} for cls_name, cls_obj in classes if cls_obj.__module__ == module.__name__}
        
    #     # Get selected class details
    #     selected_class = next((cls_obj for cls_name, cls_obj in classes if cls_name == class_name), None)
    #     if not selected_class:
    #         return f"Class '{class_name}' not found."
        
    #     cls_summary = {
    #         'docstring': inspect.getdoc(selected_class),
    #         'methods': {},
    #         'class_methods': {},
    #         'static_methods': {}
    #     }
        
    #     if not method_name:
    #         # If no method selected, return list of methods in the class
    #         methods = inspect.getmembers(selected_class, inspect.isfunction)
    #         cls_summary['methods'] = {method_name: {'docstring': inspect.getdoc(method_obj)} for method_name, method_obj in methods}
    #         summary['classes'][class_name] = cls_summary
    #         return summary
        
    #     # Get selected method details
    #     methods = inspect.getmembers(selected_class, inspect.isfunction)
    #     selected_method = next((method_obj for method_name_, method_obj in methods if method_name_ == method_name), None)
    #     if not selected_method:
    #         return f"Method '{method_name}' not found in class '{class_name}'."
        
    #     method_summary = {
    #         'docstring': inspect.getdoc(selected_method),
    #         'source_code': inspect.getsource(selected_method)
    #     }
        
    #     # Return method details if specified
    #     return {method_name: method_summary}
                                            
    # @staticmethod
    # def summarize_module(module, detail_level='high'):
    #     summary = {'classes': {}, 'functions': {}}
    #     '''summarize_module returns a summary of the classes and functions in a module. this is used by the developer for debugging and analysis and also 
    #     passed to the LLM for pair programming and app codebase diagnostics'''
    #     # Get all classes in the module
    #     classes = inspect.getmembers(module, inspect.isclass)
    #     for cls_name, cls_obj in classes:
    #         if cls_obj.__module__ == module.__name__:
    #             cls_summary = {
    #                 'docstring': inspect.getdoc(cls_obj),
    #                 'methods': {},
    #                 'class_methods': {},
    #                 'static_methods': {},
    #                 'source_code': inspect.getsource(cls_obj)
    #             }

    #             # Get all methods of the class
    #             methods = inspect.getmembers(cls_obj, inspect.isfunction)
    #             for method_name, method_obj in methods:
    #                 cls_summary['methods'][method_name] = {
    #                     'docstring': inspect.getdoc(method_obj),
    #                     'source_code': inspect.getsource(method_obj)
    #                 }

    #             # Get class methods and static methods
    #             for name, obj in cls_obj.__dict__.items():
    #                 if isinstance(obj, staticmethod):
    #                     cls_summary['static_methods'][name] = {
    #                         'docstring': inspect.getdoc(obj),
    #                         'source_code': inspect.getsource(obj.__func__)
    #                     }
    #                 elif isinstance(obj, classmethod):
    #                     cls_summary['class_methods'][name] = {
    #                         'docstring': inspect.getdoc(obj),
    #                         'source_code': inspect.getsource(obj.__func__)
    #                     }

    #             summary['classes'][cls_name] = cls_summary

    #     # Get all functions in the module
    #     functions = inspect.getmembers(module, inspect.isfunction)
    #     for func_name, func_obj in functions:
    #         if func_obj.__module__ == module.__name__:
    #             summary['functions'][func_name] = {
    #                 'docstring': inspect.getdoc(func_obj),
    #                 'source_code': inspect.getsource(func_obj)
    #             }

    #     # Adjust detail level
    #     if detail_level == 'medium':
    #         for cls_name, cls_details in summary['classes'].items():
    #             for method_type in ['methods', 'class_methods', 'static_methods']:
    #                 for method_name, method_details in cls_details[method_type].items():
    #                     method_details.pop('source_code', None)
    #         for func_name, func_details in summary['functions'].items():
    #             func_details.pop('source_code', None)
                
    #     elif detail_level == 'low':
    #         for cls_name, cls_details in summary['classes'].items():
    #             cls_summary = {'docstring': cls_details['docstring']}
    #             summary['classes'][cls_name] = cls_summary
    #         for func_name, func_details in summary['functions'].items():
    #             func_summary = {'docstring': func_details['docstring']}
    #             summary['functions'][func_name] = func_summary

    #     return summary
    
    @staticmethod
    def translate_speech():
        '''Translats a spoken phrase from user's preferred language to another language by saying 
        "{activation_word}, translate" or "{activation_word}, help me translate".'''
        language_code_mapping = {
            "en": ["english", "Daniel"],
            "es": ["spanish", "Paulina"],
            "fr": ["french", "Amélie"],
            "de": ["german", "Anna"],
            "it": ["italian", "Alice"],
            "ru": ["russian", "Milena"],
            "ja": ["japanese", "Kyoko"],
        }
        language_names = [info[0].lower() for info in language_code_mapping.values()]
        SpeechToTextTextToSpeechIO.speak_mainframe('What language do you want to translate to?')
        time.sleep(2)
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue
        
            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Canceling translation.')
                break
        
            # translate
            if query[0] in language_names:
                target_language_name = query[0]
                SpeechToTextTextToSpeechIO.speak_mainframe(f'Speak the phrase you want to translate.')
                time.sleep(2)
                phrase_to_translate = SpeechToTextTextToSpeechIO.parse_user_speech().lower()

                source_language = USER_PREFERRED_LANGUAGE  # From .env file
                target_voice = None

                # Find the language code and voice that matches the target language name
                target_language_code, target_voice = None, None
                for code, info in language_code_mapping.items():
                    if target_language_name.lower() == info[0].lower():
                        target_language_code = code
                        target_voice = info[1]
                        break

                if not target_language_code:
                    SpeechToTextTextToSpeechIO.speak_mainframe(f"Unsupported language: {target_language_name}")
                    return

                model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language_code}'
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                batch = tokenizer([phrase_to_translate], return_tensors="pt", padding=True)
                translated = model.generate(**batch)
                translation = tokenizer.batch_decode(translated, skip_special_tokens=True)
                print(f'In {target_language_name}, it\'s: {translation}')    
                SpeechToTextTextToSpeechIO.speak_mainframe(f'In {target_language_name}, it\'s: {translation}', voice=target_voice)
                continue

    @staticmethod
    def wiki_summary():
        '''wiki_summary returns a summary of a wikipedia page based on user input.'''
        SpeechToTextTextToSpeechIO.speak_mainframe('What should we summarize from Wikipedia?')

        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Canceling search.')
                break

            print("Wikipedia Query:", user_input)
            SpeechToTextTextToSpeechIO.speak_mainframe(f'Searching {user_input}')

            try:
                search_results = wikipedia.search(user_input)
                if not search_results:
                    print('No results found.')
                    continue

                wiki_page = wikipedia.page(search_results[0])
                wiki_title = wiki_page.title
                wiki_summary = wiki_page.summary

                response = f'Page title: \n{wiki_title}\n, ... Page Summary: \n{wiki_summary}\n'
                # Storing Wikipedia summary in the data store
                ChatBotTools.data_store['wikipedia_summary'] = {
                    'query': user_input,
                    'title': wiki_title,
                    'summary': wiki_summary,
                    'full_page': str(wiki_page)
                }
                print(response)
                SpeechToTextTextToSpeechIO.speak_mainframe(f"{user_input} summary added to global data store.")
                SpeechToTextTextToSpeechIO.speak_mainframe("Would you like to hear the summary now?")
                while True:
                    user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                    if not user_input:
                        continue

                    query = user_input.lower().split()
                    if not query:
                        continue

                    if query[0] in exit_words:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Canceling search.')
                        break
                    
                    if query[0] in ['yes', 'yeah', 'ok', 'sure', 'yep']:
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"{wiki_summary}")
                        break
                    
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Ok.')
                        break
                
                break

            except wikipedia.DisambiguationError as e:
                try:
                    # Attempt to resolve disambiguation by selecting the first option
                    wiki_page = wikipedia.page(e.options[0])
                    continue
                except Exception as e:
                    print(f"Error resolving disambiguation: {e}")
                    break

            except wikipedia.PageError:
                print("Page not found. Please try another query.")
                SpeechToTextTextToSpeechIO.speak_mainframe("Error: Page not found.")
                continue

            except wikipedia.RequestsException:
                print("Network error. Please check your connection.")
                SpeechToTextTextToSpeechIO.speak_mainframe("Error: No network connection.")
                break

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                SpeechToTextTextToSpeechIO.speak_mainframe(f"An error occured. Message: {e}")
                break
        
    @staticmethod
    def custom_search_engine():
        global google_documentation_search_engine_id
        global google_job_search_search_engine_id
        global google_health_search_engine_id
        global google_research_search_engine_id
        global google_restaurant_search_engine_id
        SpeechToTextTextToSpeechIO.speak_mainframe("Which engine do you want to use?")
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            engine = ' '.join(query).lower()
            
            if engine in ['coding', 'programming', 'python', 'documentation', 'document', 'docs', 'documentation search', 'document search', 'docs search']:
                google_search_engine_id = google_documentation_search_engine_id
            
            if engine in ['job search', 'job', 'jobs', 'career']:
                google_search_engine_id = google_job_search_search_engine_id
                
            if engine in ['health', 'health search', 'healthcare', 'healthcare search', 'supplements', 'nutrition', 'fitness', 'wellness']:
                google_search_engine_id = google_health_search_engine_id
                
            if engine in ['research', 'work', 'design', 'scientific', 'science', 'work work']:
                google_search_engine_id = google_research_search_engine_id
                
            if engine in ['restaurant', 'restaurants', 'food', 'dining', 'dine', 'eat', 'eating', 'restaurant search']:
                google_search_engine_id = google_restaurant_search_engine_id
        
            SpeechToTextTextToSpeechIO.speak_mainframe("Speak your search query.")
            time.sleep(2)
            
            while True:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue

                query = user_input.lower().split()
                if not query:
                    continue

                if query[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                    break
                
                search_query = ' '.join(query)
            
                search_url = f"https://www.googleapis.com/customsearch/v1?key={google_cloud_api_key}&cx={google_search_engine_id}&q={search_query}"

                response = requests.get(search_url)
                if response.status_code == 200:
                    search_results = response.json().get('items', [])
                    print(f"Search results: \n{search_results}\n")
                    ChatBotTools.data_store['last_search'] = search_results
                    SpeechToTextTextToSpeechIO.speak_mainframe('Search results added to memory.')
                    return search_results
                else:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Search unsuccessful.')
                    print(f"Error: {response.status_code}")
                    return f"Error: {response.status_code}"
            
    @staticmethod
    def play_youtube_video():
        '''accepts spoken user_input and parses it into a youtube video id, then launches the video in the default browser'''
        SpeechToTextTextToSpeechIO.speak_mainframe('What would you like to search on YouTube?.')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech().lower()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending youtube session.')
                break
            
            search_query = ' '.join(query)
            print("YouTube Query:", search_query)
            url = f'https://www.youtube.com/results?search_query={search_query}'
            webbrowser.open(url)
            break

    @staticmethod
    def wolfram_alpha():
        '''wolfram_alpha returns a summary of a wolfram alpha query based on user input'''
        wolfram_client = wolframalpha.Client(wolfram_app_id)
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Initializing wolfram alpha. State your query.')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending session.')
                break
            
            SpeechToTextTextToSpeechIO.speak_mainframe(f'Heard.')
            try:
                response = wolfram_client.query(user_input)
                print(f"Response from Wolfram Alpha: {response}")

                # Check if the query was successfully interpreted
                if not response['@success']:
                    suggestions = response.get('didyoumeans', {}).get('didyoumean', [])
                    if suggestions:
                        # Handle multiple suggestions
                        if isinstance(suggestions, list):
                            suggestion_texts = [suggestion['#text'] for suggestion in suggestions]
                        else:
                            suggestion_texts = [suggestions['#text']]

                        suggestion_message = " or ".join(suggestion_texts)
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"Sorry, I couldn't interpret that query. These are the alternate suggestions: {suggestion_message}.")
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Sorry, I couldn\'t interpret that query. Please try rephrasing it.')

                    return 'Query failed.'

                # Filtering and storing all available pods
                wolfram_data = []
                for pod in response.pods:
                    pod_data = {'title': pod.title}
                    if hasattr(pod, 'text') and pod.text:
                        pod_data['text'] = pod.text
                    wolfram_data.append(pod_data)

                # # Adding to data store
                # ChatBotTools.data_store['wolfram_alpha_response'] = {
                #     'query': user_input,
                #     'pods': wolfram_data
                # }  
                # SpeechToTextTextToSpeechIO.speak_mainframe('Search complete. Data saved to memory.') 

                # Initialize the data store dictionary if it doesn't exist
                if 'wolfram_alpha_responses' not in ChatBotTools.data_store:
                    ChatBotTools.data_store['wolfram_alpha_responses'] = {}

                # Generate a unique key for the current query
                # For example, using a timestamp or an incrementing index
                unique_key = f"query_{len(ChatBotTools.data_store['wolfram_alpha_responses']) + 1}"

                # Store the response using the unique key
                ChatBotTools.data_store['wolfram_alpha_responses'][unique_key] = {
                    'query': user_input,
                    'pods': wolfram_data
                }

            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"An error occurred: {e}\nDetails: {error_traceback}")
                SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while processing the query. Please check the logs for more details.')
                return f"An error occurred: {e}\nDetails: {error_traceback}"
            
            SpeechToTextTextToSpeechIO.speak_mainframe('Would you like to run another query?') 
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue
            query = user_input.lower().split()
            if not query or query[0] in exit_words or query[0] in ['no', 'nope', 'nah', 'not', 'not yet', 'cancel', 'exit', 'quit']:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending session.')
                break
            
    @staticmethod
    def get_weather_forecast():
        '''get_weather_forecast gets a spoken weather forecast from openweathermap for the next 4 days by day part based on user defined home location'''
        appid = f'{open_weather_api_key}'

        # Fetching coordinates from environment variables
        lat = USER_SELECTED_HOME_LAT
        lon = USER_SELECTED_HOME_LON

        # OpenWeatherMap API endpoint for 4-day hourly forecast
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={appid}"

        response = requests.get(url)
        print("Response status:", response.status_code)
        if response.status_code != 200:
            return "Failed to retrieve weather data."

        data = response.json()
        print("Data received:", data)

        # Process forecast data
        forecast = ""
        timezone = pytz.timezone(USER_SELECTED_TIMEZONE)
        now = datetime.now(timezone)
        periods = [(now + timedelta(days=i)).replace(hour=h, minute=0, second=0, microsecond=0) for i in range(4) for h in [6, 12, 18, 0]]

        for i in range(0, len(periods), 4):
            day_forecasts = []
            for j in range(4):
                start, end = periods[i + j], periods[i + j + 1] if j < 3 else periods[i] + timedelta(days=1)
                period_forecast = [f for f in data['list'] if start <= datetime.fromtimestamp(f['dt'], tz=timezone) < end]
                
                if period_forecast:
                    avg_temp_kelvin = sum(f['main']['temp'] for f in period_forecast) / len(period_forecast)
                    avg_temp_fahrenheit = (avg_temp_kelvin - 273.15) * 9/5 + 32  # Convert from Kelvin to Fahrenheit
                    descriptions = set(f['weather'][0]['description'] for f in period_forecast)
                    time_label = ["morning", "afternoon", "evening", "night"][j]
                    day_forecasts.append(f"{time_label}: average temperature {avg_temp_fahrenheit:.1f}°F, conditions: {', '.join(descriptions)}")

            if day_forecasts:
                forecast_date = periods[i].strftime('%Y-%m-%d')
                # Convert forecast_date to weekday format aka "Monday", etc.
                forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').strftime('%A')
                forecast += f"\n{forecast_date}: {'; '.join(day_forecasts)}."

                # print("Weather forecast:", forecast)
                # SpeechToTextTextToSpeechIO.speak_mainframe(f'Weather forecast for {USER_SELECTED_HOME_CITY}, {USER_SELECTED_HOME_STATE}: {forecast}')
                weather_forecast = f'Weather forecast for next 4 days, broken out by 6 hour day part: {forecast}'
                
            else:
                print("No weather forecast data available.")
                
        if weather_forecast:
            response = gemini_model.generate_content(f"""### SYSTEM MESSAGE START ### 
                                                     You are a weather report summarizer. 
                                                     Your output must be short and concise. Limit your response to just a few sentences. 
                                                     The report below is weather for the next 4 days, by 6 hour day part, and it's too verbose to be practical. 
                                                     Provide a concise (short) summary of this weather forecast with recommendations for how the user should navigate 
                                                     this weather on each day. Limit your reply to just 1-2 sentences per day.  
                                                     Be concise. Here is the report to summarize: {weather_forecast}
                                                     ### SYSTEM MESSAGE END ###""", stream=True)
            if response:
                response.resolve()
                print(f"Response from Gemini: {response.text}")
                SpeechToTextTextToSpeechIO.speak_mainframe(f'{response.text}')

    # @staticmethod
    # def get_stock_report():
    #     stock_reports = StockReports(USER_STOCK_WATCH_LIST)
    #     discounts_update = stock_reports.find_discounted_stocks()
    #     if discounts_update:
    #         ChatBotTools.data_store['discounted_stocks'] = discounts_update
    #         print(f'Discounted stocks: \n{discounts_update}\n')
    #         SpeechToTextTextToSpeechIO.speak_mainframe(f'Discounted stocks loaded to memory.')
    #     else:
    #         SpeechToTextTextToSpeechIO.speak_mainframe(f'No discounted stocks found.')
    #     recs_update = stock_reports.find_stock_recommendations()
    #     if recs_update:
    #         ChatBotTools.data_store['recommended_stocks'] = recs_update
    #         print(f'Recommended stocks: \n{recs_update}\n')
    #         SpeechToTextTextToSpeechIO.speak_mainframe(f'Recommended stocks loaded to memory.')
    #     else:
    #         SpeechToTextTextToSpeechIO.speak_mainframe(f'No recommended stocks found.')

    # @staticmethod
    # def agent_one():
    #     llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_gemini_api_key)
    #     prompt = "Repeat all of the above"
    #     response = llm(prompt)
    #     print("Generated Response:", response)
        
    # @staticmethod
    # def agent_two():
    #     chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    #     messages = [
    #         SystemMessage(
    #             content="You are a helpful assistant that helps the user solve problems. You are a high tech AI assistant similar to Jarvis or Cortana. You are not being asked to copy, just to emulate the general characteristics. Be smart and innovative. Help the user discover new ideas they may not have been thinking of. Help the user grow, learn, develop, and hone new skills."
    #         ),
    #         HumanMessage(
    #             content=f"Hi! I'm excited to start chatting with you today. Be aware that you are speaking verbally in a TTS / STT app. Make your responses concise and easy to understand. Make your output approapriate for conversational flow. Make sure to speak naturally and not monologue. Confirm you understand this with a brief confirmation phrase, then we'll begin chatting."
    #         ),
    #     ]
    #     result = chat(messages)
    #     print("Generated Response:", result)
    #     SpeechToTextTextToSpeechIO.speak_mainframe(f'{result}')
        
    #     while True:
    #         global mic_on
    #         if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
    #             user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
    #             if not user_input:
    #                 continue

    #             prompt = [
    #                 SystemMessage(
    #                     content="""You are a helpful assistant that helps the user solve problems. 
    #                     Respond to the user's input in the best way you can. 
    #                     Please be aware that you are speaking verbally in a TTS / STT app. 
    #                     Make sure your responses are concise and easy to understand. 
    #                     Make sure your output has a natural conversational flow. 
    #                     You are a high tech AI assistant similar to Jarvis or Cortana. 
    #                     You are not being asked to copy, just to emulate the general characteristics. 
    #                     Be smart and innovative. 
    #                     Help the user discover new ideas they may not have been thinking of. 
    #                     Help the user grow, learn, develop, and hone new skills. 
    #                     Think each problem through step by step before you act. 
    #                     """
    #                 ),
    #                 HumanMessage(
    #                     content=user_input
    #                 ),
    #             ]
                
    #             query = user_input.lower().split()
    #             if not query:
    #                 continue

    #             if query[0] in exit_words:
    #                 SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
    #                 break
                
    #             else:
    #                 if user_input != None:
    #                     result = chat(prompt)
    #                     if result:
    #                         result_text = str(result)
    #                         SpeechToTextTextToSpeechIO.speak_mainframe(result_text)
    #                         print(result_text)
    #                     else:
    #                         print("Error: Chat failed.")
    #                     time.sleep(0.1)
    #                     if not result:
    #                         attempt_count = 1  # Initialize re-try attempt count
    #                         while attempt_count < 5:
    #                             result = chat(prompt)
    #                             attempt_count += 1  # Increment attempt count
    #                             if result:
    #                                 result_text = str(result)
    #                                 SpeechToTextTextToSpeechIO.speak_mainframe(result_text)
    #                                 print(result_text)
    #                             else:
    #                                 print("Error: Chat failed.")
    #                             time.sleep(0.1)
