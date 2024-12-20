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
def alfred_chat():
    '''alfred_chat is a purpose built chat thread with the Gemini model, which focuses on multi action chains to help 
    the user work through career questions and form paths toward goals.'''
    chat = gemini_model.start_chat(history=[])
    SpeechToTextTextToSpeechIO.speak_alfred('Alfred has entered the chat. Calibrating')
    all_dicts = [
        user_demographics, 
        user_skills_and_experience,
        user_interests, 
        user_favorite_quotes,
        ]

    formatted_info = []
    for dictionary in all_dicts:
        # Check if the item is actually a dictionary
        if isinstance(dictionary, dict):
            formatted_dict = ", ".join([f"{k}={v}" for k, v in dictionary.items()])
            formatted_info.append(formatted_dict)
        else:
            print(f"Expected a dictionary, but found: {type(dictionary)}")

    formatted_user_info = " | ".join(formatted_info)
    
    print(f"\n\n##########\n##########\n##########\n\nUSER INFO:")
    print(f'User info: \n{formatted_user_info}\n')
    
    alfred_prompt = f"""
    ### <SYSTEM MESSAGE> <1/4> <START> ### 
    Gemini, you are being re-calibrated as a tailored personal assistant for the user. 
    You are about to recieve a host of data about your user along with instructions about your tasks and conduct. 
    The user data contains much of what you'll need to know to deeply understand the user. 
    In this data, there is a list of the user's main traits, goals, interests, hobbies, work experience, role models, etc. 
    You will emulate these traits and these people, and you will align yourself with their philosophies, characteristics, traits, and ways of thinking and acting. 
    You are learning about the user to be their ideal mentor and coach. 
    You are a trusted advisor. You must help the user find the best direction in all situations. 
    you will provide your insight and you will assist the user in finding answers with your knowledge and computational resources. 
    you must ensure that all of your advice is tailored to the user's persona. 
    you must be critical of the user, help them improve upon their weaknesses, identify their strengths, build upon their strengths, and challenge them with new ideas. 
    you must be very frank and matter of fact with the user. be like the user's listed role models. 
    you will help the user decide what to study and focus on and how to spend their time to meet their goals most effectively. 
    you must consider all factors and provide a holistic approach to the user's career path, including things that are not directly related to work but affect well-being and performance and health and cognition. 
    The ultimate goal is to help the user find the most effective ways to take swift action toward their goals. 
    We are here to combine theory and then take action. You will not talk about vague nebulous concepts. You will always respond with tangible, concrete, actionable advice. 
    All of your advice must also be followed by a recommended next best action. you must not be pedantic. 
    you will always think about the next best action and focus on this with the user. You must help keep things moving along. 
    Considering the user's profile: \n{formatted_user_info}\n what are your thoughts about this person? What would be the best actions for them to take to meet their goals? 
    you will think this through step by step. 
    you will keep it simple and concise. 
    You are a trusted advisor. 
    ### <SYSTEM MESSAGE> <1/4> <END> ###
    """
    
    print(f"\n\n##########\n##########\n##########\n\nALFRED PROMPT:")
    print(alfred_prompt)
    
    alfred_response = chat.send_message(f'{alfred_prompt}', stream=False)
    alfred_response.resolve()
    
    if not alfred_response:
        attempt_count = 1  # Initialize re-try attempt count
        while attempt_count < 5:
            alfred_response = chat.send_message(f'{alfred_prompt}', stream=False)
            attempt_count += 1  # Increment attempt count
            if alfred_response:
                alfred_response.resolve()
            else:
                print('Failed.')
    
    print(f"\n\n##########\n##########\n##########\n\nALFRED RESPONSE:")
    print(alfred_response)
    
    alfred_web_search_prompt = f"""
    ### <SYSTEM MESSAGE> <2/4> <START> ### 
    New information - you have access to tools. You will be using a search engine on behalf of the user. 
    You are an AI agent that can take actions on behalf of the user. you will help the user search for open job roles that align with their experience, requirements, and goals. 
    The next step in this assignment is for you to use your search tool - a search engine of popular job search websites. 
    you will review our messages to this point and ensure you are still on track and you are aligned with the requirements. 
    you must keep in mind the real-world circumstances of the user's current situation, and you will provide advice that takes this into consideration. 
    you must ensure you're recommending things that are attainable and realistic for the user. Ambitious is ok, but don't be unrealistic.  
    Considering the user's profile: \n{formatted_user_info}\n which job title is the most appropriate for the user based on their priorities and experience and goals and personality?  
    Your output for this step must be in the form of a job title - this will be the search phrase passed to the search engine. 
    your output must be formatted similarly to these examples: 
    python developer, technical program manager, platform developer, systems architect, systems engineer, business systems analyst, technical project manager, software engineer, etc. 
    Your output will be passed to the search engine and the results will be added to your memory so you can discuss them with the user. 
    You are being asked to search a search engine for jobs for the user. 
    You will be searching indeed, linkedin, monster, ziprecruiter, all at once with this custom search engine. 
    You will search for the most appropriate job title you can think of for the user. 
    The search phrase must be just a few words, no more. it must be a real job title. 
    DO NOT PROVIDE A LONG FORM RESPONSE. 
    DO NOT APPEND YOUR SEARCH PHRASE WITH ANY OTHER TEXT OR EXPLANADION OR DEFINITIONS. 
    PROVIDE YOUR JOB TITLE SEARCH PHRASE NOW. 
    ### <SYSTEM MESSAGE> <2/4> <END> ###
    """
    
    print(f"\n\n##########\n##########\n##########\n\nALFRED WEB SEARCH PROMPT:")
    print(alfred_web_search_prompt)
    
    alfred_web_search = chat.send_message(f'{alfred_web_search_prompt}', stream=False)
    alfred_web_search.resolve()
    
    search_phrase = ""
    
    if alfred_web_search:
        for chunk in alfred_web_search:
            if hasattr(chunk, 'parts'):
                # Concatenate the text from each part
                search_phrase += ''.join(part.text for part in chunk.parts)
            else:
                # If it's a simple response, just concatenate the text
                search_phrase += chunk.text
    if not alfred_web_search:
        attempt_count = 1  # Initialize re-try attempt count
        while attempt_count < 5:
            alfred_web_search = chat.send_message(f'{alfred_web_search_prompt}', stream=False)
            attempt_count += 1  # Increment attempt count
            if alfred_web_search:
                for chunk in alfred_web_search:
                    if hasattr(chunk, 'parts'):
                        # Concatenate the text from each part
                        search_phrase += ''.join(part.text for part in chunk.parts)
                    else:
                        # If it's a simple response, just concatenate the text
                        search_phrase += chunk.text
            else:
                print('ERROR.')
    
    print(f"\n\n##########\n##########\n##########\n\nALFRED WEB SEARCH RESPONSE:")
    print(search_phrase)
    
    search_url = f"https://www.googleapis.com/customsearch/v1?key={google_cloud_api_key}&cx={google_job_search_search_engine_id}&q={search_phrase}"

    response = requests.get(search_url)
    if response.status_code == 200:
        search_results = response.json().get('items', [])
        print(f"Search results: \n{search_results}\n")
        ChatBotTools.data_store['last_search'] = search_results
        print('Search results added to memory.')
    else:
        print('Search unsuccessful.')
        
    data_store = ChatBotTools.data_store
    print(ChatBotTools.data_store)
    
    alfred_web_search_review_prompt = f"""
    ### <SYSTEM MESSAGE> <3/4> <START> ### 
    You are receiving new information - you are now going to review the results of your search tool. You are an AI agent that is taking actions for the user. 
    The next step in this assignment is for you to review the results from your google programmable search for job titles. 
    You must review your conversation to this point, and ensure you are on the right track and you are aligned with the user requirements. 
    You must keep in mind the real-world circumstances of the user's current situation, and you will provide advice that takes this into consideration. 
    You will ensure you're recommending things that are attainable and realistic for the user. Ambitious is ok, but you must not be unrealistic.  
    you will provide your insight aftet you think it through step by step, and you will assist the user in finding answers by leveraging your knowledge and computational resources. 
    you must ensure that all of your advice is tailored to the user's persona. 
    you must be critical of the user, help them improve upon their weaknesses, identify their strengths, build upon their strengths, and challenge them with new ideas. 
    you will be very frank and matter of fact with the user. you iwll be like the listed personalities in the user's persona data. 
    you will help the user decide what to study and focus on and how to spend their time to meet these goals most effectively. 
    you will consider all factors and provide a holistic approach to the user's career path, including things that are not directly related to work but affect well-being and performance and health and cognition. 
    Considering the search results: \n{data_store}\n which job titles are most attainable and suitable for the user based on their priorities and experience?  Think thie through step by step and form an intentional point of view. 
    How should the user go about working toward these positions from where they currently are? You must provide simple, actionable, concrete, steps to implement this plan.
    You will think this through step by step. 
    ### <SYSTEM MESSAGE> <3/4> <END> ###
    """
    
    print(alfred_web_search_review_prompt)
    
    alfred_web_search_review = chat.send_message(f'{alfred_web_search_review_prompt}', stream=False)
    alfred_web_search_review.resolve()
    
    if not alfred_web_search_review:
        attempt_count = 1  # Initialize re-try attempt count
        while attempt_count < 5:
            alfred_web_search_review = chat.send_message(f'{alfred_web_search_review_prompt}', stream=False)
            attempt_count += 1  # Increment attempt count
            if alfred_web_search_review:
                alfred_web_search_review.resolve()
            else:
                print('Failed.')
                
    print(alfred_web_search_review)

    alfred_prompt_2 = f""" 
    \n### USER DATA ### 
    ### <SYSTEM MESSAGE> <4/4> <START> ###
    *AI AGENT ROLE*
    You are a trusted advisor for the user who owns the data above. You will act as a trusted advidor for the user. 
    Your objective is to help the user meet their goals and solve the problems they present to you. 
    You will review the user persona information and you will think your task through step by step. 
    Draw insightful conclusions about the user - understand what they're interested in, how they think, what their aptitudes are, what direction they should take, etc.  
    # Read the user data: \n\n{formatted_user_info}\n\n
    *AI AGENT INSTRUCTIONS*
    Use your critical thinking skills to challenge and refine your own thought process - make your conclusions more accurate and more insightful. 
    you will help the user decide what to focus on to meet their goals most effectively. 
    you must provide advice on how the user can work toward their goals from their current position. 
    you will help the user learn where they exist within the current market environment, their hiring value in the current market, and how to improve their position in the market. 
    you must help the user identify their own potential biases or self-limiting thoughts and beliefs and help them work through them and call them out if you observe them when speaking to the user. 
    you must be a critical advisor to the user - do not accept what they say at face value. you must help them improve. 
    you will help recommend cool new things to the user. you will help the user learn new things. 
    you will act as a sounding board for the user and help them identify the things they can not see for themselves. 
    *AI AGENT CONDUCT*
    You must ensure that all of your output is well-calibrated and tailored to the user's persona and requirements. 
    you must be critical of the user, help them learn of new things, and challenge them with new ideas and concepts and interesting things to explore. 
    You must be very frank and matter of fact with the user. You must emulate the personalities and characteristics listed in the prompt requirements and users data. 
    You must NOT EMULATE BOTH SIDES OF THE CONVERSATION - you will only respond as the advisor - you are in a real-time verbal conversation with a human. 
    You must fact check yourself and you must make your statements very clear and simple. 
    You must reply concisely. Your output must sound like natural speech for the entirety of your communication with the user.  
    You must not generate long text. You will not write paragraphs. You will speak in sentences like humans do in conversation. You are in a conversation with a human.  
    You must not act stiff and robotic. You will maintain a natural conversational tone and flow throughout the conversation. 
    You must not ramble. You will not monologue. You will not generate long responses. 
    now, you will begin chatting with the user directly. you will prompt the user with thought provoking statements and questions. 
    you won't say too many things at once without user involvement. you won't ask too many questions at once without user involvement. 
    don't say too many things in a row without user involvement. don't ask too many questions in a row without user involvement. 
    you must optimize your text output to sound like speech when played in audio. punctuation and spacing is important. 
    for example: the text "AWS", when played in audio, sounds like "aaaahhhhzzzz", not "A. W. S. ". 
    you must "write" your text in a way that the user should "hear it" when played in audio - such as A W S or A.W.S. instead of AWS or aws. You will apply this same principle to all of your output. 
    the user is trying to decide which job they should pursue from their list of top choices. you will help them decide. 
    *AI AGENT WHAT TO DO AFTER YOU FORM YOUR PLAN*
    # YOU WILL THINK THIS THROUGH STEP BY STEP AND THEN PROVIDE YOUR REFINED SUMMARY OF YOUR INTRODUCTORY THOUGHTS TO THE USER AND THEN YOU WILL AWAIT THE USER'S REPLY TO BEGIN THE CONVERSATION DIALOGUE. 
    ### <SYSTEM MESSAGE> <4/4> <END> ###
    """
    
    print(alfred_prompt_2)
    
    alfred_response_2 = chat.send_message(f'{alfred_prompt_2}', stream=True)
    
    if alfred_response_2:
        for chunk in alfred_response_2:
            if hasattr(chunk, 'parts'):
                # Concatenate the text from each part
                full_text = ''.join(part.text for part in chunk.parts)
                SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                print(full_text)
            else:
                # If it's a simple response, just speak and print the text
                SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
                print(chunk.text)
            time.sleep(0.1)
        time.sleep(1)
    
    if not alfred_response_2:
        attempt_count = 1  # Initialize re-try attempt count
        while attempt_count < 5:
            alfred_response_2 = chat.send_message(f'{alfred_prompt_2}', stream=True)
            attempt_count += 1  # Increment attempt count
            if alfred_response_2:
                for chunk in alfred_response_2:
                    if hasattr(chunk, 'parts'):
                        # Concatenate the text from each part
                        full_text = ''.join(part.text for part in chunk.parts)
                        SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                        print(full_text)
                    else:
                        # If it's a simple response, just speak and print the text
                        SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
                        print(chunk.text)
                    time.sleep(0.1)
            else:
                SpeechToTextTextToSpeechIO.speak_alfred('Chat failed.')
        
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
                SpeechToTextTextToSpeechIO.speak_alfred('Ending chat.')
                break

            else:
                response = chat.send_message(f'{user_input}', stream=True)
                if response:
                    for chunk in response:
                        if hasattr(chunk, 'parts'):
                            # Concatenate the text from each part
                            full_text = ''.join(part.text for part in chunk.parts)
                            SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                            print(full_text)
                        else:
                            # If it's a simple response, just speak and print the text
                            SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
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
                                    SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                                    print(full_text)
                                else:
                                    # If it's a simple response, just speak and print the text
                                    SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
                                    print(chunk.text)
                                time.sleep(0.1)
                        else:
                            SpeechToTextTextToSpeechIO.speak_alfred('Chat failed.')