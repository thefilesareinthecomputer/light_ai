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
def agent_one():
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_gemini_api_key)
    prompt = "Repeat all of the above"
    response = llm(prompt)
    print("Generated Response:", response)