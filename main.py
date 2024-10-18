'''
This is the main module of the chatbot application. 
It is responsible for starting the user interface and the chatbot.
This module is under development and not currently in use as of 2024-04-17.
The app is currently being run from src_local_chatbot/chatbot_app_logic.py
'''

# standard imports
import threading
import flet as ft

from src_local_chatbot.chatbot_app_logic import ui_main, run_chatbot, SpeechToTextTextToSpeechIO

if __name__ == '__main__':
    '''
    The speech to text and the UI are run on separate threads.
    '''
    threading.Thread(target=SpeechToTextTextToSpeechIO.speech_manager, daemon=True).start()
    threading.Thread(target=run_chatbot, daemon=True).start()
    ft.app(target=ui_main)