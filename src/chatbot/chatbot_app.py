import threading
import flet as ft

from chatbot.chatbot_brain import ChatBotBrain
from chatbot.chatbot_speech import SpeechToTextTextToSpeechIO
from chatbot.chatbot_tools import ChatBotTools
from chatbot.chatbot_ui import ui_main

def run_chatbot():
    chatbot_brain = ChatBotBrain()  # Initialize the chatbot app
    chatbot_tools = ChatBotTools()  # Initialize the chatbot tools
    chatbot_brain.chat(chatbot_tools)  # Run the chatbot app with the tools

def main():
    try:
        threading.Thread(target=SpeechToTextTextToSpeechIO.speech_manager, daemon=True).start()  # Speech manager thread
    except Exception as e:
        print(f"Error starting speech manager thread: {e}")
    try:
        threading.Thread(target=run_chatbot, daemon=True).start()  # Chatbot app logic thread
    except Exception as e:
        print(f"Error starting chatbot logic thread: {e}")
    try:
        ft.app(target=ui_main)  # Flet UI runs on the main thread
    except Exception as e:
        print(f"Error starting Flet UI: {e}")
         
if __name__ == '__main__':
    main()
