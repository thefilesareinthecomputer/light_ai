# standard imports
import os
import queue
import subprocess
import threading
import time
# third-party imports
import speech_recognition as sr
# local imports
import chatbot.chatbot_global_state

# ENVIRONMENT VARIABLES ###################################################################################################################################
from dotenv import load_dotenv
load_dotenv()
USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Evan')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')

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

# CLASS DEFINITIONS ###################################################################################################################################

# conversation_history = []

class SpeechToTextTextToSpeechIO:
    '''SpeechToTextTextToSpeechIO handles the speech to text and text to speech functionality of the chatbot. It also handles the speech output queue.
    the speech output queue places all text chunks output from the bot and plays them in order so they don't overlap. The speech manager thread is constantly checking the queue for new items. 
    the speech manager runs on its own thread so that the bot can still recieve input while speaking. this hasn't been built out to its full potential yet 
    because we haven't figured out how to get the bot to listen for input while it is speaking without hearing itself. we also sometimes have issues 
    with the bot hearing and speaking to itself by mistake. we are trying to solve this by using time.sleep() to pause the bot while the speech manager 
    is producing auido, but the timing is not perfect yet.'''
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
                        # input_speech = listener.listen(source, timeout=20, phrase_time_limit=8)  # experimenting with different timeout and phrase time limit settings
                        input_speech = listener.listen(source, timeout=10, phrase_time_limit=10)  # this setting feels better
                    print('Processing...')
                    query = listener.recognize_google(input_speech, language='en_US')  # online transcription with Google Speech Recognition API - most accurate
                    # query = listener.recognize_sphinx(input_speech, language='en_US')  # offline transcription with PocketSphinx - not as accurate - needs fine tuning
                    print(f'You said: {query}\n')
                    return query

                except sr.WaitTimeoutError:
                    print('Listening timed out. Please try again.')
                    return None

                except sr.UnknownValueError:
                    print('Speech not recognized. Please try again.')
                    return None

    @classmethod
    def calculate_speech_duration(cls, text, rate):
        '''calculate_speech_duration calculates the duration of the speech based on text length and speech rate. 
        the intent for calculate_speech_duration is to calculate how long each piece of speech output will "take to say". 
        it will be used for various reasons, primarily a time.sleep() for bot listening in the speech manager. 
        that said, this is a workaround that will eventually conflict with our desired funcitonality of the bot being able to 
        listen while it is speaking. also, the timing is sometimes inaccurate.'''
        words = text.split() if text else []
        number_of_words = len(words)
        minutes = number_of_words / rate
        seconds = minutes * 60
        return seconds + 1
    
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
    def speak_mainframe(cls, text, rate=185, chunk_size=1000, voice=USER_PREFERRED_VOICE):
        '''speak_mainframe contains the bot's speech output voice settings, and it puts each chunk of text output from the bot or the LLM 
        into the speech output queue to be processed in sequential order. it also separately returns the estimated duration of the speech 
        output (in seconds), using the calculate_speech_duration function.'''
        # global conversation_history
        chatbot.chatbot_global_state.conversation_history.append("Bot: " + text)
        cls.queue_lock.acquire()
        try:
            cls.speech_queue.put((text, rate, chunk_size, voice))
            speech_duration = cls.calculate_speech_duration(text, rate)
        finally:
            cls.queue_lock.release()
        return speech_duration
    
    @classmethod
    def speak_alfred(cls, text, rate=185, chunk_size=1000, voice="Oliver"):
        # global conversation_history
        chatbot.chatbot_global_state.conversation_history.append("Bot: " + text)
        cls.queue_lock.acquire()
        try:
            cls.speech_queue.put((text, rate, chunk_size, voice))
            speech_duration = cls.calculate_speech_duration(text, rate)
        finally:
            cls.queue_lock.release()
        return speech_duration
    
    @classmethod
    def speak_alignment(cls, text, rate=185, chunk_size=1000, voice=USER_PREFERRED_VOICE):
        '''speak_mainframe contains the bot's speech output voice settings, and it puts each chunk of text output from the bot or the LLM 
        into the speech output queue to be processed in sequential order. it also separately returns the estimated duration of the speech 
        output (in seconds), using thecalculate_speech_duration function.'''
        # global conversation_history
        chatbot.chatbot_global_state.conversation_history.append("Bot: " + text)
        cls.queue_lock.acquire()
        try:
            cls.speech_queue.put((text, rate, chunk_size, voice))
            speech_duration = cls.calculate_speech_duration(text, rate)
        finally:
            cls.queue_lock.release()
        return speech_duration