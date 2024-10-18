# standard imports
import json
import threading
import time
import flet as ft

from dotenv import load_dotenv
load_dotenv()

# FRONT END ###################################################################################################################################

class ChatBotUI(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.speech_process = None  # To store the speech subprocess
        
    def build(self):
        self.mic_btn = ft.ElevatedButton(
            text="MICROPHONE ON/OFF", 
            tooltip="Click to toggle microphone on/off",
            on_click=self.toggle_mic, 
            elevation=10,
            bgcolor=ft.colors.GREY, 
            color=ft.colors.WHITE,
        )
        
        self.stop_btn = ft.ElevatedButton(
            text="RESET CONVERSATION", 
            on_click=self.reset_conversation, 
        )
        
        self.response_text = ft.Text(
            value="Mic is off"
        )
        
        self.conversation_list = ft.ListView(
            auto_scroll=True,
            width=600,
            height=1000,
        )
        
        self.data_text = ft.TextField(
            value="", 
            multiline=True, 
            read_only=True, 
            height=800, 
            width=600,
        )
        
        controls_column = ft.Column(
            controls=[
                self.mic_btn, 
                self.stop_btn, 
                self.response_text,
                ], 
            spacing=15,
        )
        
        convo_column = ft.Column(
            controls=[
                self.conversation_list], 
            spacing=15, 
            scroll=ft.ScrollMode.ALWAYS, 
            height=800, 
            width=600, 
            wrap=True,
        )
        
        data_column = ft.Column(
            controls=[
                self.data_text], 
            spacing=15,
        )
        
        return ft.Container(
            content=ft.Row(
                controls=[
                    controls_column, 
                    convo_column, 
                    data_column
                    ], 
                spacing=15,
                ), 
                padding=10,
        )
    
    def reset_conversation(self, e):
        """Resets the speech queue and conversation history, and terminates any ongoing speech."""
        # Clear speech queue and conversation history
        SpeechToTextTextToSpeechIO.queue_lock.acquire()
        try:
            while not SpeechToTextTextToSpeechIO.speech_queue.empty():
                SpeechToTextTextToSpeechIO.speech_queue.get_nowait()
            global conversation_history
            conversation_history = []
            self.conversation_list.controls.clear()  # Clear the UI
        finally:
            SpeechToTextTextToSpeechIO.queue_lock.release()

        # Terminate ongoing speech process
        if self.speech_process and self.speech_process.poll() is None:
            self.speech_process.terminate()

        self.update()
        
    def toggle_mic(self, e):
        global mic_on
        mic_on = not mic_on
        self.response_text.value = "Mic is on" if mic_on else "Mic is off"
        self.update()
    
    def update_conversation(self):
        global conversation_history
        last_index = 0
        while True:
            current_len = len(conversation_history)
            for i in range(last_index, current_len):
                self.conversation_list.controls.append(ft.Text(value=conversation_history[i], width=None, expand=True, overflow=ft.TextOverflow.VISIBLE))
                last_index += 1
            self.update()
            time.sleep(1)

    def update_data_store(self):
        global ChatBotTools
        last_data_keys = set()
        while True:
            current_data_keys = set(ChatBotTools.data_store.keys())
            new_keys = current_data_keys - last_data_keys
            if new_keys:
                new_data_texts = []
                for key in new_keys:
                    value = ChatBotTools.data_store[key]
                    formatted_value = json.dumps(value, indent=4) if isinstance(value, dict) else str(value)
                    new_data_texts.append(f"{key}:\n{formatted_value}")
                # Append the new data to the existing text
                self.data_text.value += "\n".join(new_data_texts)
                last_data_keys.update(new_keys)
            self.update()
            time.sleep(1)

    def start_threads(self):
        threading.Thread(target=self.update_conversation, daemon=True).start()
        threading.Thread(target=self.update_data_store, daemon=True).start()

def ui_main(page: ft.Page):
    page.window.width = 1400
    page.window.height = 1000
    page.title = "ROBOT"
    page.scroll = "adaptive"
    page.bgcolor = ft.colors.BLACK
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.theme.Theme(color_scheme_seed="black")
    page.padding = 50
    chatbot_ui = ChatBotUI()
    page.add(chatbot_ui)
    chatbot_ui.start_threads()

# MAIN EXECUTION ###################################################################################################################################
    
def run_chatbot():
    chatbot_app = ChatBotBrain()  # Initialize the chatbot app
    chatbot_tools = ChatBotTools()  # Initialize the chatbot tools
    chatbot_app.chat(chatbot_tools)  # Run the chatbot app with the tools

if __name__ == '__main__':
    threading.Thread(target=SpeechToTextTextToSpeechIO.speech_manager, daemon=True).start()  # Speech manager thread
    threading.Thread(target=run_chatbot, daemon=True).start()  # Chatbot app logic thread
    ft.app(target=ui_main)  # Flet UI runs on the main thread
    