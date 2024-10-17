@staticmethod
def agent_two():
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    messages = [
        SystemMessage(
            content="You are a helpful assistant that helps the user solve problems. You are a high tech AI assistant similar to Jarvis or Cortana. You are not being asked to copy, just to emulate the general characteristics. Be smart and innovative. Help the user discover new ideas they may not have been thinking of. Help the user grow, learn, develop, and hone new skills."
        ),
        HumanMessage(
            content=f"Hi! I'm excited to start chatting with you today. Be aware that you are speaking verbally in a TTS / STT app. Make your responses concise and easy to understand. Make your output approapriate for conversational flow. Make sure to speak naturally and not monologue. Confirm you understand this with a brief confirmation phrase, then we'll begin chatting."
        ),
    ]
    result = chat(messages)
    print("Generated Response:", result)
    SpeechToTextTextToSpeechIO.speak_mainframe(f'{result}')
    
    while True:
        global mic_on
        if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            prompt = [
                SystemMessage(
                    content="""You are a helpful assistant that helps the user solve problems. 
                    Respond to the user's input in the best way you can. 
                    Please be aware that you are speaking verbally in a TTS / STT app. 
                    Make sure your responses are concise and easy to understand. 
                    Make sure your output has a natural conversational flow. 
                    You are a high tech AI assistant similar to Jarvis or Cortana. 
                    You are not being asked to copy, just to emulate the general characteristics. 
                    Be smart and innovative. 
                    Help the user discover new ideas they may not have been thinking of. 
                    Help the user grow, learn, develop, and hone new skills. 
                    Think each problem through step by step before you act. 
                    """
                ),
                HumanMessage(
                    content=user_input
                ),
            ]
            
            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            else:
                if user_input != None:
                    result = chat(prompt)
                    if result:
                        result_text = str(result)
                        SpeechToTextTextToSpeechIO.speak_mainframe(result_text)
                        print(result_text)
                    else:
                        print("Error: Chat failed.")
                    time.sleep(0.1)
                    if not result:
                        attempt_count = 1  # Initialize re-try attempt count
                        while attempt_count < 5:
                            result = chat(prompt)
                            attempt_count += 1  # Increment attempt count
                            if result:
                                result_text = str(result)
                                SpeechToTextTextToSpeechIO.speak_mainframe(result_text)
                                print(result_text)
                            else:
                                print("Error: Chat failed.")
                            time.sleep(0.1)
