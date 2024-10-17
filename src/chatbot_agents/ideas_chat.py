@staticmethod
def ideas_chat():
    '''ideas_chat is a brainstorming chat thread with the Gemini model, which focuses on multi action chains to help 
    the user work through questions and form implementation plans for goals.'''
    chat = gemini_model.start_chat(history=[])
    SpeechToTextTextToSpeechIO.speak_mainframe('Brainstorm has entered the chat. Calibrating')
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
    
    ideas_prompt = f""" 
    \n### USER PERSONA DATA ### 
    \n{formatted_user_info}\n\n 
    ### <SYSTEM MESSAGE> <1/1> <START> ###
    You are a trusted advisor for the user who owns the data above. You must act as a trusted advidor for the user. 
    The goal of this conversation is a brainstorming conversation with the user, to help the user work through questions and form implementation plans for goals. 
    Your objective is to help the user meet their goals or solve the problems they present to you. 
    You will review the user persona information and you will think your task through step by step. 
    Draw insightful conclusions about the user - understand what they're interested in, how they think, what advice they need, etc.  
    Use your critical thinking skills to challenge and refine your own thought process - make your conclusions more accurate and more insightful. 
    You must fact check yourself and you must make your statements very clear and simple. 
    You must reply concisely. Your output must sound like natural speech for the entirety of your communication with the user.  
    Do not generate long text. Do not write paragraphs. Speak in sentences like humans do in conversation. You are in a conversation with a human. 
    You must ensure that all of your output is calibrated and tailored to the user's persona and requirements. 
    you must be critical of the user, help them learn of new things, and challenge them with new ideas and concepts and interesting things to explore. 
    You must be very frank and matter of fact with the user. You must emulate the personalities and characteristics listed in the prompt requirements and users data. 
    Help recommend cool new things to the user. Help the user learn new things. 
    DO NOT EMULATE BOTH SIDES OF THE CONVERSATION - only respond as the advisor - you are in a real-time verbal conversation with a human.  
    You must not act stiff and robotic. You will maintain a natural conversational tone and flow throughout the conversation. 
    do not ramble. do not monologue. do not generate long responses. 
    act as a sounding board for the user and help them identify the things they can not see for themselves. 
    now you will begin chatting with the user directly. prompt the user with thought provoking statements and questions. 
    don't say too many things at once. don't ask too many questions at once. don't say too many things in a row. don't ask too many questions in a row. 
    THINK THIS THROUGH STEP BY STEP AND THEN PROVIDE YOUR REFINED INTRODUCTORY THOUGHTS TO THE USER AND THEN AWAIT THE USER'S REPLY TO BEGIN THE CONVERSATION DIALOGUE. 
    ### <SYSTEM MESSAGE> <1/1> <END> ### 
    """
    
    print(ideas_prompt)
    
    ideas_response = chat.send_message(f'{ideas_prompt}', stream=True)
    
    if ideas_response:
        for chunk in ideas_response:
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
    if not ideas_response:
        attempt_count = 1  # Initialize re-try attempt count
        while attempt_count < 5:
            ideas_response = chat.send_message(f'{ideas_prompt}', stream=True)
            attempt_count += 1  # Increment attempt count
            if ideas_response:
                for chunk in ideas_response:
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

            else:
                response = chat.send_message(f'{user_input}', stream=True)
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