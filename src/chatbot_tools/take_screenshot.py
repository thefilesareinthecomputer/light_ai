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