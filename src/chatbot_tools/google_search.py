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