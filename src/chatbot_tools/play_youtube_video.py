@staticmethod
def play_youtube_video():
    '''accepts spoken user_input and parses it into a youtube video id, then launches the video in the default browser'''
    SpeechToTextTextToSpeechIO.speak_mainframe('What would you like to search on YouTube?.')
    while True:
        user_input = SpeechToTextTextToSpeechIO.parse_user_speech().lower()
        if not user_input:
            continue

        query = user_input.lower().split()
        if not query:
            continue

        if query[0] in exit_words:
            SpeechToTextTextToSpeechIO.speak_mainframe('Ending youtube session.')
            break
        
        search_query = ' '.join(query)
        print("YouTube Query:", search_query)
        url = f'https://www.youtube.com/results?search_query={search_query}'
        webbrowser.open(url)
        break