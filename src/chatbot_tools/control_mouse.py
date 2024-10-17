@classmethod
def control_mouse(cls):
    '''control_mouse is a simple mouse control function that allows the user to control the mouse with their voice by 
    saying "{activation_word}, mouse control" or "{activation_word}, control the mouse". this will activate the mouse control 
    which the user can trigger by saying "mouse click" or "mouse up 200" (pixels), etc.'''
    SpeechToTextTextToSpeechIO.speak_mainframe('Mouse control activated.')
    direction_map = {
        'north': (0, -1),
        'south': (0, 1),
        'west': (-1, 0),
        'east': (1, 0),
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }
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
        if query[0] == 'click':
            pyautogui.click()
        elif query[0] in ['move', 'mouse', 'go'] and len(query) > 2 and query[1] in direction_map and query[2].isdigit():
            move_distance = int(query[2])  # Convert to integer
            direction_vector = direction_map[query[1]]
            pyautogui.move(direction_vector[0] * move_distance, direction_vector[1] * move_distance, duration=0.1)