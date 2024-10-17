@staticmethod
def save_note():
    SpeechToTextTextToSpeechIO.speak_mainframe('What is the subject of the note?')
    time.sleep(1.5)
    while True:
        subject_response = SpeechToTextTextToSpeechIO.parse_user_speech()
        if not subject_response:
            continue  # Wait for valid input

        subject_query = subject_response.lower().split()
        if not subject_query or subject_query[0] in exit_words:
            SpeechToTextTextToSpeechIO.speak_mainframe('Note saving cancelled.')
            return
        else:
            break  # Valid input received

    subject = subject_response.strip().lower()

    SpeechToTextTextToSpeechIO.speak_mainframe('Please say the content of the note.')
    time.sleep(1.5)
    while True:
        content_response = SpeechToTextTextToSpeechIO.parse_user_speech()
        if not content_response:
            continue  # Wait for valid input

        content_query = content_response.lower().split()
        if not content_query or content_query[0] in exit_words:
            SpeechToTextTextToSpeechIO.speak_mainframe('Note saving cancelled.')
            return
        else:
            break  # Valid input received

    content = content_response.strip()

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                session.run("""
                    CREATE (n:UserVoiceNotes:Note:UserChatBotInteractions {subject: $subject, content: $content, timestamp: $timestamp})
                """, subject=subject, content=content, timestamp=timestamp)
            SpeechToTextTextToSpeechIO.speak_mainframe('Note saved successfully.')
    except Exception as e:
        SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while saving the note.')
        print(e)