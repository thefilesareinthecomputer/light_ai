@staticmethod
def recall_notes():
    SpeechToTextTextToSpeechIO.speak_mainframe('Say "list", "statistics", or "recall".')
    while True:
        user_choice = SpeechToTextTextToSpeechIO.parse_user_speech()
        if not user_choice:
            continue

        choice_query = user_choice.lower().split()
        if choice_query[0] in exit_words:
            SpeechToTextTextToSpeechIO.speak_mainframe('Operation cancelled.')
            return

        # Listing available subjects
        if 'list' in choice_query:
            try:
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                    with driver.session() as session:
                        result = session.run("MATCH (n:Note) RETURN DISTINCT n.subject ORDER BY n.subject")
                        subjects = [record['n.subject'] for record in result]
                    if subjects:
                        subject_list = ', '.join(subjects)
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"Available subjects: {subject_list}")
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe('No subjects found.')
            except Exception as e:
                SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while retrieving subjects.')
                print(e)
            return

        # Getting database statistics
        elif 'statistics' in choice_query:
            try:
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                    with driver.session() as session:
                        # Count nodes by label
                        label_counts = session.run("MATCH (n) UNWIND labels(n) AS label RETURN label, COUNT(*) AS count")
                        labels_info = [f"Label {record['label']}: {record['count']} nodes" for record in label_counts]

                        # Add more statistics as needed

                    if labels_info:
                        stats_info = '\n'.join(labels_info)
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"Database statistics:\n{stats_info}")
                        print(f"Database statistics:\n{stats_info}")
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe('No statistics found.')
            except Exception as e:
                SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while retrieving database statistics.')
                print(e)
            return

        # Recalling specific notes
        elif 'recall' in choice_query:
            SpeechToTextTextToSpeechIO.speak_mainframe('Which subject notes would you like to recall?')
            subject_response = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not subject_response or subject_response.lower().split()[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Note recall cancelled.')
                return

            subject = subject_response.strip().lower()
            try:
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                    with driver.session() as session:
                        result = session.run("""
                            MATCH (n:Note {subject: $subject})
                            RETURN n.content, n.timestamp
                            ORDER BY n.timestamp DESC
                        """, subject=subject)
                        notes = [f"Date: {record['n.timestamp']}, Note: {record['n.content']}" for record in result]
                    if notes:
                        SpeechToTextTextToSpeechIO.speak_mainframe(" ".join(notes))
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe('No notes found for the subject.')
            except Exception as e:
                SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred during note recall.')
                print(e)
            return

        else:
            SpeechToTextTextToSpeechIO.speak_mainframe('Please specify "list", "statistics", or "recall".')   