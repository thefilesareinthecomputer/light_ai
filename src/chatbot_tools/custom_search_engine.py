@staticmethod
def custom_search_engine():
    global google_documentation_search_engine_id
    global google_job_search_search_engine_id
    global google_health_search_engine_id
    global google_research_search_engine_id
    global google_restaurant_search_engine_id
    SpeechToTextTextToSpeechIO.speak_mainframe("Which engine do you want to use?")
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
        
        engine = ' '.join(query).lower()
        
        if engine in ['coding', 'programming', 'python', 'documentation', 'document', 'docs', 'documentation search', 'document search', 'docs search']:
            google_search_engine_id = google_documentation_search_engine_id
        
        if engine in ['job search', 'job', 'jobs', 'career']:
            google_search_engine_id = google_job_search_search_engine_id
            
        if engine in ['health', 'health search', 'healthcare', 'healthcare search', 'supplements', 'nutrition', 'fitness', 'wellness']:
            google_search_engine_id = google_health_search_engine_id
            
        if engine in ['research', 'work', 'design', 'scientific', 'science', 'work work']:
            google_search_engine_id = google_research_search_engine_id
            
        if engine in ['restaurant', 'restaurants', 'food', 'dining', 'dine', 'eat', 'eating', 'restaurant search']:
            google_search_engine_id = google_restaurant_search_engine_id
    
        SpeechToTextTextToSpeechIO.speak_mainframe("Speak your search query.")
        time.sleep(2)
        
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
            
            search_query = ' '.join(query)
        
            search_url = f"https://www.googleapis.com/customsearch/v1?key={google_cloud_api_key}&cx={google_search_engine_id}&q={search_query}"

            response = requests.get(search_url)
            if response.status_code == 200:
                search_results = response.json().get('items', [])
                print(f"Search results: \n{search_results}\n")
                ChatBotTools.data_store['last_search'] = search_results
                SpeechToTextTextToSpeechIO.speak_mainframe('Search results added to memory.')
                return search_results
            else:
                SpeechToTextTextToSpeechIO.speak_mainframe('Search unsuccessful.')
                print(f"Error: {response.status_code}")
                return f"Error: {response.status_code}"