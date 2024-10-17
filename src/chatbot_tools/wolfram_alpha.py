@staticmethod
def wolfram_alpha():
    '''wolfram_alpha returns a summary of a wolfram alpha query based on user input'''
    wolfram_client = wolframalpha.Client(wolfram_app_id)
    SpeechToTextTextToSpeechIO.speak_mainframe(f'Initializing wolfram alpha. State your query.')
    while True:
        user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
        if not user_input:
            continue

        query = user_input.lower().split()
        if not query:
            continue

        if query[0] in exit_words:
            SpeechToTextTextToSpeechIO.speak_mainframe('Ending session.')
            break
        
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Heard.')
        try:
            response = wolfram_client.query(user_input)
            print(f"Response from Wolfram Alpha: {response}")

            # Check if the query was successfully interpreted
            if not response['@success']:
                suggestions = response.get('didyoumeans', {}).get('didyoumean', [])
                if suggestions:
                    # Handle multiple suggestions
                    if isinstance(suggestions, list):
                        suggestion_texts = [suggestion['#text'] for suggestion in suggestions]
                    else:
                        suggestion_texts = [suggestions['#text']]

                    suggestion_message = " or ".join(suggestion_texts)
                    SpeechToTextTextToSpeechIO.speak_mainframe(f"Sorry, I couldn't interpret that query. These are the alternate suggestions: {suggestion_message}.")
                else:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Sorry, I couldn\'t interpret that query. Please try rephrasing it.')

                return 'Query failed.'

            # Filtering and storing all available pods
            wolfram_data = []
            for pod in response.pods:
                pod_data = {'title': pod.title}
                if hasattr(pod, 'text') and pod.text:
                    pod_data['text'] = pod.text
                wolfram_data.append(pod_data)

            # # Adding to data store
            # ChatBotTools.data_store['wolfram_alpha_response'] = {
            #     'query': user_input,
            #     'pods': wolfram_data
            # }  
            # SpeechToTextTextToSpeechIO.speak_mainframe('Search complete. Data saved to memory.') 

            # Initialize the data store dictionary if it doesn't exist
            if 'wolfram_alpha_responses' not in ChatBotTools.data_store:
                ChatBotTools.data_store['wolfram_alpha_responses'] = {}

            # Generate a unique key for the current query
            # For example, using a timestamp or an incrementing index
            unique_key = f"query_{len(ChatBotTools.data_store['wolfram_alpha_responses']) + 1}"

            # Store the response using the unique key
            ChatBotTools.data_store['wolfram_alpha_responses'][unique_key] = {
                'query': user_input,
                'pods': wolfram_data
            }

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"An error occurred: {e}\nDetails: {error_traceback}")
            SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while processing the query. Please check the logs for more details.')
            return f"An error occurred: {e}\nDetails: {error_traceback}"
        
        SpeechToTextTextToSpeechIO.speak_mainframe('Would you like to run another query?') 
        user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
        if not user_input:
            continue
        query = user_input.lower().split()
        if not query or query[0] in exit_words or query[0] in ['no', 'nope', 'nah', 'not', 'not yet', 'cancel', 'exit', 'quit']:
            SpeechToTextTextToSpeechIO.speak_mainframe('Ending session.')
            break