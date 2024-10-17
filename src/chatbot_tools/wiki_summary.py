@staticmethod
def wiki_summary():
    '''wiki_summary returns a summary of a wikipedia page based on user input.'''
    SpeechToTextTextToSpeechIO.speak_mainframe('What should we summarize from Wikipedia?')

    while True:
        user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
        if not user_input:
            continue

        query = user_input.lower().split()
        if not query:
            continue

        if query[0] in exit_words:
            SpeechToTextTextToSpeechIO.speak_mainframe('Canceling search.')
            break

        print("Wikipedia Query:", user_input)
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Searching {user_input}')

        try:
            search_results = wikipedia.search(user_input)
            if not search_results:
                print('No results found.')
                continue

            wiki_page = wikipedia.page(search_results[0])
            wiki_title = wiki_page.title
            wiki_summary = wiki_page.summary

            response = f'Page title: \n{wiki_title}\n, ... Page Summary: \n{wiki_summary}\n'
            # Storing Wikipedia summary in the data store
            ChatBotTools.data_store['wikipedia_summary'] = {
                'query': user_input,
                'title': wiki_title,
                'summary': wiki_summary,
                'full_page': str(wiki_page)
            }
            print(response)
            SpeechToTextTextToSpeechIO.speak_mainframe(f"{user_input} summary added to global data store.")
            SpeechToTextTextToSpeechIO.speak_mainframe("Would you like to hear the summary now?")
            while True:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue

                query = user_input.lower().split()
                if not query:
                    continue

                if query[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Canceling search.')
                    break
                
                if query[0] in ['yes', 'yeah', 'ok', 'sure', 'yep']:
                    SpeechToTextTextToSpeechIO.speak_mainframe(f"{wiki_summary}")
                    break
                
                else:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Ok.')
                    break
            
            break

        except wikipedia.DisambiguationError as e:
            try:
                # Attempt to resolve disambiguation by selecting the first option
                wiki_page = wikipedia.page(e.options[0])
                continue
            except Exception as e:
                print(f"Error resolving disambiguation: {e}")
                break

        except wikipedia.PageError:
            print("Page not found. Please try another query.")
            SpeechToTextTextToSpeechIO.speak_mainframe("Error: Page not found.")
            continue

        except wikipedia.RequestsException:
            print("Network error. Please check your connection.")
            SpeechToTextTextToSpeechIO.speak_mainframe("Error: No network connection.")
            break

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            SpeechToTextTextToSpeechIO.speak_mainframe(f"An error occured. Message: {e}")
            break
    