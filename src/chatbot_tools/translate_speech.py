@staticmethod
def translate_speech():
    '''Translats a spoken phrase from user's preferred language to another language by saying 
    "{activation_word}, translate" or "{activation_word}, help me translate".'''
    language_code_mapping = {
        "en": ["english", "Daniel"],
        "es": ["spanish", "Paulina"],
        "fr": ["french", "Amélie"],
        "de": ["german", "Anna"],
        "it": ["italian", "Alice"],
        "ru": ["russian", "Milena"],
        "ja": ["japanese", "Kyoko"],
    }
    language_names = [info[0].lower() for info in language_code_mapping.values()]
    SpeechToTextTextToSpeechIO.speak_mainframe('What language do you want to translate to?')
    time.sleep(2)
    while True:
        user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
        if not user_input:
            continue

        query = user_input.lower().split()
        if not query:
            continue
    
        if query[0] in exit_words:
            SpeechToTextTextToSpeechIO.speak_mainframe('Canceling translation.')
            break
    
        # translate
        if query[0] in language_names:
            target_language_name = query[0]
            SpeechToTextTextToSpeechIO.speak_mainframe(f'Speak the phrase you want to translate.')
            time.sleep(2)
            phrase_to_translate = SpeechToTextTextToSpeechIO.parse_user_speech().lower()

            source_language = USER_PREFERRED_LANGUAGE  # From .env file
            target_voice = None

            # Find the language code and voice that matches the target language name
            target_language_code, target_voice = None, None
            for code, info in language_code_mapping.items():
                if target_language_name.lower() == info[0].lower():
                    target_language_code = code
                    target_voice = info[1]
                    break

            if not target_language_code:
                SpeechToTextTextToSpeechIO.speak_mainframe(f"Unsupported language: {target_language_name}")
                return

            model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language_code}'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            batch = tokenizer([phrase_to_translate], return_tensors="pt", padding=True)
            translated = model.generate(**batch)
            translation = tokenizer.batch_decode(translated, skip_special_tokens=True)
            print(f'In {target_language_name}, it\'s: {translation}')    
            SpeechToTextTextToSpeechIO.speak_mainframe(f'In {target_language_name}, it\'s: {translation}', voice=target_voice)
            continue