def generate_json_intent(self):
    '''generate_json_intent is called by the chatbot when the user input is not recognized. it "works" but the content is not very intelligent yet.'''
    print("UNRECOGNIZED INPUT: writing new intent to chatbot_unrecognized_message_intents.json")
    json_gen_prompt = '''# System Message Start # - Gemini, ONLY GENERATE ONE SHORT SENTENCE FOR EACH PROMPT ACCORDING TO THE USER INSTRUCTIONS. KEEP EACH SENTENCE TO UNDER 10 WORDS, IDEALLY CLOSER TO 5. - # System Message End #'''
    # Generate an initial response using Gemini
    initial_reply = gemini_model.generate_content(f"{json_gen_prompt}. // Please provide a response to: {self.user_input}")
    initial_reply.resolve()
    bot_reply = initial_reply.text
    json_function_name = re.sub(r'\W+', '', self.user_input).lower() + '_function'
    new_intent = {
        "tag": f"unrecognized_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patterns": [self.user_input],
        "responses": [bot_reply],
        "action": json_function_name
    }

    print(f"\nAttempting to write to:\n", unrecognized_file_path)
    print(f"\nNew Intent:\n", new_intent)

    try:
        with open(unrecognized_file_path, 'r+') as file:
            data = json.load(file)
            data["intents"].append(new_intent)
            file.seek(0)
            json.dump(data, file, indent=4)
            print('New intent written to chatbot_unrecognized_message_intents.json')
    except FileNotFoundError:
        try:
            with open(unrecognized_file_path, 'w') as file:
                json.dump({"intents": [new_intent]}, file, indent=4)
                print('New file created and intent written to chatbot_unrecognized_message_intents.json')
        except Exception as e:
            print(f"Error creating new file: {e}")
    except Exception as e:
        print(f"Error updating existing file: {e}")

    print('Intent update attempted. Check the file for changes.')