## APP OVERVIEW & NOTES:
- This is a voice-activated speech-to-text ai assistant chatbot desktop app.
- It's designed to act as a copilot with the ability to access external web services. 
- It can also interact with files and browsers on the user's computer.
- It can act as a note taker, question answerer, research assistant, automator, etc. 
- It's the basis for a jarvis-like ai assistant. 
- It can be extended with any number of attitional tools and features. 
- The app contains a collection of agents and tools that work together in one cohesive experience.
- The tools available to the bot exist in the chatbot_tools module. 
- The agents are powered by LLM APIs like gemini and openai. 
- The agent can search the web, access various APIs like Wolfram|Alpha, and speak with the user.
- The agent can maintain a stateful conversation while using tools.
- The agent can also access a neo4j graph database for long-term memory that persists between sessions.
- The chatbot_intents.json file is a collection of possible user phrases and responses.
- The chatbot_intents.json file is used to train the app's base chatbot_model.keras.
- Functions (tools and agents) are triggered by chatbot_model.keras based on training data in chatbot_intents.json.
- All of the available functions are listed below in detail.
- The chatbot_model.keras model is the first model to activate when the app is run.
- It acts as the "router" and function caller, triggering tools or more complex agents based on user commands. 
- The agent has access to a shared dictionary called data_store which stores data rendered by tools. 
- when the app is running, it listens idly until it hears the activation word. 
- the activation word, followed by a recognized phrase, will trigger responses from the locally trained model (chatbot_model.keras), and then will call functions based on recognized phrases if applicable. 
- some functions are one-off actions, and others trigger sub-loops such as the 'robot, AI' command which will enter a stateful TTS<>STT chat loop with the Gemini LLM. 
- there is also base code for langchain agents powered by Gemini and GPT-3.5-turbo currently under "agent_one" and "agent_two", but they're not as built out as the Gemini chat loop yet. 
- the user interacts with the app by speaking the activation word (a global constant variable) followed by their phrases. 
- if the user speaks a phrase the bot doesn't recognize, it will save that interaction in the form of a JSON intent template that can then be vetted and corrected by the user and added into its intents.json training data file for the next training session. 
- this "memory logging" has an opportunity to become a more robust and automatic process in a future sprint. 
- List of currently available tools: 
    - google search engine via voice prompt
    - youtube search via voice prompt
    - Wolfram|Alpha queries via voice prompt
    - wikipedia search via voice prompt
    - speech translation (Opus-MT models from transformers)
    - text file translation (Opus-MT models from transformers, batch processing, no file size limit)
    - cursor control via voice prompt
    - computer vision image analysis (Gemini)
    - weather forecasts (open weather api + Gemini)
    - spotify search via voice prompt
    - note taking and recall (Neo4j graph database)
    - stock watch list reports

## VERSION
- light_ai 0.0.1
    

## LICENSE

### Open Source License
- This project is licensed under the GNU General Public License v3.0 - see the LICENSE.txt file for details. This open-source license is primarily for individual researchers, academic institutions, and non-commercial use. Contributions to the open-source version fall under the same GPLv3 licensing terms.

### Commercial License
- For commercial use of this project, a separate commercial license is required. This includes use in a commercial entity or for commercial purposes. Please contact us at https://github.com/thefilesareinthecomputer for more information about obtaining a commercial license.

## DEVELOPMENT ENVIRONMENT / DEPENDENCIES / INSTALLATION / CONFIG:

```bash
development os: macOS Sonoma 14
python 3.11.4

this app requires the following homebrew packages to be installed on the local machine:
brew install portaudio
brew install flac

clone the repository from github: https://github.com/thefilesareinthecomputer/light_ai.git

set up a virtual environment:

cd {REPO_FOLDER}

python3.11 -m venv {VENV_NAME}

source {VENV_NAME}/bin/activate
 
pip install --upgrade pip pip-check-reqs wheel python-dotenv certifi setuptools

pip install -r requirements.txt -c constraints.txt

echo "{VENV_NAME}/
venv/
archive/
secrets/
app_generated_files/
db/
__pycache__/
*.pyc
*/migrations/*
db.sqlite3
.env
notes.txt
staticfiles/" > .gitignore

cat .gitignore

create a .env file in the root directory (required variables outlined below):
.env

create a user_persona.py file in the secrets directory (optional, but recommended):
user_persona.py

This version of the app utilizes built-in macOS text to speech (TTS) engine for the bot voice, and will need slight modification on windows and linux with pyttsx3 or other TTS libraries.

Java 17 is required for the neo4j graph database
https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html

neo4j community edition is the graph database which will be used to extend long-term memory to the bot via graph RAG. Neo4j is a locally hosted self managed graph database. Download from the URL below and and place in the project root directory after unzipping.
https://neo4j.com/deployment-center/?ref=subscription#community
```

## USER-DEFINED VARIABLES IN THE .env FILE:

```bash
within the .env file, optionally declare any of these variables (or others of your own) to extend tool functionality to the assistant:

PROJECT_ROOT_DIRECTORY=
PROJECT_VENV_DIRECTORY=
PROJECT_TEST_DIRECTORY=
PROJECT_FILE_DROP_FOLDER=
PROJECT_ARCHIVE_DIRECTORY=
PYTHONPATH=

ACTIVATION_WORD=
EXIT_WORDS=

USERNAME=
PASSWORD=

USER_PREFERRED_LANGUAGE=
USER_PREFERRED_VOICE=
USER_PREFERRED_NAME=
USER_SELECTED_HOME_CITY=
USER_SELECTED_HOME_COUNTY=
USER_SELECTED_HOME_STATE=
USER_SELECTED_HOME_COUNTRY=
USER_SELECTED_HOME_LAT=4
USER_SELECTED_HOME_LON=
USER_SELECTED_TIMEZONE=
USER_STOCK_WATCH_LIST=

CENSUS_API_KEY=
GOOGLE_CLOUD_API_KEY=
GOOGLE_CLOUD_PROJECT_NUMBER=
GOOGLE_CLOUD_PROJECT_ID=
GOOGLE_MAPS_API_KEY=
GOOGLE_GEMINI_API_KEY=
GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID=
GOOGLE_JOB_SEARCH_SEARCH_ENGINE_ID=
GOOGLE_HEALTH_SEARCH_ENGINE_ID=
GOOGLE_RESEARCH_SEARCH_ENGINE_ID=
GOOGLE_RESTAURANT_SEARCH_ENGINE_ID=
OPEN_WEATHER_API_KEY=
WOLFRAM_APP_ID=
OPENAI_API_KEY_OLD=
OPENAI_API_KEY=

JAVA_HOME=
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
NEO4J_DATABASE=
NEO4J_PATH=
```

## USER DEFINED VARIABLES IN THE src/src_local_chatbot/user_persona.py FILE:

```bash
optionally, within the secrets folder, the user can create a user_persona.py file for themself that is used to calibrate better responses from the bots. the user_persona.py file is a series of dictionaries that can be passed to the bot in prompt templates. some examples are:
user_demographics, 
user_skills_and_experience,
user_interests, 
user_favorite_quotes,
```

## FEATURE BACKLOG (planned additions, improvements, and bug fixes):

```bash
implement rag with the neo4j graph database.
populate the neo4j graph database with nodes and relationships.
gain the ability to ingest knowledge from various media, interpret and summarize it, index it to a knowledge graph using a neo4j database, be able to query it based on user input, and be able to converse about it with the user in real time.
new voices for the speech interface independent of hardware - pyttsx3 or Google Cloud Text-to-Speech. 
tailored news reports. Implement a feature to fetch news from APIs, filtering content based on user preferences or persona. 
communications integration (sms, google voice, whatsapp, signal, email, etc.).
add tqdm progress bars to long running tasks.
translators: currently using the Opus-MT models from transformers for real time speech translation. may need to move to google translate for in-browser, and something like deepl for longer documents.
click a link on the screen by name, and select a text field, check box, or button, etc. based on user speech.
select a tab or window by description based on user speech (a description, such as 'top left of the screen, in back' or 'bottom right of the screen, in front', or 'minimized browser windows' or 'minimized VS code windows' or 'minimized images').
system commands.
meditation coach.
restaurant reservation assistant.
real estate analyzer. Utilize the Zillow API to fetch real estate data and implement ml to analyze deals.
add the ability to follow specific predefined user voice prompts to click windows, open links, type in fields, interact with UIs and apps, edit and crop, adjust settings like brightness and volume, etc. based on voice commands.
add knowledge bases and retrieval augmented generation from custom knowledge in a vector database or graph database with a semantic layer.
vad (voice activity detection) for bot voice and background noice cancellation, or a way to filter out the bot's own voice from the user's input without needing rule-based on/off of the mic input while the bot talks.
integrate it with a bluetooth printer.
add the ability to conduct legal research with websites like casetext, lexisnexis, westlaw, docketbird, pacer, bloomberg law, bna, fastcase, bestlaw, case text, casecheck, case notebook.
more unit tests.
input validations for each function.
performance optimization.
code profiling to identify bottlenecks.
additional secutiry measures - encryption, access control, input sanitation, etc.
to modularize the "chatbot tools initial conversation" we can implement args into the tools and then have the "inquiry" function get user input for each arg in a loop - for arg in args: arg = input("what is your value for " + arg + "?") - then we can pass the args into the tool function.
give the bot the ability to do CRUD operations on the data_store variable in the tools class.
implement asyncio to run the speech recognition and speech output in separate threads so that the bot can listen while it speaks.
implement cachching for speed increase.
```


## CURRENT SPRINT DETAILS:

```bash
the speech timeout settings are still a bit clunky with room for improvement.
occasionaly, the bot is hearing its own output which can interfere with the user input.
building the ui in flet.
add functionality for the agents to operate on the ne04j graph database.
```

## COMPLETION LOG / COMMIT MESSAGES:

```bash
0.1.1 - 2023-11-30 - added google search, wikipedia search, and wolfram alpha query

0.1.1 - 2023-12-01 - note taking and note recall added

0.1.1 - 2023-12-01 - moved speech output to a separate thread so that in the future the bot can listen while speaking.

0.1.1 - 2023-12-02 - added more user details in the .env file to personalize the output in various functions.

0.1.1 - 2023-12-03 - speech queue and speech manager have been implemented to prevent the bot from trying to say multiple things at once.

0.1.1 - 2023-12-03 - wolfram alpha function improved to consolidate specified pods from the api call rather than just the first pod.

0.1.1 - 2023-12-03 - screenshot function added.

0.1.1 - 2023-12-03 - verbal translation function added.

0.1.1 - 2023-12-03 - verbal translation function improved with native accents for translated speech.

0.1.1 - 2023-12-03 - added the youtube search function.

0.1.1 - 2023-12-04 - finalized the spoken 4 day weather forecast function.

0.1.1 - 2023-12-08 - added the initial 'voice to move cursor' and 'voice to click functions' via pyautogui.

0.1.1 - 2023-12-09 - added stock report function.

0.1.1 - 2023-12-09 - added stock recommendation function.

0.1.1 - 2023-12-10 - moved all stock functions into a finance class.

0.1.1 - 2023-12-20 - began testing chat functionality with gemini rather than chatgpt with good success.

0.1.1 - 2023-12-21 - simplified the none handling in the main loop.

0.1.1 - 2023-12-21 - added the ability to enter a chat sub-loop with gemini chatbot by saying "robot, call gemini".

0.1.1 - 2023-12-21 - fixed a bug where the speech recognizer was retuurning 'None' vs None for unrecognized speech input.

0.1.1 - 2023-12-22 - installed auto py to exe and docker in anticipation of building a standalone app (tbd on containerization choice).

0.1.1 - 2023-12-25 - moved the activation word from hard-coded 'robot' into a user-defined variable in the .env file.

0.1.1 - 2023-12-30 - downloaded new voices for the speech interface.

0.1.2 - 2024-01-01 - removed the redundant second parsing attempt in the speech parsing function and simplified the error handling there.

0.1.2 - 2024-01-01 - removed the obsolete standby and reset code blocks to make space for a better future reset feature.

0.1.2 - 2024-01-01 - moved the mouse movement and clicking controls into a more streamlined function.

0.1.2 - 2024-01-01 - added verbal password check when the app runs.

0.1.2 - 2024-01-01 - moved the gemini chat loop into its own function.

0.1.2 - 2024-01-01 - integrated a google custom search engine and LLM search assistant agent with brief analysis of results.

0.1.2 - 2024-01-02 - added a prompt template into the gemini chat initialization asking for good output for a tts app.

0.1.2 - 2024-01-02 - added some additional prompt template steps into the search assistant chatbot.

0.2.1 - 2024-01-03 - re-built the neural nine chatbot that uses intents.json and modernized the tensorflow imports in the training module.

0.2.1 - 2024-01-04 - implemented function calling and stt audio recognition and tts bot output for the neural network based chatbot.

0.2.1 - 2024-01-04 - implemented a function that generates an intent json object for any message interaction the bot doesn't recognize.

0.2.1 - 2024-01-04 - imported and called the chatbot_training module at the top of the chatbot_app module.

0.2.1 - 2024-01-05 - improved mouse control function (responds to up down left right + north south east west) after migrating to v 0.2.1.

0.2.1 - 2024-01-05 - added a function to run diagnostics on the codebase with inspect and then call in the llm as a pair programmer copilot.

0.2.2 - 2024-01-05 - trimmed out commented old functions and migrated more tool methods from 0.1.2 to 0.2.2.

0.2.2 - 2024-01-05 - added a function for the llm to simply read in the codebase and stand by as a pair programmer.

0.2.2 - 2024-01-05 - added more thorough docstrings to all classes and methods.

0.2.2 - 2024-01-06 - added class level data_store to ChatBotTools to store data rendered by tools for access by the llm agent.

0.2.2 - 2024-01-07 - migrated the remaining functions from 0.1.2 to 0.2.2.

0.2.2 - 2024-01-12 - implemented a neo4j graph database with the intention of turning this into the bot's full knowledge base.

0.2.3 - 2024-01-13 - removed t5 from the generate_json_intent function along with the generate_variations helper function.

0.2.4 - 2024-01-13 - added a rudimentary ui with a mic on/off button using flet.

0.2.5 - 2024-01-17 - added a chat log and visibility to the data_store to the flet UI.

0.2.5 - 2024-01-17 - placed the entire flet UI except ui_main() within a ChatBotUI class.

0.2.5 - 2024-01-17 - added __init__.py and main.py to the src directory to run the UI.

0.2.5 - 2024-01-25 - added a .gitignore'd user_persona.py file containing dictionaries of user details by category (movie, music, book preferences, general interests, etc.) to be used by the chatbot for more calibrated responses. Dictionaries can contain anything and can be passed individually or together in a prompt template.

0.2.5 - 2024-01-27 - focusing on the in-line pair programmer agent and neo4j graph database.

0.3.0 - 2024-04-17 - today i'm focusing on building a recipe tool in a tools subdirectory to test routing and function calling before moving the majority of the tools from the original monolith into their own respective files for better modularity and maintainability.

0.0.1 - 2024-10-17 - moved all source code over from hey_data to light_ai and divided monoloth app into modules. created new venv and remote repo.

0.0.1 - 2024-10-17 - created tests for model training.
```
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
