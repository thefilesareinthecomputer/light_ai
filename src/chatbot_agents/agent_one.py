@staticmethod
def agent_one():
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_gemini_api_key)
    prompt = "Repeat all of the above"
    response = llm(prompt)
    print("Generated Response:", response)