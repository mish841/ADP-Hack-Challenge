import openai

openai.api_key = "your_openai_api_key"

def chatbot(query):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"HR Query: {query}\nAI Response:",
        max_tokens=100,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

st.header("HR Chatbot")
user_input = st.text_input("Ask the chatbot:")
if user_input:
    st.write(chatbot(user_input))