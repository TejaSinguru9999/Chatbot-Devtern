import os
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents for the chatbot
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"], 
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", " What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the Weather app or Website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": [
            "To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses by categorizing them into fixed (like rent) and variable (like groceries).",
            "One effective budgeting strategy is the 50/30/20 rule: allocate 50% of your income to needs, 30% to wants, and 20% to savings and debt repayment.",
            "To create a budget, follow these steps: Determine your income, list expenses, set financial goals, and adjust your budget as needed."
        ]
    },
]

# Initialize the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
clf.fit(x, tags)

# Define the chatbot function
def chatbot(user_input):
    user_input_vector = vectorizer.transform([user_input])
    predicted_tag = clf.predict(user_input_vector)[0]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I didn't understand that."

# Streamlit app
def main():
    st.title("Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Get user input
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        # Get the chatbot's response
        response = chatbot(user_input.lower())
        
        # Store the conversation
        st.session_state.conversation.append(f"You: {user_input}")
        st.session_state.conversation.append(f"Chatbot: {response}")
    
    # Display the conversation history
    for message in st.session_state.conversation:
        st.write(message)

    # End chat if the response is a goodbye
    if st.session_state.conversation and "goodbye" in st.session_state.conversation[-1].lower():
        st.write("Thank you for chatting with me. Have a great day!")
        st.stop()

if __name__ == "_main_":
    main()