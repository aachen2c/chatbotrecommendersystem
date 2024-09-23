import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
from gensim.models import KeyedVectors
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from location_filter import get_current_location, filter_events_by_proximity
import os
import subprocess

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyBr3XahdwUNipwIzXYtrpSTgzO7vQcxBe0'

# Load models and data once and store in session state
if "word2vec_model" not in st.session_state:
    st.session_state["word2vec_model"] = api.load('word2vec-google-news-300')

if "nlp" not in st.session_state:
    st.session_state["nlp"] = spacy.load('en_core_web_sm')

word2vec_model = st.session_state["word2vec_model"]
nlp = st.session_state["nlp"]

# Precompute proximity embeddings once
if "proximity_embeddings" not in st.session_state:
    proximity_phrases = ['around here', 'near me', 'local', 'close to me', 'close to here', 'nearby', 'in my area', 'close by', 'in the area', 'close to my location']
    st.session_state["proximity_embeddings"] = [np.mean([word2vec_model[word] for word in phrase.split() if word in word2vec_model], axis=0) for phrase in proximity_phrases]

proximity_embeddings = st.session_state["proximity_embeddings"]

# Function to generate embeddings using Word2Vec
def get_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Word2Vec-based keyword extractor using spaCy
class Word2VecKeywordExtractor:
    def __init__(self, model):
        self.model = model

    def extract_keywords(self, conversation_history):
        doc = nlp(conversation_history)
        return [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and not token.is_stop and token.is_alpha]

# Load database once and store in session state
if "events" not in st.session_state:
    conn = sqlite3.connect('events.db')
    st.session_state["events"] = pd.read_sql('SELECT * FROM events', conn)
    events = st.session_state["events"]
    events['embedding'] = events['combined_features'].apply(lambda x: get_embedding(x, word2vec_model))
    events['start_date'] = pd.to_datetime(events['start_date'])  # Convert start_date to datetime
    st.session_state["events"] = events
    conn.close()

if "attendance" not in st.session_state:
    st.session_state['attendance'] = pd.read_csv(r'attendance.csv')
    #attendance_df = st.session_state['attendance']

# Function to filter and recommend events
def getrecs_content(events, interactions, userID, N, input_context_words=None, weight_history=0.3, weight_input=0.7, nearme=False):
    user_event_matrix = interactions.pivot_table(index='userID', columns='eventID', aggfunc='size', fill_value=0)
    # Load user profiles once and store in session state
    if "df_user_profiles" not in st.session_state:
        users_features = pd.merge(interactions, events, left_on='eventID', right_on='id')[['userID', 'eventID', 'combined_features']]
        df_user_profiles = users_features.groupby('userID')['combined_features'].apply(lambda x: ' '.join(x)).reset_index()
        df_user_profiles['embedding'] = df_user_profiles['combined_features'].apply(lambda x: get_embedding(x, word2vec_model))
        st.session_state["df_user_profiles"] = df_user_profiles
    else:
        df_user_profiles = st.session_state["df_user_profiles"]

    # User embedding and context combination
    user_idx = df_user_profiles[df_user_profiles['userID'] == userID].index[0]
    user_profile_embedding = df_user_profiles['embedding'].iloc[user_idx]

    if input_context_words:
        input_context_embedding = get_embedding(' '.join(input_context_words), word2vec_model)
        combined_user_embedding = (weight_history * user_profile_embedding) + (weight_input * input_context_embedding)
    else:
        combined_user_embedding = user_profile_embedding

    if nearme:
        user_lat, user_long = get_current_location()
        events = filter_events_by_proximity(events, user_lat, user_long)

    # Cosine similarity calculation
    event_embeddings = np.stack(events['embedding'].values)
    cosine_sim = cosine_similarity([combined_user_embedding], event_embeddings).flatten()

    # Filter and sort events based on similarity
    s = user_event_matrix.loc[userID]
    idx = np.argsort(-cosine_sim)
    user_events = pd.merge(s[s == 0], events, left_on='eventID', right_on='id', how='left')
    user_events = user_events[user_events['start_date'] > pd.Timestamp('2024-10-10 00:00:00', tz='UTC')]

    return pd.merge(events.iloc[idx, :1], user_events, left_on='id', right_on='id').head(N)

# Initialize Langchain components
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
memory = ConversationBufferMemory()

# Dynamic conversation prompt template
prompt_template = PromptTemplate(
    input_variables=["history", "user_query", "available_events"],
    template="""
    You are a chatbot helping a user find events they're interested in based on their preferences.
    The user has requested events matching the following query: {user_query}.
    
    Here is a sample of available events:
    {available_events}

    Based on the user's preferences, recommend 3 events that best match, prioritizing sooner events. If no events match or if a specific element of the query \
         cannot be satisfied (e.g., user asked for events in Chicago, but no available events are in Chicago), apologize and suggest alternatives from the list.

    Now respond to the user accordingly. Remember to use full names for months, days, and years. For example, use "January 1st" instead of "Jan 1".
    """
)

prompt_template_refine = PromptTemplate(
        input_variables=["history", "user_query", "available events"],
        template='''
        You are a chatbot providing additional details about an event a user has already been recommended. 
    
        The user is asking about an event from this list that you previously recommended: 
        {available_events}

        Based on their query: {user_query}, find out which event they want to learn more about and describe it to them.

        Please describe the event the user asks about in detail, including date, location, URL, the maximum and minimum prices, and any other interesting facts.
        
        Conversation history:
        {history}

        Now respond to the user accordingly.'''
)

conversation = prompt_template | llm | StrOutputParser()
refine_conversation = prompt_template_refine | llm | StrOutputParser()
keyword_extractor = Word2VecKeywordExtractor(word2vec_model)

# Streamlit Web App
st.title("Chatbot Event Recommender")
# Check if userID is in session state
if "userID" not in st.session_state:
    user_input = st.text_input("Please enter your userID:")
    if st.button("Submit"):
        st.session_state["userID"] = str(user_input)
        st.rerun()  # Rerun the app to reflect the stored userID

# Block further app functionality until userID is provided
if st.session_state.get("userID", 0):
    userID = st.session_state["userID"]
    st.write(f"Welcome, user {userID}!")
    st.write("Chatbot: Hey there! What kind of events are you looking for?")

    user_input = st.text_input("Input your query in the box below, then hit the 'Send' button.")
    refine = st.checkbox("Refine your query?", value=False)

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if st.button("Send"):
        st.session_state["history"].append(f"You: {user_input}")
        if user_input or refine:
            if not refine:
                print(f"You: {user_input}")
                extracted_keywords = keyword_extractor.extract_keywords(user_input)
                user_input_embedding = get_embedding(user_input, word2vec_model)

                # Check if any proximity phrase has a high similarity to user input
                similarity_threshold = 0.7
                nearme = any(np.dot(user_input_embedding, proximity_emb) / (np.linalg.norm(user_input_embedding) * np.linalg.norm(proximity_emb)) > similarity_threshold for proximity_emb in proximity_embeddings)

                # Get event recommendations
                recommended_events = getrecs_content(st.session_state["events"], st.session_state['attendance'], userID, 10, input_context_words=extracted_keywords, weight_history=0.3, weight_input=0.7, nearme=nearme)
                print(f"Recommendations: {recommended_events[['name']]}")

                # Format the events into a string for the model
                recommended_events_str = "\n".join([f"{e['name']} on {e['start_date']} in {e['city']}: {e['venue_name']}, URL: {e['url']}, maximum price: {e['price_max']} {e['currency']}, minimum price: {e['price_min']} {e['currency']}" for e in recommended_events.to_dict(orient='records')])
                
                # record history
                st.session_state['last_recommended_events'] = recommended_events_str

                # Invoke the language model with Langchain
                response = conversation.invoke({
                    "history": "\n".join(reversed(st.session_state["history"])),
                    "user_query": user_input,
                    "available_events": recommended_events_str
                }).replace('$', r'\$')

                st.write(f"Chatbot: {response}")
                print(f"Chatbot: {response}")
                st.session_state["history"].append(f"Chatbot: {response}")
            else:
                # Refine the existing recommendations, giving more information
                if "last_recommended_events" in st.session_state:
                    # Use last recommended events for refinement
                    last_recommended_events_str = st.session_state["last_recommended_events"]
                    response = refine_conversation.invoke({
                        "history": "\n".join(reversed(st.session_state["history"])),
                    "user_query": user_input,
                    "available_events": last_recommended_events_str
                    }).replace('$', r'\$')
                else:
                    response = "Sorry, but there are no recommendations available to refine. Please uncheck the 'refine' checkbox and type something to start the conversation."
                st.write(f"Chatbot: {response}")
                print(f"Chatbot: {response}")
                st.session_state["history"].append(f"Chatbot: {response}")
        else:
            st.write("Please type something to start the conversation.")
            

    # Display conversation history
    if st.session_state["history"]:
        st.write("Conversation History:")
        for h in reversed(st.session_state["history"]):
            st.write(h)
else:
    st.stop()  # Stops execution until userID is provided