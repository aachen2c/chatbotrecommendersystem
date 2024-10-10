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
#'''This document contains the logic for the Streamlit webapp and recommender system. We use the Gemini API w/ LangChain Conv prompts,
#gensim library for google news Word2Vec, and sklearn for computing the cosine similarity between embedded event wordbags.'''

#'''First off, we store the API key as an environment variable. In a public app, we'd need to hide this.'''
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyBr3XahdwUNipwIzXYtrpSTgzO7vQcxBe0'

# Load models and data once and store in session state
#'''By using gensim's API, we can download a pretrained word2vec model and store it in the webapp's local state. This saves us compute and makes our webapp run faster.'''
if "word2vec_model" not in st.session_state:
    st.session_state["word2vec_model"] = api.load('word2vec-google-news-300')

#'''use the spacy library to load in an english language model (for keyword recognition)'''
if "nlp" not in st.session_state:
    st.session_state["nlp"] = spacy.load('en_core_web_sm')

#'''Grab the word2vec and english language models out'''
word2vec_model = st.session_state["word2vec_model"]
nlp = st.session_state["nlp"]

# Precompute proximity embeddings once
#'''For "nearme" detection, we use a simple computation of context-similar phrases. If a search query contains language similar to these phrases, it will trigger nearme'''
if "proximity_embeddings" not in st.session_state:
    proximity_phrases = ['around here', 'near me', 'local', 'close to me', 'close to here', 'nearby', 'in my area', 'close by', 'in the area', 'close to my location']
    st.session_state["proximity_embeddings"] = [np.mean([word2vec_model[word] for word in phrase.split() if word in word2vec_model], axis=0) for phrase in proximity_phrases]

proximity_embeddings = st.session_state["proximity_embeddings"]

# Function to generate embeddings using Word2Vec
#'''For each combined_features string, we generate an embedding using bag-of-words encoding via Word2Vec. We only use words that are in the model.'''
def get_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    mean_embedding = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    norm = np.linalg.norm(mean_embedding)
    return mean_embedding / norm if norm > 0 else mean_embedding

# Word2Vec-based keyword extractor using spaCy
#'''From a given search query, we run the english language model to tokenize the text. We then extract nouns, adjectives, and proper nouns and strip stopwords
#and non-alphanumeric characters.'''
class Word2VecKeywordExtractor:
    def __init__(self, model):
        self.model = model

    def extract_keywords(self, conversation_history):
        doc = nlp(conversation_history)
        return [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and not token.is_stop and token.is_alpha]

# Load database once and store in session state
#'''We load the SQLlite database into the webapp. Open the connection by defining the connection variable. Use pd.read_sql with a query and the connection variable. 
#Then close the connection. We add the embedding into the database as a variable after embedding "combined_features". '''
if "events" not in st.session_state:
    conn = sqlite3.connect('events2.db')
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
#'''This is the main recommender system. We pivot the interactions matrix into a users x events matrix, then generate and load the user profiles by embedding
#features from all the events that they previously attended. We then store these profiles as a table inside our session state. '''
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
    #'''For a given user, we lookup their user profile and grab the index that it's at. Then, we grab the embedding for that user.'''
    user_idx = df_user_profiles[df_user_profiles['userID'] == userID].index[0]
    user_profile_embedding = df_user_profiles['embedding'].iloc[user_idx]
    # user_profile_embedding = df_user_profiles[df_user_profiles['userID'] == userID]['embedding'] ?

    if input_context_words:
        #'''We take the search query keywords and embed them, then we create a weighted embedding between it and the userprofile embeding'''
        input_context_embedding = get_embedding(' '.join(input_context_words), word2vec_model)
        combined_user_embedding = (weight_history * user_profile_embedding) + (weight_input * input_context_embedding)
    else:
        #'''In the absence of a query, we use the user_profile alone'''
        combined_user_embedding = user_profile_embedding

    if nearme:
        #'''If nearme is flagged, we filter the possible events for recommendation by 50km of the user's location'''
        user_lat, user_long = get_current_location()
        events = filter_events_by_proximity(events, user_lat, user_long)

    # Cosine similarity calculation
    #'''We take the cosine similarity of the user/search embeddings and every event embedding in our database'''
    event_embeddings = np.stack(events['embedding'].values)
    cosine_sim = cosine_similarity([combined_user_embedding], event_embeddings).flatten()

    # Filter and sort events based on similarity
    #'''We identify the events that the user has not seen, find those that have not started yet (Oct 10th), and sort them by cosine similarity.
    #We return the first N valid events sorted by cosine similarity.'''
    s = user_event_matrix.loc[userID]
    idx = np.argsort(-cosine_sim)
    user_events = pd.merge(s[s == 0], events, left_on='eventID', right_on='id', how='left')
    user_events = user_events[user_events['start_date'] > pd.Timestamp('2024-10-10 00:00:00', tz='UTC')]

    return pd.merge(events.iloc[idx, :1], user_events, left_on='id', right_on='id').head(N)

# Initialize Langchain components
#'''Using langchain with GeminiFlash, we can make the webapp exclusively speedy. Set temperature to 0 for maximum reproducibility.'''
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
#'''The bread and butter of the chatbot is in the prompt template. Pretty simple stuff -- chatbot reads the query and the valid events and responds accordingly.'''
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

#'''If the user enters into "refining" mode, then the recommender turns off and the chatbot simply continues the conversation based on the last three recommendations.'''
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

#'''LangChain works off of the "conversation" object, which takes a template, a model, and an output parser. 
#In this case, we have two types of conversation prompts, the GeminiFlash model, and a standard string output parser'''
conversation = prompt_template | llm | StrOutputParser()
refine_conversation = prompt_template_refine | llm | StrOutputParser()
keyword_extractor = Word2VecKeywordExtractor(word2vec_model)

#'''This is the frontend for the app itself. Much of it is boilerplate code written by ChatGPT.'''
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
                #'''We extract keywords using the class defined previously in this file, and then embed it.'''
                extracted_keywords = keyword_extractor.extract_keywords(user_input)
                user_input_embedding = get_embedding(user_input, word2vec_model)

                # Check if any proximity phrase has a high similarity to user input
                #'''If the input embedding is similar to any of the proximity phrases, we trigger the nearme flag.'''
                similarity_threshold = 0.7
                nearme = any(np.dot(user_input_embedding, proximity_emb) / (np.linalg.norm(user_input_embedding) * np.linalg.norm(proximity_emb)) > similarity_threshold for proximity_emb in proximity_embeddings)

                # Get event recommendations
                recommended_events = getrecs_content(st.session_state["events"], st.session_state['attendance'], userID, 10, input_context_words=extracted_keywords, weight_history=0.3, weight_input=0.7, nearme=nearme)
                print(f"Recommendations: {recommended_events[['name']]}")

                # Format the events into a string for the model
                #'''Since geminiflash and many other lightweight LLMs reason best over text, we convert the event recommendations table into a string.'''
                recommended_events_str = "\n".join([f"{e['name']} on {e['start_date']} in {e['city']}: {e['venue_name']}, URL: {e['url']}, maximum price: {e['price_max']} {e['currency']}, minimum price: {e['price_min']} {e['currency']}" for e in recommended_events.to_dict(orient='records')])
                
                # record history
                st.session_state['last_recommended_events'] = recommended_events_str

                # Invoke the language model with Langchain
                #'''LangChain "invoke" allows us to query a given conversation with all of its variables. Because Streamlit uses markdown, we have a special case 
                #for dollar signs to prevent them from ruining the formatting.'''
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
                #'''If we're in refinement mode, invoke the refining prompt'''
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