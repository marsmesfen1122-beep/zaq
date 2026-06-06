import streamlit as st
import pandas as pd
import sqlite3
import os

# Import functions/classes from the modularized files
from engine import AIEngine # Assuming AIEngine is in engine.py
from utils import (
    multilingual_search, multilingual_chatbot,
    classify_difficulty, generate_tags, enrich_book,
    translate_text, detect_language, SUPPORTED_LANGUAGES,
    ask_ai
)
# from auth import login, create_token # Uncomment if authentication is added to the app

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Library System 📚",
    page_icon="📚",
    layout="wide"
)

# =========================
# CUSTOM STYLE
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #4B8BBE;
}
.sub-title {
    font-size: 18px;
    color: gray;
}
.book-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #f7f7f7;
    margin-bottom: 12px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='main-title'>📚 AI Library System Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Smart Multilingual Book Recommendation System</div>", unsafe_allow_html=True)

st.divider()

# =========================
# LOAD DATA & AI ENGINE
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("google_books_dataset.csv")
    # Fill NaN values for columns used in utilities to prevent errors
    df['description'] = df['description'].fillna('')
    df['ratings_count'] = df['ratings_count'].fillna(0)
    df['average_rating'] = df['average_rating'].fillna(0)
    df['page_count'] = df['page_count'].fillna(0)
    df['authors'] = df['authors'].fillna('Unknown Author')
    df['categories'] = df['categories'].fillna('Uncategorized')
    return df

df = load_data()

@st.cache_resource
def get_ai_engine(dataframe):
    # AIEngine will build its own index from the dataframe
    return AIEngine(dataframe)

ai_engine = get_ai_engine(df)

# Placeholder for RandomForestRegressor model if predict_popularity is used in the app
# (Not currently in the dashboard, but good to note)
# from sklearn.ensemble import RandomForestRegressor
# @st.cache_resource
# def get_popularity_model(dataframe):
#     features = dataframe[['average_rating', 'ratings_count']].fillna(0)
#     target = dataframe['ratings_count']
#     model = RandomForestRegressor()
#     model.fit(features, target)
#     return model
# popularity_model = get_popularity_model(df)


# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔎 Search & Filters")

# Language selection for multilingual search/chatbot
lang_options = list(SUPPORTED_LANGUAGES.keys())
lang = st.sidebar.selectbox(
    "Select Language",
    lang_options,
    format_func=lambda x: SUPPORTED_LANGUAGES[x].capitalize()
)

category = st.sidebar.selectbox(
    "Category",
    ["All"] + list(df["categories"].dropna().unique()[:30])
)

author = st.sidebar.text_input("Author Filter (English)")

title_search = st.sidebar.text_input("Title Search (English)")

st.sidebar.divider()
st.sidebar.info("🚀 AI Library System Dashboard")

# =========================
# FILTER DATA (Local filtering for display)
# For multilingual search, the actual search uses the AI engine.
# This part is for displaying/filtering the *original* dataframe.
# =========================
filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[filtered_df["categories"] == category]

if author:
    filtered_df = filtered_df[filtered_df["authors"].str.contains(author, na=False, case=False)]

if title_search:
    filtered_df = filtered_df[filtered_df["title"].str.contains(title_search, case=False, na=False)]


# =========================
# KPI ANALYTICS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("📚 Total Books", len(df))
col2.metric("🔎 Results Found", len(filtered_df))
col3.metric("📂 Categories", df["categories"].nunique())

st.divider()

# =========================
# BOOK DISPLAY (CARD STYLE UI)
# Initial display is based on local filters, not multilingual search yet
# =========================
st.subheader("📖 Book Results (Filtered by sidebar)")

if len(filtered_df) == 0:
    st.warning("No books found. Try different filters.")
else:
    # Display top 25 books from the filtered_df
    for i, row in filtered_df.head(25).iterrows():
        # Example of using a utility function
        # difficulty = classify_difficulty(row['page_count'], row['ratings_count'])
        # tags = generate_tags(row)

        st.markdown(f"""
        <div class="book-card">
            <h4>📘 {translate_text(row['title'], target=lang)}</h4>
            <p>👤 <b>Author:</b> {translate_text(row['authors'], target=lang)}</p>
            <p>📚 <b>Category:</b> {translate_text(row['categories'], target=lang)}</p>
            <!-- <p>Difficulty: {difficulty}</p> -->
            <!-- <p>Tags: {', '.join(tags)}</p> -->
        </div>
        """, unsafe_allow_html=True)

# =========================
# MULTILINGUAL SEARCH & CHATBOT
# =========================
st.subheader("🌐 Multilingual AI Search")
search_query_multilingual = st.text_input(
    f"Search in {SUPPORTED_LANGUAGES[lang].capitalize()}",
    key="multilingual_search_input"
)

if search_query_multilingual:
    st.info(f"Searching for '{search_query_multilingual}' in {SUPPORTED_LANGUAGES[lang].capitalize()}...")
    # Call multilingual_search from utils.py, passing the ai_engine.search method
    multilingual_results = multilingual_search(
        search_query_multilingual,
        user_lang=lang,
        search_books_func=ai_engine.search
    )

    if multilingual_results:
        st.write("Multilingual Search Results:")
        for r in multilingual_results:
            st.markdown(f"""
            <div class="book-card">
                <h4>📘 {r['title']}</h4>
                <p>👤 <b>Author:</b> {r['authors']}</p>
                <p>📚 <b>Category:</b> {r['category']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No multilingual search results found.")

st.divider()

st.subheader("💬 Multilingual AI Assistant")

user_chatbot_input = st.text_input(
    f"Ask something about books (in {SUPPORTED_LANGUAGES[lang].capitalize()}):",
    key="chatbot_input"
)

if user_chatbot_input:
    st.info("AI is processing your request...")

    # Placeholder warning for OpenAI API key
    # Note: os.getenv("OPENAI_API_KEY") might be used if key is set as env var for Colab secrets
    # For this demonstration, we check for the literal string in utils.py
    try:
        # A more robust check might involve actually trying to call ask_ai or checking the content of utils.py programmatically
        # For now, a simple check if the default placeholder is still there.
        # This check is heuristic and might not cover all cases if user changes utils.py but not the key.
        with open('utils.py', 'r') as f:
            utils_content = f.read()
        if "api_key=\"YOUR_API_KEY\"" in utils_content:
            st.warning("Please set your OpenAI API key in `utils.py` (replace `\"YOUR_API_KEY\"`) to enable the full AI assistant features. Using basic chatbot for now.")
    except FileNotFoundError:
        st.error("Could not find `utils.py` to check OpenAI API key status.")

    # Call multilingual_chatbot from utils.py, passing the ai_engine.search method
    chatbot_response = multilingual_chatbot(
        user_chatbot_input,
        lang=lang,
        search_books_func=ai_engine.search
    )
    st.success(chatbot_response)


# =========================
# FOOTER
# =========================
st.divider()
st.caption("🚀 Built with Streamlit | AI Library System | Final Year Project")

# =========================
# IMPORTANT: PLACEHOLDER REMINDERS
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Important Setup Notes 🚨")
st.sidebar.info("Remember to replace `YOUR_NGROK_AUTH_TOKEN` in the ngrok setup cell with your actual ngrok token for public deployment.")
st.sidebar.info("Replace `YOUR_API_KEY` in `utils.py` with your OpenAI API key for full AI assistant functionality.")
st.sidebar.info("Update `send_email` function in `utils.py` with your actual email credentials to enable email notifications.")
