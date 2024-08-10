import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    df['Question_Embedding'] = df['Question_Embedding'].apply(eval).apply(np.array)
    return df

df = load_data()

# Load the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("Heart, Lung, and Blood Health FAQ Assistant")

# Initialize session state
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

user_question = st.text_input("Ask your health-related question:", value=st.session_state.user_question)

col1, col2 = st.columns(2)
with col1:
    search_button = st.button("Search", key="search")
with col2:
    clear_button = st.button("Clear", key="clear")

if clear_button:
    st.session_state.user_question = ""
    st.experimental_rerun()

if search_button and user_question:
    # Generate embedding for the user's question
    user_embedding = model.encode([user_question])[0]

    # Calculate cosine similarity
    similarities = cosine_similarity([user_embedding], df['Question_Embedding'].tolist())[0]

    # Find the most similar question
    max_similarity = np.max(similarities)
    max_similarity_index = np.argmax(similarities)

    # Set a threshold for similarity
    threshold = 0.7

    if max_similarity >= threshold:
        answer = df.iloc[max_similarity_index]['Answer']
        st.subheader("Answer:")
        st.write(answer)
        st.write(f"Similarity score: {max_similarity:.2f}")

        # Add a rating feature
        st.write("Was this answer helpful?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Yes", key="yes"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 No", key="no"):
                st.error("We're sorry the answer wasn't helpful. We'll work on improving it.")
        with col3:
            if st.button("😐 Somewhat", key="somewhat"):
                st.warning("Thank you for your feedback. We'll try to improve the answer.")
    else:
        st.warning("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")

# Display common FAQs
st.sidebar.header("Common FAQs")
faq_search = st.sidebar.text_input("Search FAQs:", key="faq_search")
filtered_df = df[df['Question'].str.contains(faq_search, case=False, na=False)]
for i, (_, row) in enumerate(filtered_df.iterrows()):
    with st.sidebar.expander(row['Question'], key=f"faq_{i}"):
        st.write(row['Answer'])

# Update session state
st.session_state.user_question = user_question
# Clear button to reset the input field
if st.button("Clear"):
    st.experimental_rerun()


