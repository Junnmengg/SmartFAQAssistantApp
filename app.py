import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load dataset and embeddings
data = pd.read_csv('qa_dataset_with_embeddings.csv')
data['Question_Embedding'] = data['Question_Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
embeddings = np.vstack(data['Question_Embedding'].values)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app
st.title('Heart, Lung, and Blood Health FAQ Assistant')

# Section for common FAQs
st.sidebar.header("Common FAQs")
common_faqs = data.sample(5)  # Display 5 random FAQs
for idx, row in common_faqs.iterrows():
    st.sidebar.write(f"**Q:** {row['Question']}")
    st.sidebar.write(f"**A:** {row['Answer']}")
    st.sidebar.write("---")

# User input section
user_question = st.text_input("Ask your question about heart, lung, or blood health:")
search_button = st.button("Search for Answer")

# Initialize variables to store results
answer_displayed = False
similarity_score = 0.0

if search_button:
    user_embedding = model.encode(user_question)
    similarities = cosine_similarity([user_embedding], embeddings)[0]
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    threshold = 0.7  # Adjust this value based on testing

    if best_score >= threshold:
        answer = data.iloc[best_index]['Answer']
        answer_displayed = True
        similarity_score = best_score
        st.success(f"**Answer:** {answer}")
        st.info(f"**Similarity Score:** {similarity_score:.2f}")
    else:
        st.error("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")

# Allow users to rate the answer's helpfulness
if answer_displayed:
    rating = st.radio("Was this answer helpful?", options=["Yes", "No"])
    if rating:
        st.write(f"Thank you for your feedback! You rated this answer as: {rating}")

# Clear button to reset the input field
if st.button("Clear"):
    st.experimental_rerun()

