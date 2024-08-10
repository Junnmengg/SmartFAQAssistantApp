import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_answer(user_question, question_embeddings):
    try:
        user_embedding = model.encode(user_question)
        similarities = cosine_similarity(user_embedding.reshape(1, -1), question_embeddings)
        most_similar_index = np.argmax(similarities)
        similarity_score = similarities[0][most_similar_index]
        threshold = 0.7

        if similarity_score > threshold:
            return data['Answer'][most_similar_index], similarity_score
        else:
            return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", None
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Please try again later.", None

# Load data and embeddings
data = pd.read_csv("qa_dataset_with_embeddings.csv")
question_embeddings = np.array(data['Question_Embedding'].tolist())

# Choose an embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample common FAQs
common_faqs = [
    "What is heart disease?",
    "What are the symptoms of lung cancer?",
    "How can I prevent blood clots?"
]

def main():
    st.title("Health FAQ Assistant")

    # Search bar
    search_query = st.text_input("Search FAQs")

    # Common FAQs
    st.header("Common FAQs")
    for faq in common_faqs:
        if search_query.lower() in faq.lower():
            st.write(faq)

    # User question
    user_question = st.text_input("Ask your question:")
    if st.button("Clear"):
        user_question = ""

    if st.button("Submit"):
        answer, similarity = get_answer(user_question, question_embeddings)
        st.write(answer)
        if similarity:
            st.write(f"Similarity Score: {similarity:.2f}")

        # Add answer rating
        rating = st.radio("How helpful was this answer?", ["Very Helpful", "Helpful", "Neutral", "Not Helpful", "Very Not Helpful"])

if __name__ == "__main__":
    main()
