import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer  # For embedding generation
from sklearn.metrics.pairwise import cosine_similarity

# Load data and embeddings
data = pd.read_csv("qa_dataset_with_embeddings.csv")
question_embeddings = np.array(data['Question_Embedding'].tolist())

# Choose an embedding model (you can experiment with different models)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_answer(user_question):
    # Generate embedding for user question
    user_embedding = model.encode(user_question)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding.reshape(1, -1), question_embeddings)
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[0][most_similar_index]
    
    # Set a similarity threshold (adjust as needed)
    threshold = 0.7
    
    if similarity_score > threshold:
        return data['Answer'][most_similar_index], similarity_score
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", None

def main():
    st.title("Health FAQ Assistant")
    
    user_question = st.text_input("Ask your question:")
    
    if st.button("Submit"):
        answer, similarity = get_answer(user_question)
        st.write(answer)
        if similarity:
            st.write(f"Similarity Score: {similarity:.2f}")

if __name__ == "__main__":
    main()
