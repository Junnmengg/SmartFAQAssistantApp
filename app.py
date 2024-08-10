import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load dataset
data = pd.read_csv('qa_dataset_with_embeddings.csv')

# Check if 'Question_Embedding' already exists
if 'Question_Embedding' not in data.columns:
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use the same model for consistency

    # Generate embeddings for all questions in the dataset
    data['Question_Embedding'] = data['Question'].apply(lambda x: model.encode(x).tolist())
    embeddings = np.vstack(data['Question_Embedding'].values)
    
    # Optionally save the embeddings back to the CSV file
    data.to_csv('qa_dataset_with_embeddings.csv', index=False)
else:
    # Load the pre-calculated embeddings
    data['Question_Embedding'] = data['Question_Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    embeddings = np.vstack(data['Question_Embedding'].values)

# Load the embedding model for user questions
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
    if user_question.strip() == "":
        st.warning("Please enter a question before searching.")
    else:
        # Generate the user embedding and reshape
        user_embedding = model.encode(user_question).astype(np.float32).reshape(1, -1)
        st.write("User embedding shape:", user_embedding.shape)  # Debugging line
        st.write("Embeddings shape:", embeddings.shape)  # Debugging line

        # Calculate cosine similarity
        similarities = cosine_similarity(user_embedding, embeddings)[0]  # Use the reshaped embedding
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


