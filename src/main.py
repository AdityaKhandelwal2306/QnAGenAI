from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the input file
data = pd.read_csv('placement-questions-excel.csv')

# Load LLaMA Instruct LLM for Question Generation (Correct Model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
question_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS Index on GPU
embedding_dim = 384  # Depends on the embedding model
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Vector Database to Store Metadata
metadata_store = []

# Helper Function: Add to FAISS
def add_to_faiss(index, embedding, metadata):
    index.add(np.array([embedding]))
    metadata_store.append(metadata)

# Helper Function: Generate Questions
# Helper Function: Generate Questions
def generate_questions_llama(question, context, num_questions=4):
    """
    Generates multiple questions based on the given input using Llama-3.1-8B-Instruct.

    Args:
        question (str): The original question to inspire the generated questions.
        context (str): Additional information to contextualize the generated questions.
        num_questions (int): Number of questions to generate.
    Returns:
        list: A list of generated questions.
    """
    # Define the input prompt for the model
    prompt = (
        f"You are a language model specializing in generating diverse, high-quality questions.\n\n"
        f"I need {num_questions} distinct questions derived from the following details:\n"
        f"**Original Question**: {question}\n"
        f"**Context**: {context}\n\n"
        "Ensure that each question is complete, grammatically correct, meaningful, and follows a logical sequence. "
        "The questions should vary in phrasing, focus, or approach while staying relevant to the given details."
    )

    # Use the pipeline for question generation
    responses = question_generator(
        prompt,
        max_length=300,
        num_return_sequences=num_questions,
        temperature=0.4
    )
    
    # Extract and return the generated questions
    return [response['generated_text'].strip() for response in responses]



# Helper Function: Calculate Similarity
def calculate_similarity(original_question, generated_question):
    # Create embeddings for both original and generated questions
    original_emb = embedding_model.encode([original_question])
    generated_emb = embedding_model.encode([generated_question])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(original_emb, generated_emb)
    return similarity[0][0]

# Process Each Row
output_data = []
for _, row in data.iterrows():
    # Create the full context from all relevant columns
    context = (
        f"Degree: {row['Degree']}, Role: {row['Role']}, Section: {row['Section']}, "
        f"Proficiency Level: {row['Proficiency Level']}, Options: {row['Options']}, "
        f"Correct Answer: {row['Correct Answer']}, Explanation: {row['Explanation']}"
    )
    
    # Create Embedding
    embedding = embedding_model.encode(context)
    
    # Add to Vector Database
    add_to_faiss(faiss_index, embedding, {"question": row['Question'], "context": context})
    
    # Generate Questions
    new_questions = generate_questions_llama(row['Question'], context)
    
    # For each generated question, calculate similarity with the original question
    for q in new_questions:
        similarity_score = calculate_similarity(row['Question'], q)
        
        # Append the generated question and similarity score to the output
        output_data.append({
            "Generated Question": q,
            "Options": row['Options'],
            "Correct Answer": row['Correct Answer'],
            "Explanation": row['Explanation'],
            "Similarity Score": similarity_score
        })

# Save Output to CSV
output_df = pd.DataFrame(output_data)
output_df.to_csv("expanded_questions.csv", index=False)
