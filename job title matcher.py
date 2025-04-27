import boto3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Comprehend
comprehend = boto3.client('comprehend')

def analyze_job_title(job_title):
    """Use Amazon Comprehend to detect entities and key phrases."""
    entities = comprehend.detect_entities(Text=job_title, LanguageCode='en')
    key_phrases = comprehend.detect_key_phrases(Text=job_title, LanguageCode='en')
    return entities, key_phrases

def match_job_title(job_title, job_titles_df):
    """Match a given job title with the most similar one using TF-IDF."""
    # Vectorize job titles using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(job_titles_df['Job Title'])
    
    # Find the most similar job title using cosine similarity
    query_vector = vectorizer.transform([job_title])
    cosine_similarities = cosine_similarity(query_vector, X)
    most_similar_idx = cosine_similarities.argmax()
    
    matched_title = job_titles_df['Job Title'][most_similar_idx]
    similarity = cosine_similarities[0][most_similar_idx]
    
    return matched_title, similarity

def main():
    # Load the dataset of job titles
    job_titles_df = pd.read_csv('data/job_titles.csv')
    
    # Example job title to match
    job_title = "Machine Learning Engineer"
    
    # Analyze job title using Comprehend (optional)
    entities, key_phrases = analyze_job_title(job_title)
    print("Entities detected:", entities)
    print("Key Phrases detected:", key_phrases)
    
    # Find the most similar job title
    matched_title, similarity = match_job_title(job_title, job_titles_df)
    print(f"Most similar job title: {matched_title}")
    print(f"Similarity score: {similarity}")

if __name__ == '__main__':
    main()
