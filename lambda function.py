import json
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
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(job_titles_df['Job Title'])
    
    query_vector = vectorizer.transform([job_title])
    cosine_similarities = cosine_similarity(query_vector, X)
    most_similar_idx = cosine_similarities.argmax()
    
    matched_title = job_titles_df['Job Title'][most_similar_idx]
    similarity = cosine_similarities[0][most_similar_idx]
    
    return matched_title, similarity

def lambda_handler(event, context):
    # Example job titles dataset
    job_titles = [
        "Software Engineer", "Data Scientist", "Web Developer", 
        "AI Specialist", "Database Administrator", "Machine Learning Engineer",
        "Business Analyst", "Cloud Architect", "DevOps Engineer", "Network Administrator"
    ]
    
    job_titles_df = pd.DataFrame(job_titles, columns=["Job Title"])
    
    # Extract job title from event
    job_title = event['job_title']
    
    # Find most similar job title
    matched_title, similarity = match_job_title(job_title, job_titles_df)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'matched_title': matched_title,
            'similarity_score': similarity
        })
    }
