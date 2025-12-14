# src/text_features.py
import pandas as pd
import os
import pickle
import gzip  # Added for compression
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(
    cleaned_csv="D:\\PYTHON\\Project\\Book_Recommender_System\\data\\processed\\cleaned_books.csv",
    tfidf_pkl="D:\\PYTHON\\Project\\Book_Recommender_System\\models\\tfidf_features.pkl.gz"  # .gz extension
):
    """
    Load cleaned dataset, create TF-IDF features, and save compressed pickle (<25MB).
    """

    # Load cleaned dataset
    df = pd.read_csv(cleaned_csv)
    
    print(f"Loaded dataset with {len(df)} books")
    print(f"Columns: {list(df.columns)}")

    # Verify Cover_URL column exists
    if 'Cover_URL' in df.columns:
        covers_present = len(df[df['Cover_URL'] != ''])
        print(f"Cover URLs present: {covers_present}/{len(df)} ({covers_present/len(df)*100:.1f}%)")
    else:
        print("Warning: Cover_URL column not found in dataset")

    # Ensure no NaNs in important columns
    for col in ['Book', 'Author', 'Description', 'Genres']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    # Create combined text column
    df['combined_text'] = df['Book'] + " " + df['Author'] + " " + df['Description'] + " " + df['Genres']

    # TF-IDF Vectorization (optimized for size)
    print("\nCreating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=4000,      # Reduced from 5000 
        ngram_range=(1, 2),     # Captures bigrams 
        sublinear_tf=True
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Save compressed pickle
    os.makedirs(os.path.dirname(tfidf_pkl), exist_ok=True)
    with gzip.open(tfidf_pkl, 'wb') as f:  # gzip compression
        pickle.dump({
            "tfidf_matrix": tfidf_matrix,
            "vectorizer": tfidf_vectorizer,
            "df": df
        }, f)

    print(f"\nCOMPRESSED TF-IDF features saved to: {tfidf_pkl}")
    print(f"Dataframe includes Cover_URL column: {'Cover_URL' in df.columns}")

    return tfidf_matrix, tfidf_vectorizer, df


if __name__ == "__main__":
    create_tfidf_features()