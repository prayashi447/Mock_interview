import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = DATA_DIR / "raw/questions.csv"

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Initialize model
encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

def generate_embeddings():
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Generate embeddings
    print("Generating question embeddings...")
    question_embeddings = encoder.encode(
        df['Cleaned_Question'].tolist(),
        normalize_embeddings=True
    )
    
    print("Generating answer embeddings...")
    answer_embeddings = encoder.encode(
        df['Cleaned_Answer'].tolist(),
        normalize_embeddings=True
    )
    
    # Generate category embeddings
    print("Generating category embeddings...")
    unique_categories = df['Category'].unique()
    category_embeddings = encoder.encode(
        unique_categories.tolist(),
        normalize_embeddings=True
    )
    
    # Create metadata with encoded categories
    label_encoder = LabelEncoder()
    category_encoded = label_encoder.fit_transform(df['Category'])
    
    metadata = []
    for i, row in df.iterrows():
        metadata.append({
            "ID": row['ID'],
            "Category": row['Category'],
            "Category_Encoded": int(category_encoded[i]),
            "Difficulty": row['Difficulty'],
            "Original_Index": i
        })
    
    # Save files
    print("Saving embeddings...")
    np.save(PROCESSED_DIR / "questions.npy", question_embeddings)
    np.save(PROCESSED_DIR / "answers.npy", answer_embeddings)
    np.save(PROCESSED_DIR / "categories.npy", category_embeddings)
    
    with open(PROCESSED_DIR / "metadata.pkl", 'wb') as f:
        pickle.dump({
            "records": metadata,
            "category_mapping": dict(zip(
                label_encoder.transform(unique_categories),
                unique_categories
            )),
            "category_embeddings": category_embeddings
        }, f)
    
    print(f"Embeddings generated successfully in {PROCESSED_DIR}")

if __name__ == "__main__":
    generate_embeddings()