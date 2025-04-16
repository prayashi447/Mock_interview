import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Initialize encoder
encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Load existing data
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
df = pd.read_csv(DATA_DIR / "raw/questions.csv")

with open(PROCESSED_DIR / "metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

def regenerate_files():
    print("=== Regenerating Embedding Files ===")
    
    # 1. Fix categories.npy
    unique_categories = sorted(set(m['Category'] for m in metadata['records']))
    print(f"\nFound {len(unique_categories)} categories:")
    for cat in unique_categories:
        print(f"- {cat}")
    
    categories_emb = encoder.encode(unique_categories)
    np.save(PROCESSED_DIR / "categories.npy", categories_emb)
    print("\n✅ Regenerated categories.npy with shape:", categories_emb.shape)
    
    # 2. Verify question embeddings
    if 'questions.npy' not in [f.name for f in PROCESSED_DIR.iterdir()]:
        print("\nGenerating question embeddings...")
        questions_emb = encoder.encode(df['Question'].tolist())
        np.save(PROCESSED_DIR / "questions.npy", questions_emb)
        print("✅ Created questions.npy with shape:", questions_emb.shape)
    
    # 3. Update metadata with proper category indices
    category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    for record in metadata['records']:
        record['Category_Encoded'] = category_to_idx[record['Category']]
    
    with open(PROCESSED_DIR / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    print("\n✅ Updated metadata with proper category encodings")
    
    # 4. Verify answer embeddings if needed
    if 'answers.npy' not in [f.name for f in PROCESSED_DIR.iterdir()]:
        print("\nGenerating answer embeddings...")
        answers_emb = encoder.encode(df['Answer'].tolist())
        np.save(PROCESSED_DIR / "answers.npy", answers_emb)
        print("✅ Created answers.npy with shape:", answers_emb.shape)

    print("\n=== Verification ===")
    print("1. categories.npy shape:", np.load(PROCESSED_DIR / "categories.npy").shape)
    print("2. Metadata sample:", metadata['records'][0])
    print("3. Category mapping:", {k: v for k, v in list(category_to_idx.items())[:3]})

if __name__ == "__main__":
    regenerate_files()