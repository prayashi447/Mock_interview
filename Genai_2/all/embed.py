import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import torch

def generate_interview_embeddings(data_path: str, output_dir: str = "interview_embeddings"):
    """
    Generate and save high-quality embeddings from CSV file
    
    Args:
        data_path: Path to CSV file (e.g., 'data.csv')
        output_dir: Directory to save embeddings
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load CSV file
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Verify required columns exist
    required_columns = ['ID', 'Category', 'Question', 'Answer', 'Difficulty', 'Cleaned_Question', 'Cleaned_Answer']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns. Needed: {required_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        return
    
    # Initialize embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'BAAI/bge-large-en-v1.5'
    model = SentenceTransformer(model_name, device=device)
    print(f"Using {model_name} on {device}")

    # Generate embeddings
    print("Generating question embeddings...")
    question_embeddings = model.encode(
        df['Cleaned_Question'].tolist(),
        batch_size=128 if device == "cuda" else 32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print("Generating answer embeddings...")
    answer_embeddings = model.encode(
        df['Cleaned_Answer'].tolist(),
        batch_size=128 if device == "cuda" else 32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save all artifacts
    print("Saving embeddings...")
    np.save(output_path/"questions.npy", question_embeddings)
    np.save(output_path/"answers.npy", answer_embeddings)
    df.to_csv(output_path/"metadata.csv", index=False)  # Changed to CSV
    
    # Save configuration
    config = {
        "model": model_name,
        "dimension": question_embeddings.shape[1],
        "normalized": True,
        "device_used": device,
        "num_samples": len(df),
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    with open(output_path/"config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Successfully saved embeddings to {output_path}")

# Example usage with CSV file
generate_interview_embeddings("C:/Users/ssspr/Desktop/SEM 6/Project/Genai/data.csv")