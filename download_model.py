"""
Helper script to download and save a model to the models directory.
This script can be run separately to prepare models for use with the API.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name, output_dir):
    """Download a model and its tokenizer to the specified directory"""
    print(f"Downloading model {model_name} to {output_dir}...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")
    
    # Download and save the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    print("Download completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face model")
    parser.add_argument("model_name", type=str, help="Name or path of the model on Hugging Face")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save the model")
    
    args = parser.parse_args()
    
    # Create a subdirectory with the model name
    model_dir = os.path.join(args.output_dir, args.model_name.split("/")[-1])
    
    download_model(args.model_name, model_dir)
    
    print(f"Model is now available at {model_dir}")
    print(f"You can use this path with the /load_model endpoint.")