#!/usr/bin/env python3
"""
Setup script for downloading required NLP models and data.
Run this after installing requirements.txt
"""

import subprocess
import sys
import nltk

def download_spacy_model():
    """Download the English language model for spaCy."""
    print("Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading spaCy model: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data."""
    print("\nDownloading NLTK data...")
    try:
        # Download essential NLTK data
        nltk_downloads = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        
        for item in nltk_downloads:
            print(f"  Downloading {item}...")
            nltk.download(item, quiet=True)
        
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False
    return True

def main():
    """Main setup function."""
    print("Setting up NLP models and data...\n")
    
    success = True
    
    # Download spaCy model
    if not download_spacy_model():
        success = False
    
    # Download NLTK data
    if not download_nltk_data():
        success = False
    
    if success:
        print("\n✓ All NLP models and data downloaded successfully!")
        print("\nYou can now run the intent classification system.")
    else:
        print("\n✗ Some downloads failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
