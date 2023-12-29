from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np  # Import NumPy for the array conversion

# Load data
dataset = pd.read_csv("C:/Users/hriti/OneDrive/Desktop/Train_senti/sentiment_data (1).csv")

# Shuffle and reset index
dataset = dataset.sample(frac=1).reset_index(drop=True)

# First, split the data into 20% to be used and 80% to be discarded
used_data, _ = train_test_split(dataset, test_size=0.8, random_state=42)

# From the 20% used_data, split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    used_data['text'], used_data['label'], test_size=0.2, random_state=42)

# Convert to lists
train_texts = train_texts.tolist()
val_texts = val_texts.tolist()

# Convert labels to NumPy arrays
train_labels = train_labels.to_numpy()
val_labels = val_labels.to_numpy()

print("Dataset has been used successfully!")