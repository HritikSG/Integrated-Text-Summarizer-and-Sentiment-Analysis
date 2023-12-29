import os
from nltk.corpus import stopwords
import torch
import re
from bs4 import BeautifulSoup
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TFAutoModelForSeq2SeqLM, AutoTokenizer
from tensorflow.keras.layers import *

# Example text
document = "The OIA is excited to announce virtual events for new international students admitted for the winter semester of 2024. These information sessions are an opportunity for new international students to learn about UM-Dearborn. The OIA will present information about the F-1 Visa application process, important dates, immigration and academic requirements, course registration, and much more."

# Load tokenizer and model
sentiment_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
sentiment_model = RobertaForSequenceClassification.from_pretrained('/Users/ramazanmengi/PycharmProjects/PytorchT5SummarizerFineTuning/finetuned_models/sentiment_model_RoBERTa-4')

# Load the trained model and tokenizer
model_directory = '/Users/ramazanmengi/PycharmProjects/PytorchT5SummarizerFineTuning/finetuned_models/T5-Base-Model-FineTuned-by-RM-Rev5'  # Replace with your model directory
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_directory)

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
common_words = set(stopwords.words('english'))

def capitalize_sentences(text):
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    capitalized_sentences = [s.capitalize() for s in sentences]
    return ' '.join(capitalized_sentences)

def text_cleaner(text, num):
    """
    This function cleans a text by removing everything :)
    """
    # define new string
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)
    newString = re.sub(r'\*\*|\.\.|:\)|:\(', '', newString)  # Remove *, .., :), and :(
    newString = re.sub(r'\.\.\.', '.', newString)  # Replace ... with .
    newString = re.sub(r'!!!', '!', newString)  # Replace !!! with !
    newString = re.sub(r'!!', '!', newString)  # Replace !! with !
    newString = re.sub(r'[\[\]]', '', newString)  # Remove [ and ]

    tokens = newString.split()

    long_words = []
    for i in tokens:
        if len(i) > 1 or i == 'i':
            long_words.append(i)
    return (" ".join(long_words)).strip()

def predict_sentiment(text):
    # Encode the text using the tokenizer
    encodings = sentiment_tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Get input IDs and attention mask
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Get prediction from model
    with torch.no_grad():
        outputs = sentiment_model(input_ids, attention_mask=attention_mask)

    # Get the predicted label (0 or 1)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    # Map the label to sentiment (modify according to your label mapping)
    if predicted_label >= 0.75:
        sentiment = 'Very Positive'
    elif predicted_label >= 0.6:
        sentiment = 'Positive'
    elif predicted_label >= 0.4:
        sentiment = 'Neutral'
    elif predicted_label >= 0.25:
        sentiment = 'Negative'
    else:
        sentiment = 'Very Negative'
    return sentiment, predicted_label

cleaned_texts = text_cleaner(document, 1)

# ----- SENTIMENT ANALYZER ------
sentiment, sentiment_value = predict_sentiment(cleaned_texts)

# ----- TEXT SUMMARIZER ------
def preprocess_text(text, tokenizer, max_input_length=1200):
    """
    Preprocesses the text for summarization.
    """
    prefix = "summarize: "
    inputs = tokenizer(prefix + text, return_tensors="tf", max_length=max_input_length, truncation=True)
    return inputs

def generate_summary(text, tokenizer, model, max_target_length=120):
    """
    Generates a summary using the model.
    """
    inputs = preprocess_text(text, tokenizer)
    summary_ids = model.generate(inputs["input_ids"],
                                 max_length=max_target_length,
                                 length_penalty=2.0,
                                 num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# ----- RESULTS ------
summary = generate_summary(document, tokenizer, model)
capitalized_summary = capitalize_sentences(summary)
print(f"Generated Summary: {capitalized_summary}")
print("Sentiment Score:", sentiment, "With the following value", sentiment_value)
