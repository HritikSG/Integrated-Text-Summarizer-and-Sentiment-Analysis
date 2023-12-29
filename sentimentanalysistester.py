import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

# Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('C:/Users/hriti/OneDrive/Desktop/Train_senti/wetransfer_sentiment_model_roberta-3_2023-11-10_2253/sentiment_model_RoBERTa-3')
model.eval()  # Set the model to evaluation mode

def predict_sentiment(text):
    # Encode the text using the tokenizer
    encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Get input IDs and attention mask
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Get prediction from model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted label (0 or 1)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    # Map the label to sentiment (modify according to your label mapping)
    sentiment = 'Positive' if predicted_label == 1 else 'Negative'
    return sentiment

# Example: Get user input and predict sentiment
user_input = input("Enter a sentence to analyze sentiment: ")
predicted_sentiment = predict_sentiment(user_input)
print(f"The predicted sentiment is: {predicted_sentiment}")

# def predict_word_sentiment(sentence):
#     words = sentence.split()
#     word_predictions = {}
#
#     for word in words:
#         # Tokenize the word
#         encodings = tokenizer(word, truncation=True, padding=True, max_length=512, return_tensors="pt")
#         input_ids = encodings['input_ids']
#         attention_mask = encodings['attention_mask']
#
#         # Get prediction from model
#         with torch.no_grad():
#             outputs = model(input_ids, attention_mask=attention_mask)
#
#         # Get the predicted label's score
#         predicted_score = torch.nn.functional.softmax(outputs.logits, dim=1)
#         predicted_score = predicted_score[:, 1].item()  # Assuming label '1' is 'Positive'
#
#         # Add to dictionary
#         word_predictions[word] = predicted_score
#
#     return word_predictions
#
# # Example: Get user input and predict sentiment for each word
# user_input = input("Enter a sentence to analyze sentiment: ")
# word_sentiments = predict_word_sentiment(user_input)
#
# # Print sentiment scores for each word
# for word, score in word_sentiments.items():
#     print(f"Word: '{word}' - Sentiment Score: {score}")

###################################################################

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import nltk
from nltk.corpus import stopwords

# # Download the stopwords dataset from nltk
# nltk.download('stopwords')
#
# # Load tokenizer and model
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
# model = RobertaForSequenceClassification.from_pretrained('C:/Users/hriti/OneDrive/Desktop/Train_senti/wetransfer_sentiment_model_roberta-3_2023-11-10_2253/sentiment_model_RoBERTa-3')
#
# # Get the list of common English words (stopwords)
# common_words = set(stopwords.words('english'))
#
# def predict_word_sentiment(sentence):
#     words = sentence.split()
#     word_predictions = {}
#
#     for word in words:
#         # Check if the word is in the list of common words
#         if word.lower() in common_words:
#             # Assign a neutral sentiment score of 0.5 for common words
#             word_predictions[word] = 0.5
#         else:
#             # Tokenize the word
#             encodings = tokenizer(word, truncation=True, padding=True, max_length=512, return_tensors="pt")
#             input_ids = encodings['input_ids']
#             attention_mask = encodings['attention_mask']
#
#             # Get prediction from model
#             with torch.no_grad():
#                 outputs = model(input_ids, attention_mask=attention_mask)
#
#             # Get the predicted label's score
#             predicted_score = torch.nn.functional.softmax(outputs.logits, dim=1)
#             predicted_score = predicted_score[:, 1].item()  # Assuming label '1' is 'Positive'
#
#             # Add to dictionary
#             word_predictions[word] = predicted_score
#
#     return word_predictions
#
# # Example: Get user input and predict sentiment for each word
# user_input = input("Enter a sentence to analyze sentiment: ")
# word_sentiments = predict_word_sentiment(user_input)
#
# # Print sentiment scores for each word
# for word, score in word_sentiments.items():
#     print(f"Word: '{word}' - Sentiment Score: {score}")

# import torch
# from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
#
# # Load tokenizer and model
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
# model = RobertaForSequenceClassification.from_pretrained(
#     'C:/Users/hriti/OneDrive/Desktop/Train_senti/wetransfer_sentiment_model_roberta-3_2023-11-10_2253/sentiment_model_RoBERTa-3')
# model.eval()  # Set the model to evaluation mode
#
#
# def predict_sentiment(text):
#     # Encode the text using the tokenizer
#     encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
#
#     # Get input IDs and attention mask
#     input_ids = encodings['input_ids']
#     attention_mask = encodings['attention_mask']
#
#     # Get prediction from model
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#
#     # For binary classification models (2 classes)
#     logits = outputs.logits[:, 0].item()  # Extract the score for the first class
#
#     # Apply sigmoid to get a value between 0 and 1
#     sentiment_score = torch.sigmoid(torch.tensor(logits)).item()
#
#     # Categorize sentiment based on thresholds
#     if sentiment_score <= 0.33:
#         sentiment = 'Negative'
#     elif sentiment_score <= 0.67:
#         sentiment = 'Moderate'
#     else:
#         sentiment = 'Positive'
#
#     return sentiment
#
#
# # Example: Get user input and predict sentiment
# user_input = input("Enter a sentence to analyze sentiment: ")
# predicted_sentiment = predict_sentiment(user_input)
# print(f"The predicted sentiment is: {predicted_sentiment}")
