import os
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from preprocessor import text_cleaner

# Load the model and tokenizer from your saved directory
saved_directory = os.path.join(os.getcwd(), "T5-Base-Model-FineTuned-by-RM-Rev2")
loaded_model = TFAutoModelForSeq2SeqLM.from_pretrained(saved_directory)
loaded_tokenizer = AutoTokenizer.from_pretrained(saved_directory)

# Summarized text length can be changed here
MIN_TARGET_LENGTH = 20  # Minimum length of the output by the model
MAX_TARGET_LENGTH = 448  # Maximum length of the output by the model

# The document goes here
document = "i made careless mistake when ordering this coffee. i really prefer the whole bean coffee, but ordered the ground version by mistake. the only reason that i didn't give this ground coffee stars is because, in my opinion, the whole bean version makes fresher better-tasting cup. this ground version still makes very smooth, flavorful cup of coffee. i just think grinding the whole bean is bit more flavorful, excellent coffee"

# Pre-processing the document
# clean the text

cleaned_texts = text_cleaner(document, 1)
print(cleaned_texts)

# Tokenize the document
inputs = loaded_tokenizer(cleaned_texts, return_tensors="tf", truncation=True)

# 1 print: it has it is just very difficult to do all and it takes time
# 2 print: it just summarizes the text and it takes time to do it all.
# 3 print: if you do not know what it will do it will be too hard to do it

# Generate summary IDs without specifying max_length
summary_ids = loaded_model.generate(
    inputs["input_ids"],
    min_length=MIN_TARGET_LENGTH,  # You can change this value according to your needs
    max_length=MAX_TARGET_LENGTH,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True,
)

# Decode the summary IDs
summary = loaded_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary
print("Summary:")
print(summary)