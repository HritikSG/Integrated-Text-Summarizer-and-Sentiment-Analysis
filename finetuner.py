from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from hyperparameters import BATCH_SIZE, MAX_EPOCHS
from trainer0 import callbacks, data_collator, preprocess_function
from trainerdataprocessor import get_raw_datasets

# Load your fine-tuned model
model_directory = "T5-Base-Model-FineTuned-by-RM-Rev2"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

file_path = '/Users/ramazanmengi/PycharmProjects/KerasT5TextSummarizerFineTuning/texttosummarydataset2.csv'
new_raw_datasets = get_raw_datasets(file_path)  # Function to get the new dataset

# Preprocess the new dataset
tokenized_new_train_dataset = new_raw_datasets["train"].map(preprocess_function, batched=True)
tokenized_new_test_dataset = new_raw_datasets["test"].map(preprocess_function, batched=True)

new_train_dataset = tokenized_new_train_dataset.to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    collate_fn=data_collator,
)
new_test_dataset = tokenized_new_test_dataset.to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=False,
    collate_fn=data_collator,
)

new_history = model.fit(
    new_train_dataset,
    validation_data=new_test_dataset,
    epochs=MAX_EPOCHS,  # Define this according to your need
    callbacks=callbacks  # You can reuse the callbacks or define new ones
)

new_save_directory = "T5-Base-Model-FineTuned-Amazon+BBC"
model.save_pretrained(new_save_directory)
tokenizer.save_pretrained(new_save_directory)
