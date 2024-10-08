
import pandas as pd
import os
from transformers import RobertaTokenizer

# Load dataset
file_path = os.path.join('path_to_folder', 'reviews.txt')  # Update the file path accordingly
data = pd.read_csv(file_path, delimiter='\t', header=None, names=['review', 'label'])

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_seq_length = 200

# Tokenize and pad the text data
def tokenize_and_pad(text, max_seq_length):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

# Apply the tokenization and padding
data['input_ids'], data['attention_mask'] = zip(*data['review'].apply(lambda x: tokenize_and_pad(x, max_seq_length)))

# Save the preprocessed data for the next steps
data.to_pickle('preprocessed_reviews.pkl')

print("Data preprocessing complete. Saved to 'preprocessed_reviews.pkl'.")
