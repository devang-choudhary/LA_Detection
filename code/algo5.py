# Import necessary libraries
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from gensim.parsing.preprocessing import preprocess_string
# from nltk.tokenize import word_tokenize
import numpy as np


# Function to load and preprocess data
def load_and_preprocess_csv(filepath):
    # Load the CSV file
    df = pd.read_csv(filepath).dropna()
    print(df.shape)
    # Preprocess the 'comments' and 'method_name' fields
    df['comments_preprocessed'] = df['comments'].apply(lambda x: preprocess_string(str(x)))
    df['method_name_preprocessed'] = df['method_name'].apply(lambda x: preprocess_string(str(x)))
    return df

# Function to train or load a Word2Vec model
def get_word2vec_model(data, pretrained_model_path=None):
    if pretrained_model_path:
        model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)
    else:
        # Train a Word2Vec model
        model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Function to compute Word Mover's Distance and classify

def classify_entries(df, model):
    theta = 0.0674750 # Threshold for classification``
    # Define positive and negative reference texts
    # positive_ref = preprocess_string("Assuming a method named calculateTotalPrice with comments that explain it calculates the total price of items in a cart, the positive reference could be a concatenation of synonymous terms and phrases, e.g., 'Calculate sum total price items cart.'")
    # negative_ref = preprocess_string("For a negative example, if there's a method named updateUserProfile but the comments incorrectly suggest it retrieves user profiles, the negative reference could involve mixed signals, e.g., 'Update change user profile fetch retrieve information.'")
    # Define positive and negative reference texts for Java method-comment classification
    positive_ref = preprocess_string("Calculate total amount, return sum, get user details, update profile settings, save file to disk, load data from database, delete user account, validate input data")
    negative_ref = preprocess_string("Calculate retrieves information, update deletes data, save loads from disk, validate creates new entry, delete fetches records, get sets property")

    # Vectorize the reference texts
    positive_vec = np.mean([model.wv[word] for word in positive_ref if word in model.wv], axis=0)
    negative_vec = np.mean([model.wv[word] for word in negative_ref if word in model.wv], axis=0)
    
    def classify_row(row):
        comment_vec = np.mean([model.wv[word] for word in row['comments_preprocessed'] if word in model.wv], axis=0)
        method_vec = np.mean([model.wv[word] for word in row['method_name_preprocessed'] if word in model.wv], axis=0)
        
        # Compute WMD
        comment_distance_positive = np.linalg.norm(comment_vec - positive_vec)
        comment_distance_negative = np.linalg.norm(comment_vec - negative_vec)
        method_distance_positive = np.linalg.norm(method_vec - positive_vec)
        method_distance_negative = np.linalg.norm(method_vec - negative_vec)
        
        # Average the distances
        avg_distance_positive = (comment_distance_positive + method_distance_positive) / 2
        avg_distance_negative = (comment_distance_negative + method_distance_negative) / 2
        
        # Classify based on WMD
        if min(avg_distance_positive, avg_distance_negative) < theta:
            return 'positive'
        else:
            return 'negative'
    
    # Apply classification
    df['label'] = df.apply(classify_row, axis=1)
    return df


def algo5(filepath):
    df = load_and_preprocess_csv(filepath)
    word2vec_data = df['comments_preprocessed'] + df['method_name_preprocessed']
    model = get_word2vec_model(word2vec_data)
    classified_df = classify_entries(df, model)
    classified_df.to_csv(f"{filepath.split('_')[0]}_algo5_data.csv", index=False)
    print(classified_df['label'].value_counts())


if __name__ == "__main__":
    file = input()
    algo5(f'../{file}_data.csv')
# Note: The actual function calls are commented out because we cannot execute them here.

# pretrained_model_path = "path_to_pretrained_model.bin"  # If you have a pretrained model

