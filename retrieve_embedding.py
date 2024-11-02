import pickle
import sys
import os
import gdown
import numpy as np
import re
import spacy
from Levenshtein import distance as levenshtein_distance

#GOOGLE_DRIVE_FILE_ID = "1ADEMKwJQzZUaMF3OPAWNLbVwmr7I7COh" 
EMBEDDING_PATH = "final_dictionary.pkl"

nlp = spacy.load("en_core_web_sm")

'''
def download_embedding_matrix():
    """
    Downloads the embedding matrix from Google Drive.
    """
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", EMBEDDING_PATH, quiet=False)
    
'''

def load_embedding_matrix():
    """
    Loads the embedding matrix after downloading it, if necessary.

    Returns:
    - Dict: The embedding matrix.
    """
    # Check if the embedding file already exists
    if not os.path.exists(EMBEDDING_PATH):
        print('FULE NOT FOUND')

    # Load the embedding matrix
    try:
        with open(EMBEDDING_PATH, 'rb') as f:
            loaded_embedding_matrix = pickle.load(f)
            embedding_matrix = {}
            for word, embedding in loaded_embedding_matrix.items():
                embedding_matrix[word] = np.array(embedding) # Formatting
        return embedding_matrix
    except Exception as e:
        print(f"Error loading embedding matrix: {e}")  
        sys.exit(1)

def check_existence_in_glove(word):
    """
    Checks the existence of a given word in the embedding matrix.
    
    Parameters:
    - word (str): The word to retrieve the embedding for.

    Returns:
    - bool: True if the word exists in the embedding matrix, False otherwise.
    """
    embedding_matrix = load_embedding_matrix()

    if word in embedding_matrix:
        return True
    else:
        return False
    
def check_hyphen(word):
    """
    Checks the existence of hyphens (-) in a given word.
    
    Parameters:
    - word (str): The word to retrieve the embedding for.

    Returns:
    - bool: True if the word contains hyphens, False otherwise.
    """
    if "-" in word:
        return True
    else:
        return False

def check_neg_prefix(word):
    """
    Checks the existence of negative prefixes in a given word.
    
    Parameters:
    - word (str): The word to retrieve the embedding for.

    Returns:
    - bool: True if the word contains negative prefixes, False otherwise.
    """
    prefix_list = ["dis", "de", "mis", "un", "im", "in", "il", "ir", "anti", "non", "ex"]
    for prefix in prefix_list:
        if word.startswith(prefix):
            root_word = word[len(prefix):]
            root_word_exists = check_existence_in_glove(root_word)
            if root_word_exists:
                return prefix
    return None

def check_slash(word):
    """
    Checks the existence of slashes (/) in a given word.
    
    Parameters:
    - word (str): The word to retrieve the embedding for.

    Returns:
    - bool: True if the word contains slashes, False otherwise.
    """
    if "/" in word:
        return True
    else:
        return False

def split_words(word, separator):
    """
    Splits the given word into a list using the specified separator.
    
    Parameters:
    - word (str): The word to split.
    - separator (str): The separator to use for splitting. Can be "/", "-" or a negative prefix.

    Returns:
    - list: A list of separated words.
    """
    if separator == "-":
        word_list = re.split(f'({re.escape(separator)})', word) # Keeps "-" in the list
    elif separator == "/":
        word_list = word.split(separator) # Does not keep "/" in the list
    else:
        word_list = word.split(separator, 1) # Retrieves the root form
        word_list.append(separator) # Add the negative prefix back to the list
    return [w for w in word_list if w]

def find_best_match(word, embedding_matrix):
    """
    Finds the approximate match of the given word using Levenshtein Distance.
    
    Parameters:
    - word (str): The word to split.
    - embedding matrix (dict): The embedding matrix.

    Returns:
    - str: The closest match if the edit distance is less than min(3, len(word)), None otherwise.
    """
    min_distance = float('inf')
    for matrix_word in embedding_matrix:
        distance = levenshtein_distance(word, matrix_word)
        if distance < min_distance:
            min_distance = distance
            best_match = matrix_word
    if min_distance < min(3, len(word)): # Only words with less than 3 or the word length, whichever is smaller, edit distance is accepted
        return best_match
    else:
        return None

def get_lemmatized_word(word):
    """
    Lemmatize the given word using Spacy.
    
    Parameters:
    - word (str): The word to lemmatize.

    Returns:
    - str: Lemmatized word.
    """
    word = word.replace("'", "") # To prevent lists due to an apostrophe typo
    spacy_doc = nlp(word)
    spacy_lemma = [token.lemma_ for token in spacy_doc]
    return spacy_lemma[0]

def get_embedding(word, embedding_matrix):
    """
    Retrieves the embedding for a given word from the embedding matrix.
    
    Parameters:
    - word: The word to retrieve the embedding for.
    - embedding_matrix: The embedding_matrix that contains the word.

    Returns:
    - arr: The embedding of the word if found, None otherwise.
    """
    if word in embedding_matrix:
        return embedding_matrix[word]
    else:
        return None

def sum_embedding(word_list, embedding_matrix):
    """
    Sums the embedding for each word found in the word list from the embedding matrix.
    
    Parameters:
    - word_list: The words to retrieve the embedding for.
    - embedding_matrix: The embedding_matrix that contains the words.

    Returns:
    - arr: The embedding of the word if all words in the list are found, None otherwise.
    """
    embedding = np.zeros(300)
    for item in word_list:
        item_exist = check_existence_in_glove(item) 
        if not item_exist: # Do not calculate the summed embeddings if not all sub-words have embeddings
            return None
    for item in word_list:
        item_embedding = get_embedding(item, embedding_matrix)
        embedding = np.sum([embedding, item_embedding], axis=0)
    return embedding

def average_embedding(word_list, embedding_matrix):
    """
    Average out the embedding for each word found in the word list from the embedding matrix.
    
    Parameters:
    - word_list: The words to retrieve the embedding for.
    - embedding_matrix: The embedding_matrix that contains the words.

    Returns:
    - arr: The embedding of the word if all words in the list are found, None otherwise.
    """
    embedding = np.zeros(300)
    for item in word_list:
        item_exist = check_existence_in_glove(item) 
        if not item_exist: # Do not calculate the average embeddings if not all sub-words have embeddings
            return None
    for item in word_list:
        item_embedding = get_embedding(item, embedding_matrix)
        embedding = np.average([embedding, item_embedding], axis=0)
    return embedding

def main():
    embedding_matrix = load_embedding_matrix()
    unk_embedding = get_embedding("<unk>", embedding_matrix)

    if len(sys.argv) != 2:
        sys.exit(1)

    word = sys.argv[1]

    word_exist = check_existence_in_glove(word)
    if word_exist:
        embedding = get_embedding(word, embedding_matrix)
        print(embedding)
    else:
        hyphen_exist = check_hyphen(word)
        if hyphen_exist:
            word_list = split_words(word, "-")
            embedding = sum_embedding(word_list, embedding_matrix)
            if embedding is not None:
                print(embedding)
            else:
                print(unk_embedding)
        else:
            neg_prefix = check_neg_prefix(word)
            if neg_prefix is not None:
                word_list = split_words(word, neg_prefix)
                embedding = sum_embedding(word_list, embedding_matrix)
                if embedding is not None:
                    print(embedding)
            else:
                closest_match = find_best_match(word, embedding_matrix)
                if closest_match is not None:
                    embedding = get_embedding(closest_match, embedding_matrix)
                    print(embedding)
                else:
                    slash_exist = check_slash(word)
                    if slash_exist:
                        word_list = split_words(word, "/")
                        embedding = average_embedding(word_list, embedding_matrix)
                        print(embedding)
                    else:
                        lemmatized_word = get_lemmatized_word(word)
                        embedding = get_embedding(lemmatized_word, embedding_matrix)
                        if embedding is not None:
                            print(embedding)
                        else:
                            print(unk_embedding)
if __name__ == "__main__":
    main()
