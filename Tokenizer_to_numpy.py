from keras.preprocessing.text import Tokenizer

tok = Tokenizer(num_words=MAX_WORDS)

def text_sequence_to_numpy_array(seqs, num_words=MAX_WORDS):
    return Tokenizer(num_words=num_words).sequences_to_matrix(seqs, mode='binary')