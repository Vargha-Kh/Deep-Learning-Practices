import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GRU, RNN, LSTMCell, Activation, Input, Dense, TimeDistributed, Embedding, \
    Bidirectional, \
    RepeatVector, Flatten, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy


def load_file(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding="utf8") as f:
        data = f.read()
    return data.split('\n')


german_sen = load_file('/home/vargha/Desktop/train.de')
english_sen = load_file('/home/vargha/Desktop/train.en')


def tokenize(input):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input)
    input_tokenized = tokenizer.texts_to_sequences(input)
    return input_tokenized, tokenizer


german_tokenized, german_tokenizer = tokenize(german_sen)
english_tokenized, english_tokenizer = tokenize(english_sen)


def pad(input, length=None):
    if length is None:
        length = max([len(seq) for seq in input])
    return pad_sequences(input, maxlen=length, padding='post')


ger_padded = pad(german_tokenized)
ger_padded.reshape(*ger_padded.shape, 1)
eng_padded = pad(english_tokenized)


def advance_model(input_shape, output_len, uniq_en_words, uniq_de_words):
    model = Sequential()
    model.add(Embedding(uniq_en_words, 512, input_length=input_shape[1]))
    model.add(Bidirectional(RNN(LSTMCell(512), return_sequences=False)))
    model.add(RepeatVector(output_len))
    model.add(Bidirectional(RNN(LSTMCell(512), return_sequences=True)))
    model.add(TimeDistributed(Dense(uniq_de_words)))
    model.add(Activation('softmax'))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(0.002), metrics=['accuracy'])
    return model


model = advance_model(eng_padded.shape, ger_padded.shape[1], len(english_tokenizer.word_index),
                      len(german_tokenizer.word_index))
model.fit(eng_padded, ger_padded, batch_size=1, epochs=10, validation_split=0.1)

de_to_word = {value: key for key, value in german_tokenizer.word_index.items()}
de_to_word[0] = '|empty space|'
sentence = 'our home ise beautiful and you are dead'
sentence = [english_tokenizer.word_index[word] for word in sentence.split()]
sentence = pad_sequences([sentence], maxlen=eng_padded.shape[-1], padding='post')
sentences = np.array([sentence[0]])

predictions = model.predict(sentence)
print(' '.join([de_to_word[np.argmax(x)] for x in predictions[0]]))
