import numpy as np
import pickle
import string
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Dense, Input, LSTM
from tensorflow.keras.models import Model, load_model


def load_doc(path, file_name):
    with open(os.path.join(path, file_name), mode='r') as file:
        data = file.read()
    return data


def clean_doc(text):
    tokens = text.replace('--', ' ').split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens


def make_sequence(tokens, length=50):
    length = length + 1
    sequence = []
    for i in range(length, len(tokens)):
        seq = tokens[i - length:i]
        line = ' '.join(seq)
        sequence.append(line)
    print('Total Sequences: %d' % len(sequence))
    return sequence


def save_doc(lines, path, filename):
    data = '\n'.join(lines)
    with open(os.path.join(path, filename), 'w') as file:
        file.write(data)
    print('File Saved')


path = '/home/vargha/Desktop'
filename = 'republic.txt'
doc = load_doc(path, filename)
tokens = clean_doc(doc)
sequences = make_sequence(tokens)
save_doc(sequences, path, 'republic_sequences.txt')

train_doc = load_doc(path, 'republic_sequences.txt')
lines = train_doc.split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1


# Preprocessing x_train and y_train
def preprocessing(sequences):
    sequences = np.array(sequences)
    x_train, y_train = sequences[:, :-1], sequences[:, -1]
    y_train = to_categorical(y_train, num_classes=vocab_size)
    seq_length = x_train.shape[1]
    print('Sequence Length is: ', seq_length)
    print('X_train shape', x_train.shape)
    print('Y_train shape', y_train.shape)
    return seq_length, x_train, y_train


# Model
def model(input_shape, vocab_size, output_dim):
    inputs = Input(shape=input_shape)
    x = Embedding(input_dim=vocab_size, output_dim=output_dim)(inputs)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


seq_length, x_train, y_train = preprocessing(sequences)
model = model((seq_length,), vocab_size, seq_length)
model.fit(x_train, y_train, batch_size=1, epochs=100)

# Saving Model and Tokenizer
model.save('model.h5')
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

# Predicting and Testing
seed_text = lines[np.random.randint(0, len(lines))]
encoded = np.array(tokenizer.texts_to_sequences([sequences]))
y_hat = np.argmax(model.predict(encoded[:, :seq_length], verbose=1)[0])
out_word = ''
for word, index in tokenizer.word_index.items():
    if index == y_hat:
        out_word = word
        break
print('Predicted word is: ', out_word)


def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = np.array(tokenizer.texts_to_sequences([in_text]))
        encoded = pad_sequences(encoded, maxlen=seq_length, truncating='pre')
        y_hat = np.argmax(model.predict(encoded, verbose=0)[0])
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_hat:
                out_word = word
                break
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(seed_text)
print(generated)
