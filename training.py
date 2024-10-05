import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import pandas as pd
import re
import pickle

# DATASET https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset
# INSPIRATION https://f-a.nz/dev/develop-your-own-llm-like-chatgpt-with-tensorflow-and-keras/
txt_file = "TrainingText.txt"

file = "/Users/nabilmouss/Downloads/2020-05-01/ted_talks_en.csv"
dataset = pd.read_csv(file)
df = pd.DataFrame(dataset)
column = [18]
df = df[df.columns[column]]

with open(txt_file, "r") as file:
    text = file.read()

tokenizer = Tokenizer()

# Remove unwanted punctuation
cleaned_text = re.sub('^\w\s', " ", text)

# Split sentences by periods, exclamation marks, and question marks
sentences = re.split(r'[.!?]', cleaned_text) 
sentences = [sentence.strip() for sentence in sentences if sentence]  # Remove any empty spaces

sentences = sentences[:20000] # I AM JUST MODIFYING THIS JUST SO I CAN TEST

tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=20, padding='post')


# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
# Create input-output pairs for the model
input_data = []
output_data = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        input_data.append(sequence[:i])
        output_data.append(sequence[i])

# Pad the input sequences to a fixed length of 20
input_data = tf.keras.utils.pad_sequences(input_data, maxlen=20, padding='post')

# One-hot encode the output data
output_data = to_categorical(output_data, num_classes=vocab_size)

# Save the tokenizer to a file
with open('tokenizer.pickle', "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=20))
model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))  # Add return_sequences for stacking LSTMs
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # Another LSTM layer
model.add(Dense(vocab_size, activation='softmax')) #  activation='softmax'

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.summary()


# Train the model
model.fit(input_data, output_data, epochs=4, batch_size=128, validation_split=0.2)


model.save("next_token_predictor.keras")
