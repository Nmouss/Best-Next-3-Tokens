from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = load_model("PATH/TO/YOUR/next_token_predictor.keras")

'''# Example sentence to predict the next token
input_text = "This is the first thing that I think you can be"

# Tokenize and convert the input text to a sequence of integers
input_sequence = tokenizer.texts_to_sequences([input_text])

# Pad the sequence to the required input length (20 in your case)
padded_input = pad_sequences(input_sequence, maxlen=20, padding='post')

# Predict the next token (the model will output probabilities for all tokens in the vocabulary)
predicted_probabilities = model.predict(padded_input)

# Get the token with the highest probability
predicted_token_id = np.argmax(predicted_probabilities, axis=-1)

# Convert the token ID back to the corresponding word
predicted_word = tokenizer.index_word.get(predicted_token_id[0], "<unk>")  # handle unknown tokens
print(predicted_word)'''


# Function to apply temperature sampling
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    probabilities = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(probabilities), p=probabilities)


temperature = 0.8 # the higher the tempurature the more creative the model is


num_tokens_to_generate = 1 # generates the next 20 tokens
input_text = "The jacket was"
generated_text = input_text

for _ in range(num_tokens_to_generate):
    
    input_sequence = tokenizer.texts_to_sequences([generated_text]) # tokenizes the sentence so the model can read it
    
    
    input_sequence = input_sequence[0][-20:] # only keep the most recent 20 tokens
    padded_input = pad_sequences([input_sequence], maxlen=20, padding='post')

    
    predicted_probabilities = model.predict(padded_input)[0] # predict the next token with the highest probability
    predicted_probabilities1 = model.predict(padded_input)[0] # predict the next token with the highest probability (CAN DELETE)
    predicted_probabilities2 = model.predict(padded_input)[0] # predict the next token with the highest probability (CAN DELETE)
    
    predicted_token_id = sample_with_temperature(predicted_probabilities, temperature=temperature) # Sample the next token with randomness
    predicted_token_id1 = sample_with_temperature(predicted_probabilities1, temperature=temperature + 0.2)
    predicted_token_id2 = sample_with_temperature(predicted_probabilities2, temperature=temperature - 0.2)

    
    # Convert the token ID back to word
    next_word = tokenizer.index_word.get(predicted_token_id, "<unk>")  # handle unknown words
    next_word1 = tokenizer.index_word.get(predicted_token_id1, "<unk>") # (CAN DELETE)
    next_word2 = tokenizer.index_word.get(predicted_token_id2, "<unk>") # (CAN DELETE)

    # Append the next word to the generated text
    generated_text += " " + next_word + " " + next_word1 + " " + next_word2

print(generated_text)

