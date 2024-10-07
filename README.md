# Generating the top next 3 tokens

## What is this?

This project was an intro to generative AI and using neural network to generate the next best 3 tokens in a sentence. It gives a suggestion just like when you are texting someone on Imessage. This project uses 2 LSTM (Long Short term memory) layers that consists of 128 neurons (per layer). The last layer is a softmax activation function which gives out predictions of the best tokens since its technically a classification model. 

<img width="180" alt="Screenshot 2024-10-07 at 1 31 31 PM" src="https://github.com/user-attachments/assets/c117e1fd-2917-407b-9356-052085136c30">

* In this example the words: "wait", "believe", "imagine"; are the words that my model produces when the sentence is "Yes, I can't..."


## Dataset & Inspiration

I trained my model on nearly 30,000 sentences from tedtalks. I broke each tedtalk into sentences, tokenized the sentences (Mapped out every word to a value) and then ran it through the LSTM model. I used sequential model where I used keras by tensorflow to easily create my layers. I trained the model until I noticed the validation loss was staggering or increasing.

DATASET: https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset
Inspiration: https://f-a.nz/dev/develop-your-own-llm-like-chatgpt-with-tensorflow-and-keras/
