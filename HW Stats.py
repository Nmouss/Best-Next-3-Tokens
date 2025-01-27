import numpy as np 
import matplotlib.pyplot as plt
text_file = "/Users/nabilmouss/Downloads/formatted_lyrics_1989.txt"
file = open(text_file, "r")
words = file.read()
file.close()

words = words.split() # a list of all the words

Org_words = list(set(words)) # creating a list of the original words IN ORDER
index_to_word = {}
word_to_index = {}
for i in range(len(Org_words)): 
    index_to_word[i] = Org_words[i] # index to word
    word_to_index[Org_words[i]] = i # word to index

# a. create the graph of the most used 20 words
map = {}
for word in words:
    if word not in map:
        map[word] = 1
    else:
        map[word] += 1

sorted_data = sorted(map.items(), key=lambda item: item[1], reverse=True)

# get the top first 20 count
top_words = [item[0] for item in sorted_data[:20]]
top_counts = [item[1] for item in sorted_data[:20]] 

total_count = sum(map.values())
relative_frequencies = [count / total_count for count in top_counts]

plt.figure(figsize=(10, 6))
plt.stem(top_words, relative_frequencies, basefmt=" ") # x = top words, y = relative frequencies, base format, formats the string
plt.title("Relative Frequencies of the 20 Most Frequently Used Words")
plt.xlabel("Words")
plt.ylabel("Relative Frequency")
plt.xticks(rotation=45, ha='right') # this is rotating x axis
plt.tight_layout()
plt.show()

# b. create the 692 x 692 populate it with conditional prob. 
n = len(Org_words)
T = np.zeros((n, n)) # here i am creating a 692x692 of zeros
for i in range(len(words) - 1): # i am iterating through all of the words that have no duplicates
    current_word = words[i]
    next_word = words[i + 1] # next word is the one after 
    T[word_to_index[current_word], word_to_index[next_word]] += 1

row_sums = T.sum(axis=1, keepdims=True)  # add sum of each row
row_sums[row_sums == 0] = 1 
T = T / row_sums  # divide rows by their total summation

def most_likely_next_word(word): 
    word_idx = word_to_index.get(word) # i am getting the index of the word that i selected from the map
    probs = T[word_idx] # getting the probability of the next word
    
    next_word_idx = np.random.choice(len(probs), p=probs) # since my alg was stuck i am going to use np.random relative to prob
    return index_to_word[next_word_idx] # i am returning the word given its index

def most_likely_next_word_highestProb(word): 
    word_idx = word_to_index.get(word) # i am getting the index of the word that i selected from the map

    next_word_idx = np.argmax(T[word_idx]) # this is getting the max probability of the next word
    return index_to_word[next_word_idx]

most_likely_after_wildest = most_likely_next_word_highestProb("wildest") # running it through the function
most_likely_after_shake = most_likely_next_word_highestProb("shake")
print("PART B")
print("Word after wildest: " + most_likely_after_wildest)
print("Word after shake: " + most_likely_after_shake)
print()

# c. create a 10 word sequence after the word 'you'
current_word = 'you'

max_sentence_length = 10 

generated_sentence = current_word

for _ in range(max_sentence_length - 1):
    next_word = most_likely_next_word_highestProb(current_word) # running through the function

    generated_sentence += ' ' + next_word # adding it to the total string

    current_word = next_word # setting to current so i can predict next
print("PART C")
print("Generated Sentence:", generated_sentence)
print()

# d. create 5 sentences that start with the word 'you'
print("PART D")
current_word = 'you'

max_sentence_length = 10

generated_sentence = current_word

for i in range(5): # basically repetition of part c however now im adding some slight randomness
    generated_sentence = "you"
    for _ in range(max_sentence_length - 1):
        next_word = most_likely_next_word(current_word) 

        generated_sentence += ' ' + next_word

        current_word = next_word

    print("Generated Sentence:", generated_sentence)

"""
PART B
Word after wildest: dreams
Word after shake: it

PART C
Generated Sentence: you would i i i i i i i i

PART D
Generated Sentence: you had our broken hearts put them in your car
Generated Sentence: you windows down down in the fella over there and
Generated Sentence: you i wish you were young and fake fake fake
Generated Sentence: you fake fake and this beat forevermore the rest of
Generated Sentence: you the lights are now weve got nothin in my
"""