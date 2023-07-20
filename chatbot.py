import json
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)        
        documents.append((tokenized_words, intent['tag']))
     
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in '?']
words = sorted(set(words))

classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    tokenized_pattern = document[0]
    for word in words:
        bag.append(1) if word in tokenized_pattern else bag.append(0)
 
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])
    
# Shuffle and convert to NumPy array
np.random.shuffle(training)
training = np.array(training)

# Create train and test lists
X_train = list(training[:,0])
y_train = list(training[:,1])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, input_shape=(len(X_train[0]),), activation='relu'))
model.add(tf.keras.layers.Dense(len(y_train[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=8, verbose=1)

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def bag_of_words(text):
    tokens = clean_text(text)
    bag = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w: 
                bag[i] = 1
    return np.array(bag)

def predict_class(text):
    bow = bag_of_words(text)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

print("BOT IS READY")

# Get a response 
prediction = predict_class("hello there")
print(prediction)