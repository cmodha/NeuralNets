import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow
import random
import json
import os

#importing the stemmer
stemmer = LancasterStemmer()

#loading the intents file
with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docsX = []
docsY = []
#filling the arrays with the appopriate data from the intents file
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docsX.append(wrds)
        docsY.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append (intent["tag"])

#reducing the words in the array to their stems and sorting them 
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

#sorting the labels
labels = sorted(labels)

training = []
output = []
#creating an empty output array
outEmpty = [0 for _ in range(len(labels))]
#filling the word bag from the docsX array
for x, doc in enumerate(docsX):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    outputRow = outEmpty[:]
    outputRow[labels.index(docsY[x])] = 1

    training.append(bag)
    output.append(outputRow)

#turning the training and output array into numpy arrays
training = np.array(training)
output = np.array(output)

#using v1 compatability
tensorflow.compat.v1.reset_default_graph()

#setting out the network architecture
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

#creating the model
model = tflearn.DNN(net)
#checking whether a saved model already exists and if not savinf the current model
if os.path.exists("model.tflearn.meta"):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
#defining the function into a bag of words using the created word list
def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]

    sWords = nltk.word_tokenize(s)
    sWords = [stemmer.stem(word.lower()) for word in sWords]

    for se in sWords:
        for i, w in enumerate(words):
            if w==se:
                bag[i] = 1

    return np.array(bag)

#defining the main chat function
def chat():
    print("Start talking with the bot: (type quit to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bagOfWords(inp, words)])
        resultsIdx = np.argmax(results)
        tag = labels[resultsIdx]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
