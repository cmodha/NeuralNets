import tensorflow as tf
from tensorflow import keras
import numpy as np
#importing movie reviews
data = keras.datasets.imdb
#only taking the 10000 most frequent words
(train_data, train_labels), (test_data,test_labels) = data.load_data(num_words = 88000)

#assigning a dictionary which maps the words in the IMDB database
wordIdx = data.get_word_index()

wordIdx = {k:(v+3) for k, v in wordIdx.items()}
#adding useful tags into the word index for data processing
wordIdx["<PAD>"] = 0
wordIdx["<START>"] = 1
wordIdx["<UNK>"] = 2
wordIdx["<UNUSED>"] = 3
#reversing the word index to have an integer point to a word
reversewordIdx = dict([(value,key) for (key,value) in wordIdx.items()])
#splitting the data into test and training and adding padding to the datapoints to ensure that they are of the same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = wordIdx["<PAD>"], padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = wordIdx["<PAD>"], padding = "post", maxlen = 250)

#a function to return the decoded human readable review
def reviewDecode(text):
    return " ".join([reversewordIdx.get(i,"?") for i in text])

# #---MODEL ARCHITECTURE--------
# model = keras.Sequential()
# model.add(keras.layers.Embedding(88000,16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation = "relu"))
# model.add(keras.layers.Dense(1, activation = "sigmoid"))

#prints a summary of the model
# model.summary()

# model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# xVal = train_data[:10000]
# xTrain = train_data[10000:]

# yVal = train_labels[:10000]
# yTrain = train_labels[10000:]

# print(len(xTrain))
# fitModel = model.fit(xTrain, yTrain, epochs = 40, batch_size = 512, validation_data = (xVal,yVal), verbose = 1)

# results = model.evaluate(test_data,test_labels)

# print(results)

# model.save("model.h5")

#function to integer encode the data
def reviewEncode(string):
    encoded = [1]
    for word in string:
        if word.lower() in wordIdx:
            encoded.append(wordIdx[word.lower()])
        else:
            encoded.append(2)
    return encoded
#loading the model which has been saved
model = keras.models.load_model("model.h5")
#opeining the the review to test
with open("the_exterminating_angel_roger_ebert.txt") as f:
    for line in f.readlines():
        #getyting a new line and replacing characters which are not included in the word index
        newLine = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace(";","").replace("-","").replace("\"","").strip().split(" ")
        #encoding the review
        encode = reviewEncode(newLine)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = wordIdx["<PAD>"], padding = "post", maxlen = 250)
        predict = model.predict(encode)
        print(newLine)
        print(encode)
        print(predict[0])



