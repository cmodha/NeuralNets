import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Dataset found on the tensorflow website

#importing the fashion dataset
data = keras.datasets.fashion_mnist
#loading the data into test and train sets as well as their labels
(trainImages,trainLabels),(testImages,testLabels) = data.load_data()
#creating array to allow conversino from numeric labels to class names
classNames = ['T-shirt','Trousers','Jumper','Dress','Coat','Sandal','Shirt','Trainer','Bag','Boot']

#dividing the image intensity values by 255 to convert them into floats for faster training times
trainImages = trainImages/255.0
testImages = testImages/255.0
#setting the neural net architecture
model = keras.Sequential([keras.layers.Flatten(input_shape =(28,28)),
        keras.layers.Dense(128,activation = "relu"),
        keras.layers.Dense(10,activation = "softmax")])
#compiling the model
model.compile(
                optimizer = tf.keras.optimizers.Adam(0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ["accuracy"])
#fitting the model
model.fit(trainImages,trainLabels,epochs = 5)
testLoss,testAcc = model.evaluate(testImages,testLabels)
print(f"Tested Acc: {testAcc}")

