from keras.models import Sequential
from keras.layers import Dense
import numpy

#random
numpy.random.seed(2)

#read data
dataset = numpy.loadtxt("data_diabet.csv", delimetr = ",")

X = dataset[:,0:8]
Y = dataset[:,8]

#model ai
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))

model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


#learn ai
model.fit(X, Y, epochs = 1000, batch_size=10)

#result
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
