import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# download dataset
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

print('Training data size: ' + str(len(train_data)), 
      '\nTraining label size: ' + str(len(train_labels)), 
      '\nTesting data size: ' + str(len(test_data)), 
      '\nTesting label size: ' + str(len(test_labels)))

word_index = imdb.get_word_index()

word_index = {key : (val + 3) for key, val in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

num_index = dict([(val, key) for (key, val) in word_index.items()])

def decode_review(text):
    return ' '.join([num_index.get(i, '?') for i in text])

#print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, 
                                                        value = 0, 
                                                        padding = 'post', 
                                                        maxlen = 256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, 
                                                       value = 0, 
                                                       padding = 'post', 
                                                       maxlen = 256)

print("New lengths: " + str(len(train_data[0])) + " " + str(len(train_data[1])))

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = tf.nn.relu))
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid))

print(model.summary())

model.compile(optimizer = tf.train.AdamOptimizer(), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

# create validation set using 10000 elements
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# training:
# 88% acc on 20 epochs, loss = 0.3664
# 93% acc on 40 epochs, loss = 0.1936, val acc = 88%
# 96% acc on 60 epochs, loss = 0.1224, val acc = 88% --> possible overfitting
history = model.fit(x = partial_x_train, 
                    y = partial_y_train, 
                    epochs = 40, 
                    batch_size = 512, 
                    validation_data = (x_val, y_val))

results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label = 'Training Loss')

plt.plot(epochs, loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label = "Training accuracy")
plt.plot(epochs, val_acc, 'b', label = "Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()