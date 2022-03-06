from platform import python_version
print('Python version: {}'.format(python_version()))

import numpy as np
import os
import tensorflow as tf

# Let's start by loading a dataset: the traditional MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# DATA EXPLORATION
print('Class of x_train: {}'.format(type(x_train)))
print('Class of y_train: {}'.format(type(y_train)))
print('Shape of x_train: {}'.format(x_train.shape))
print('Shape of y_train: {}'.format(y_train.shape))
print('Shape of x_test: {}'.format(x_test.shape))
print('Shape of y_test: {}'.format(y_test.shape))

# let's visualize some data, but we need another popular library
import matplotlib.pyplot as plt

#import artist as plt

idxes = [np.random.randint(60000) for i in range(4)]
plt.figure(figsize=(15,8))
#plt.show(figsize=(15,8))
for i in range(4):
  plt.subplot(1,4,i+1)
  plt.imshow(x_train[idxes[i]],cmap='Greys')
  plt.title('Label = {}'.format(y_train[idxes[i]]))

plt.show()  # aggiunto da me
  
# Let's analyze the distribution of classes in training
from collections import Counter

train_class_counter = Counter(y_train)
plt.figure()
plt.bar(train_class_counter.keys(),train_class_counter.values())
plt.title('Class distribution in training dataset')
test_class_counter = Counter(y_test)
plt.figure()
plt.bar(test_class_counter.keys(),test_class_counter.values())
plt.title('Class distribution in test dataset')

plt.show()  # aggiunto da me

# Let's don't forget about normalization!
x_train, x_test = x_train / 255.0, x_test / 255.0


# MODEL DEFINITION
# 
# We use tf.keras to define models by means of a simple, high-level library. Each layer is defined by some common parameters:
# 
# activation: the activation function for the layer. This parameter is specified by the name of a built-in function or as a callable object. By default, no activation is applied.
# kernel_initializer and bias_initializer: the initialization schemes that create the layer's weights (kernel and bias). This parameter is a name or a callable object. This defaults to the "Glorot uniform" initializer.
# kernel_regularizer and bias_regularizer: The regularization schemes that apply the layer's weights (kernel and bias), such as L1 or L2 regularization. By default, no regularization is applied.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout


def create_model():
    model = Sequential(
    [
        Flatten(input_shape=(28,28)),
        Dense(512,activation='relu'),
        Dropout(0.2),
        Dense(10,activation='softmax')
    ])

    # The compile step specifies the training configuration.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = create_model()


model.summary()
tf.keras.utils.plot_model(model) # non funzia

history = model.fit(x_train,y_train,batch_size=32,epochs=10,validation_split=0.1)
model.save("model1_mnist_10epoch.h5")

plt.figure()
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.grid()
plt.title('Training vs validation accuracy')
plt.legend()

plt.show()  # aggiunto da me

#EVALUATION
[test_loss, test_accuracy] = model.evaluate(x_test,y_test)
print('Test accuracy: {}'.format(test_accuracy))

y_pred = np.argmax(model.predict(x_test),axis=-1)

mismatch = np.where(y_pred!=y_test)[0]

mismatch_class_counter = Counter(y_test[mismatch])
mismatch_percentage = dict()
for digit in test_class_counter.keys():
    mismatch_percentage[digit]=mismatch_class_counter[digit]/test_class_counter[digit]*100
plt.figure()
plt.bar(mismatch_percentage.keys(),mismatch_percentage.values())
plt.title('Class distribution for mismatch in label prediction')

plt.show()  # aggiunto da me


np.random.shuffle(mismatch)
idxes =  mismatch[:4]
plt.figure(figsize=(15,8))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(x_test[idxes[i]],cmap='Greys')
    plt.title('True label = {}\nPredicted label: {}'.format(y_test[idxes[i]],y_pred[idxes[i]]))

plt.show()  # aggiunto da me

from sklearn.metrics import confusion_matrix

cm = np.log(1+confusion_matrix(y_test,y_pred))

fig,ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest',cmap='Reds')
ax.figure.colorbar(im, ax=ax)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.show()  # aggiunto da me


##-----------------------------------------------------------------------------------------------------------------

# First we need to connect Colab to our g-drive folder:
from google.colab import drive
drive.mount('/content/gdrive')
#drive.mount('/content/gdrive/Colab_Notebooks_test')

# folder = '/content/gdrive/My Drive/Colab Notebooks/Master Big Data and Machine Learning'
# #destination_folder = os.path.join(folder,'01_FFNN')
# destination_folder = os.path.join(folder,'PROVA')

folder = 'C:\\Users\\Antonio\\Desktop\\Master\\python\\python-MNIST'
#destination_folder = os.path.join(folder,'01_FFNN')
destination_folder = os.path.join(folder,'PROVA')
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)
    
os.listdir(destination_folder)


# Save the weights
model.save_weights(destination_folder+'/weights')
# Create a basic model instance
model_reloaded = create_model()
# Restore the weights
model_reloaded.load_weights(destination_folder+'/weights')

[reloaded_test_loss, reloaded_test_accuracy] = model_reloaded.evaluate(x_test,y_test)
print('Test accuracy from original model: {}'.format(test_accuracy))
print('Test accuracy from reloaded model: {}'.format(reloaded_test_accuracy))


# Saving checkpoints during training with Keras callbacks
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = destination_folder+"/training_00_FFNN/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#we define a callback to trigger checkpoints saving during training
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    # Save weights, every 2-epochs.
    period=2)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(x_train, y_train,
          epochs = 10, 
          callbacks = [cp_callback],
          validation_data = (x_test,y_test),
          verbose=1)
          
          
# Saving fully-functional models
model.save(destination_folder+'/mnist_00_ffnn.h5')

new_model = tf.keras.models.load_model(destination_folder+'/mnist_00_ffnn.h5')
print(new_model.summary())

[new_test_loss, new_test_accuracy] = new_model.evaluate(x_test,y_test)
print('\n\nTest accuracy from original model: {}'.format(test_accuracy))
print('Test accuracy from new model: {}'.format(new_test_accuracy))







