import numpy as np
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History 

def cnn_classify(train_data_cwt, uci_har_labels_train, test_data_cwt, uci_har_labels_test, epochs):
    history = History()

    x_train = train_data_cwt
    y_train = list(uci_har_labels_train)
    x_test = test_data_cwt
    y_test = list(uci_har_labels_test)
    num_classes = 6
    
    batch_size = 16
    
    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    input_shape = np.shape(train_data_cwt)[1:]
    
    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    print("x_train shape: {}, x_test shape: {}".format(x_train.shape, x_test.shape))
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
        
    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=keras.optimizers.Adam(), 
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=batch_size, 
              epochs=epochs, verbose=1, 
              validation_data=(x_test, y_test), 
              callbacks=[history])
    
    train_score = model.evaluate(x_train, y_train, verbose=0)
    #print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    test_score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))    
    
    return train_score,test_score,history
