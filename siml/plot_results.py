import matplotlib.pyplot as plt
from keras.callbacks import History 


def plot(history,epochs):
    #fig, axarr = plt.subplots(figsize=(12,6), ncols=1)
    fig, axarr = plt.plot() 
    axarr[0].plot(range(1, epochs+1), history.history['accuracy'], label='train score')
    axarr[0].plot(range(1, epochs+1), history.history['val_accuracy'], label='test score')
    axarr[0].set_xlabel('Number of Epochs', fontsize=18)
    axarr[0].set_ylabel('Accuracy', fontsize=18)
    #axarr[0].set_ylim([0,1])
    plt.legend()
    plt.show()    