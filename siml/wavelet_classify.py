import numpy as np
import pywt
from keras.callbacks import History
import uci_har_data
import scaleograms
import cnn_classify
import plot_results

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def main():    
    # Load and randomize the UCI HAR dataset
    train_signals, train_labels, test_signals, test_labels = uci_har_data.load()
    uci_har_signals_train, uci_har_labels_train = randomize(train_signals, np.array(train_labels))
    uci_har_signals_test, uci_har_labels_test = randomize(test_signals, np.array(test_labels))
    
    wavlist = pywt.wavelist(kind='continuous')
    nw = len(wavlist)
    wav_bat1 = ['morl']  #wavlist[:int(nw/3)]
    wav_bat2 = wavlist[int(nw/3)+1:int((2*nw)/3)]
    wav_bat2 = wavlist[int((2*nw)/3)+1:]
    epochs = 2
    no_train = 1000
    no_test = 100
    results = []
    for wvlt in wav_bat1:
        train_data_cwt,test_data_cwt = scaleograms.gen(uci_har_signals_train[:no_train], uci_har_signals_test[:no_test], wvlt)
        train_score,test_score,history = cnn_classify.cnn_classify \
            (train_data_cwt,uci_har_labels_train[:no_train],test_data_cwt,uci_har_labels_test[:no_test],epochs)
        
        # We will store the results of this iteration in a list of dicts for later retrieval
        results.append({'wavelet':wvlt,'train_score':train_score,'test_score':test_score,'history':history})
        plot_results.plot(results[-1]['history'],epochs)

if __name__ == '__main__':
    main()