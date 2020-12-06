import pywt
import numpy as np

def gen(signals_train, signals_test, waveletname):
    scales = range(1,128)
    
    train_size = np.shape(signals_train)[0]
    size_xy = np.shape(signals_train)[1]
    nchan = np.shape(signals_train)[2]
    train_data_cwt = np.ndarray(shape=(train_size, size_xy-1, size_xy-1, nchan))
    
    for ii in range(0,train_size):
        #if ii % 1000 == 0:
        #    print(ii)
        for jj in range(0,nchan):
            signal = signals_train[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:,:size_xy-1]
            train_data_cwt[ii, :, :, jj] = coeff_
    
    test_size = np.shape(signals_test)[0]
    test_data_cwt = np.ndarray(shape=(test_size, size_xy-1, size_xy-1, nchan))
    for ii in range(0,test_size):
        #if ii % 100 == 0:
        #    print(ii)
        for jj in range(0,nchan):
            signal = signals_test[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:,:size_xy-1]
            test_data_cwt[ii, :, :, jj] = coeff_

    return train_data_cwt,test_data_cwt