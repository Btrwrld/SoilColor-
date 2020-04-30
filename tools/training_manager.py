import numpy as np

from tools.data_loader import hue_norm, chroma_norm,value_norm




def classify(pred):

    # Get the index of the closest value
    hue = np.argmin((pred[0] - hue_norm) ** 2)
    chroma = np.argmin((pred[2] - chroma_norm) ** 2)
    value = np.argmin((pred[2] - value_norm) ** 2)

    # if there are ties take the first index
    if(type(hue) == list):
        hue = hue[0]
    if(type(chroma) == list):
        chroma = chroma[0]
    if(type(value) == list):
        value = value[0]

    # Get the value
    hue = hue_norm[hue]
    chroma = chroma_norm[chroma]
    value = value_norm[value]

    # Return the classification
    return [hue, chroma, value]

def batch_accuracy(pred, y):

    # Convert to numpy array create accuracy
    pred = pred.data.numpy()
    y = y.data.numpy()
    acc = np.zeros(3)

    for i in range(len(pred)):

        c = np.array(classify(pred[i, :]))
        acc += np.isclose(c, y[i, :])
        #print(str(c) + '\t=\t' + str(y[i, :]) + '\t===>' + str(np.isclose(c, y[i, :])) + '\t' + str(acc))
    
    return acc




def get_data_batch(x, y, i, batch_size):
    # Calc the batch upper limmit
    upper_limit = np.amin([len(x), i*batch_size + batch_size])

    # Get the batch and predictions
    x = x[i*batch_size: upper_limit, :]
    y = y[i*batch_size: upper_limit, :]

    return x, y