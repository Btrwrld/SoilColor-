import numpy as np
import matplotlib.pyplot as plt

from tools.data_manager import hue_norm, chroma_norm,value_norm

def classify(pred):

    # Get the index of the closest value
    hue = np.argmin((pred[0] - hue_norm) ** 2)
    chroma = np.argmin((pred[1] - chroma_norm) ** 2)
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
        # Comparamos los resultados con las respuestas correctas 
        # y contamos los aciertos
        acc += np.isclose(np.array(classify(pred[i, :])), y[i, :])
    
    return acc




def get_data_batch(x, y, i, batch_size):
    # Calc the batch upper limmit
    upper_limit = np.amin([len(x), i*batch_size + batch_size])

    # Get the batch and predictions
    x = x[i*batch_size: upper_limit, :]
    y = y[i*batch_size: upper_limit, :]

    return x, y



def plot_results(save_dir, model_name, epochs, hiperparameters, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc):

    # Create the image
    fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0.5})
    fig.suptitle(hiperparameters + ' model loss and accuracy')

    # Plot the loss
    axs[0].set_title('Model loss')
    axs[0].plot(np.arange(epochs), epoch_train_loss, 'r-', label='Training')
    axs[0].plot(np.arange(epochs), epoch_val_loss, 'b-', label='Validation')
    axs.flat[0].set(xlabel='Epochs', ylabel='MSE loss')
    axs[0].legend()

    # Plot the training accuracy
    axs[1].set_title('Training accuracy')
    axs[1].plot(np.arange(epochs), epoch_train_acc[:, 0], 'g-', label='Hue')
    axs[1].plot(np.arange(epochs), epoch_train_acc[:, 1], 'c-', label='Chroma')
    axs[1].plot(np.arange(epochs), epoch_train_acc[:, 2], 'y-', label='Value')
    axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')
    axs[1].legend()

    # Plot the validation accuracy
    axs[2].set_title('Validation accuracy')
    axs[2].plot(np.arange(epochs), epoch_val_acc[:, 0], 'g-', label='Hue')
    axs[2].plot(np.arange(epochs), epoch_val_acc[:, 1], 'c-', label='Chroma')
    axs[2].plot(np.arange(epochs), epoch_val_acc[:, 2], 'y-', label='Value')
    axs.flat[2].set(xlabel='Epochs', ylabel='Accuracy')
    axs[2].legend()

    plt.savefig(save_dir + '/' + hiperparameters + '.png')
    plt.close('all')
    #plt.show()

    #print('Saving data in: ' + save_dir)