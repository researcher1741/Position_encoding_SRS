import pickle
import os

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import RESULTS


def save_and_plot(mydict, mystring, model_name):
    mystring = mystring + "_" + model_name
    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)
    with open(os.path.join(RESULTS, mystring + '.pkl'), 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Show/save figure as desired.
    plt.figure()
    plt.xlabel('EPOCHS')
    plt.ylabel("")
    for k in mydict.keys():
        if len(mydict[k]) > 0 and k != "confusion":
            plt.plot(list(range(1, len(mydict[k]) + 1)), mydict[k], label=k)
    plt.title(model_name)
    plt.legend()
    plt.show()
    plt.tight_layout()
    plt.savefig(mystring + '.png')

    if "confusion" in mydict.keys():
        disp = ConfusionMatrixDisplay(confusion_matrix=mydict['confusion'][-1])
        disp.plot()
        plt.savefig('confusion_' + mystring + '.png')


def str2bool(s):
    """ Convert a boolean saved as string into a Boolean. """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def load_data(filename):
    """ Returns the saved file from a pkl or an empty list otherwise. """
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x


def save_data(data, filename):
    """ Save data as a pkl file """
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def check_device(printit=False):
    """ This function checks the cuda's setting. It prints the setting. """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if printit:
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
            print("Cuda architectures lis{}".format(torch.cuda.get_arch_list()))
            print("Device capability {}".format(torch.cuda.get_device_capability()))
    else:
        if printit:
            print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def get_mask(tensor):
    return tensor.to(torch.bool) if tensor is not None else None


def get_mask2(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0.0, 0.0, 1.0)
