import os.path
import torch


def Early_Stopping(monitor, patience, verbose=0):
    min_loss = min(monitor)
    early_stop = False
    no_improve = 0
    if monitor[-1] < min_loss:
        min_loss = monitor
    else:
        no_improve += 1
    if no_improve == patience:
        early_stop = True
        if verbose:
            print("early stopping")
    return early_stop


def Model_checkpoint(path, model, monitor, verbose=0, file_name="best.pth"):
    name = os.path.join(path, file_name)
    if monitor[-1] < min(monitor):
        if monitor.index(min(monitor)) == len(monitor):
            os.remove(name)
        torch.save(model, name)
        if verbose:
            print("model improved in this epoch")

#
# def CSV_log(path,filename):
