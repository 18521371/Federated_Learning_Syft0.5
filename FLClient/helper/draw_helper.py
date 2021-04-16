import matplotlib.pyplot as plt
import numpy as np

from .config_helper import CYCLES_LOG

def plot():
    fig, axs = plt.subplots(2, figsize=(10, 10))
    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")
    offset = 0
    for i, cycle_log in enumerate(CYCLES_LOG):
        losses, accuracies = cycle_log
        x = range(offset, offset + len(losses))
        axs[0].plot(x, losses)
        axs[1].plot(x, accuracies)
        offset += len(losses)
        print(f"[INFO]\tCycle {i + 1}:\t\tLoss: {np.mean(losses)}\tAcc: {np.mean(accuracies)}")
    plt.show(block=True)
    