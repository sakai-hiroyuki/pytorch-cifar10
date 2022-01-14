import pandas as pd
from matplotlib import pyplot as plt


if __name__ == '__main__':
    opt_dict = {
        'adam': 'results/csv/efficientnet-b0/adam.csv',
    }

    fig, axes = plt.subplots(2, 2, tight_layout=True)
    for opt_name in opt_dict:
        df = pd.read_csv(opt_dict[opt_name])
        epochs = range(len(df))

        axes[0, 0].plot(epochs, df.train_loss.values, label=opt_name)
        axes[0, 1].plot(epochs, df.train_acc.values, label=opt_name)
        axes[1, 0].plot(epochs, df.test_loss.values, label=opt_name)
        axes[1, 1].plot(epochs, df.test_acc.values, label=opt_name)

    axes[0, 0].set_yscale('log')
    axes[1, 0].set_yscale('log')
    axes[0, 0].set_ylabel('train loss')
    axes[0, 1].set_ylabel('train acc')
    axes[1, 0].set_ylabel('test loss')
    axes[1, 1].set_ylabel('test acc')
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlabel('epoch')
            axes[i, j].grid(axis='both', which='major')
            axes[i, j].legend(loc='lower left')
    plt.show()
