import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description='draw model heatmap')
    parser.add_argument('-d', '--data_dir', help='data dir')
    return parser.parse_args()

def load_models(model_dir):
    print model_dir
    model_weight = np.load(os.path.join(model_dir, 'model.npy'))
    return model_weight

def draw_heatmap(model1, model2, save_dir):
    ax = sns.heatmap(model1-model2)
    plt.show()

def main():
    args = get_parser()
    models = load_models(args.data_dir)
    model1 = models[0]
    model2 = models[-1]
    draw_heatmap(model1, model2, args.data_dir)


if __name__ == '__main__':
    main()
