import math
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix as skl_plot_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from wordcloud import WordCloud


def math_lcm(x, y):
    return abs(x * y) // math.gcd(x, y)


def post_feature_to_user_feature(user_ids, features):
    assert len(user_ids.shape) == 1
    assert len(features.shape) == 2
    assert user_ids.shape[0] == features.shape[0]
    unique_user_ids = np.unique(user_ids)
    results = np.empty([unique_user_ids.shape[0], features.shape[1]], dtype=np.float64)
    for row, user_id in enumerate(unique_user_ids):
        results[row, :] = np.mean(features[user_ids == user_id], axis=0)
    return results


def plot_compare_histogram(
        x, y,
        normalize_support=False,
        x_label='', y_label='', title='',
        hist_kws=None,
        **kwargs,
):
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    assert x.shape[0] == y.shape[0]

    hist_kws = hist_kws or {}
    hist_kws.setdefault('histtype', 'step')

    unique_y = unique_labels(y)
    xs = [x[y == value] for value in unique_y]
    if normalize_support:
        # noinspection PyUnresolvedReferences
        lcm = reduce(math_lcm, [xx.shape[0] for xx in xs])
        lcm = min(lcm, 50000)
        xs = [np.tile(xx, int(lcm / xx.shape[0])) for xx in xs]

    fig, ax = plt.subplots()
    for xx, value in zip(xs, unique_y):
        sns.distplot(
            xx,
            kde=False,
            rug=False,
            hist_kws=hist_kws,
            norm_hist=normalize_support,
            label=value,
            ax=ax, **kwargs,
        )
    ax.legend(unique_y)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    return fig


def make_word_cloud(counter, width=500, height=500, ellipse=False):
    if ellipse:
        mask = Image.new('RGB', (width, height), color='white')
        ImageDraw.Draw(mask).ellipse(xy=[(0, 0), (width, height)], fill='black')
        mask = np.asarray(mask)
    else:
        mask = None
    return WordCloud(
        width=width,
        height=height,
        mask=mask,
        background_color='white',
        max_words=len(counter),
    ).fit_words(counter)


def plot_compare_tsne(x, y, pca=500, verbose=False):
    assert x.shape[0] == y.shape[0]
    if len(x.shape) != 2:
        raise ValueError('x must be a 2-D matrix.')
    if x.shape[1] < 2:
        raise ValueError('x must have at least 2 features.')
    if x.shape[1] > 2:
        if pca is not None and x.shape[1] > pca:
            x = PCA(n_components=pca).fit_transform(x)
        x = TSNE(verbose=verbose).fit_transform(x)

    data = pd.DataFrame.from_dict({'x': x[:, 0], 'y': x[:, 1], 'Class': y})
    return sns.relplot(
        x='x', y='y', hue='Class', data=data,
        hue_order=unique_labels(y),
    )


def plot_confusion_matrix(estimator, x, y, title=''):
    fig, ax = plt.subplots()
    skl_plot_confusion_matrix(estimator, x, y, normalize='true', ax=ax)
    if title:
        ax.set_title(title)
    return fig
