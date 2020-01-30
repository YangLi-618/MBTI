import csv
import html
import json
import random
import re
import string
from collections import Counter, OrderedDict
from functools import partial, reduce
from itertools import repeat, compress, filterfalse, starmap, chain
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Tuple, List
from unicodedata import normalize as normalize_unicode

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS as SKL_STOPWORDS
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
    classification_report,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS as WC_STOPWORDS

from utils import (
    post_feature_to_user_feature,
    plot_compare_histogram,
    make_word_cloud,
    plot_compare_tsne,
    plot_confusion_matrix,
)

ROOT = Path(__file__).parent
DATA_FILE = ROOT / 'mbti_personality.csv'  # Downloaded from. https://www.kaggle.com/datasnaek/mbti-type
OUTPUT_DIR = ROOT / 'output'

# Use only a small sample of the data set to speed up the testing of your code. Set to None for disabling.
SAMPLE_DATA = None
# SAMPLE_DATA = 500  # Number of users to be used in the sample.

# Transformer for MBTI types.
TYPE_TRANSFORMER = itemgetter(3)  # Use only the J vs P dimension.

SELECT_K = 500  # Select K best TF-IDF features.

# Placeholders
URL = '__url__'
HASHTAG = '__hashtag__'
AT = '__at__'
EMOJI = '__emoji__'
NUMBER = '__number__'
PLACEHOLDERS = [URL, HASHTAG, AT, EMOJI, NUMBER]

# Placeholder patterns
URL_PATTERN = re.compile(
    r'(https?://)?(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.'
    r'[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@,:;%_+.~#?&/=]*)'
)
HASHTAG_PATTERN = re.compile(r'#([a-zA-Z0-9]+)')
AT_PATTERN = re.compile(r'@([a-zA-Z0-9]+)')
EMOJI_PATTERN = re.compile(r':([a-zA-Z0-9]+):')
NUMBER_PATTERN = re.compile(r'\d+(,\d+)*')

# Misc
PUNCTUATION_TRANS = str.maketrans('', '', string.punctuation)
STOPWORDS = frozenset(nltk_stopwords.words('english')) | WC_STOPWORDS | SKL_STOPWORDS | {'nt', 'let'}
STEMMER = SnowballStemmer('english')
SENTIMENT_ANALYSER = SentimentIntensityAnalyzer()

# Feature names
METADATA_FEATURE_NAMES = [
    'raw_length',
    'raw_words',
    'normalized_words',
    'url_count',
    'hashtag_count',
    'at_count',
    'emoji_count',
    'number_count',
]
METADATA_FEATURE_NAMES_VERBOSE = [
    'Raw Post Length',
    'Raw Post Words Count',
    'Normalized Post Words Count',
    'URL Count',
    'Hashtag Count',
    '@ Count',
    'Emoji Count',
    'Number Count',
]
assert len(METADATA_FEATURE_NAMES) == len(METADATA_FEATURE_NAMES_VERBOSE)
SENTIMENT_FEATURE_NAMES = ['pos', 'neu', 'neg', 'compound']
SENTIMENT_FEATURE_NAMES_VERBOSE = [
    'Positive Score',
    'Neutral Score',
    'Negative Score',
    'Compound Score',
]
assert len(SENTIMENT_FEATURE_NAMES) == len(SENTIMENT_FEATURE_NAMES_VERBOSE)


def load_data() -> Iterable[Tuple[int, str, str]]:
    with DATA_FILE.open('rt', encoding='utf-8', newline='') as data_file:
        reader = csv.DictReader(data_file)
        for uid, record in enumerate(reader):
            yield from zip(
                repeat(uid),
                record['posts'].strip('\'').split('|||'),
                repeat(TYPE_TRANSFORMER(record['type'].upper())),
            )


def tokenize_post(post: str) -> List[str]:
    for func in [
        # Unescape HTML characters: '&gt;' -> '>', ...
        html.unescape,
        # Replace words with placeholders
        partial(URL_PATTERN.sub, f' {URL} '),
        partial(HASHTAG_PATTERN.sub, f' {HASHTAG} '),
        partial(AT_PATTERN.sub, f' {AT} '),
        # partial(EMOJI_PATTERN.sub, f' {EMOJI} '),
        partial(EMOJI_PATTERN.sub, r'\g<1>'),
        # Number should be replaced lastly since it may be contained by other patterns.
        partial(NUMBER_PATTERN.sub, f' {NUMBER} '),
    ]:
        post = func(post)
    return word_tokenize(post)


def normalize_post(words: List[str]) -> List[str]:
    # Remove non-ascii characters
    words = map(partial(normalize_unicode, 'NFKD'), words)
    words = map(partial(str.encode, encoding='ascii', errors='ignore'), words)
    words = map(partial(bytes.decode, encoding='utf-8', errors='ignore'), words)
    # To lowercase
    words = map(str.lower, words)
    # Remove punctuation
    words = (
        word.translate(PUNCTUATION_TRANS)
        if word not in set(PLACEHOLDERS)
        else word
        for word in words
    )
    words = filter(bool, words)
    # Remove stop words
    words = filterfalse(STOPWORDS.__contains__, words)
    # Stem words
    words = map(STEMMER.stem, words)

    return list(words)


def get_metadata_features(post: str, tokenized_post: List[str], normalized_post: List[str]) -> List[int]:
    counter = Counter(normalized_post)

    raw_length = len(post)
    raw_words = len(tokenized_post)
    normalized_words = len(normalized_post)
    placeholder_counts = list(map(counter.__getitem__, PLACEHOLDERS))

    return [raw_length, raw_words, normalized_words, *placeholder_counts]


def get_sentiment_features(post: str) -> List[int]:
    return list(map(
        SENTIMENT_ANALYSER.polarity_scores(post).__getitem__,
        SENTIMENT_FEATURE_NAMES,
    ))


def test_clustering(xx, yy, norm=None, axis=0):
    if norm:
        xx = normalize(xx, norm=norm, axis=axis)
    cluster_num = len(set(yy))
    cluster_algorithms = {
        'K-Means': KMeans(cluster_num),
        'Spectral': SpectralClustering(cluster_num),
        'Hierarchical': AgglomerativeClustering(cluster_num),
    }
    metrics = OrderedDict([
        ('Adjusted Rand Index', adjusted_rand_score),
        ('Homogeneity', homogeneity_score),
        ('Completeness', completeness_score),
        ('V-Measure', v_measure_score),
        ('Fowlkes-Mallows Index', fowlkes_mallows_score),
    ])
    results = []
    for name, algorithm in cluster_algorithms.items():
        y_predict = algorithm.fit_predict(xx)
        results.append(OrderedDict([
            ('Name', name),
            *(
                (metric_name, metric_func(yy, y_predict))
                for metric_name, metric_func in metrics.items()
            ),
        ]))
    return pd.DataFrame.from_records(results)


OUTPUT_DIR.mkdir(exist_ok=True)
sns.set()

uids, posts, types = list(zip(*sorted(tqdm(load_data(), desc='Loading data'))))
if SAMPLE_DATA is not None:
    uids, posts, types = list(zip(*compress(
        tqdm(zip(uids, posts, types), desc='Sampling data'),
        map(set(random.sample(set(uids), k=SAMPLE_DATA)).__contains__, uids),
    )))
uids = np.array(uids)
posts = np.array(posts, dtype=object)
types = np.array(types)
unique_types = unique_labels(types)

tokenized_posts = np.array(list(map(
    tokenize_post,
    tqdm(posts, desc='Tokenizing posts'),
)), dtype=object)
normalized_posts = np.array(list(map(
    normalize_post,
    tqdm(tokenized_posts, desc='Normalizing posts'),
)), dtype=object)

post_metadata = np.array(list(starmap(
    get_metadata_features,
    tqdm(zip(posts, tokenized_posts, normalized_posts), desc='Extracting metadata features'),
)), dtype=np.float64)
assert post_metadata.shape == (uids.shape[0], len(METADATA_FEATURE_NAMES))

post_sentiment = np.array(list(map(
    get_sentiment_features,
    tqdm(posts, desc='Extracting sentiment features'),
)), dtype=np.float64)
assert post_sentiment.shape == (uids.shape[0], len(SENTIMENT_FEATURE_NAMES))

with (OUTPUT_DIR / 'Basic Statistic.txt').open('wt', encoding='utf-8') as f:
    printf = partial(print, file=f)
    printf('User # (Total):', len(set(uids)))
    for type_ in unique_types:
        printf(f'User # ({type_}):', len(set(uids[types == type_])))
    printf()
    printf('Post # (Total):', posts.shape[0])
    for type_ in unique_types:
        printf(f'Post # ({type_}):', np.sum(types == type_))

user_metadata = post_feature_to_user_feature(uids, post_metadata)
user_sentiment = post_feature_to_user_feature(uids, post_sentiment)
user_types = np.array(list(map(
    dict(zip(uids, types)).__getitem__,
    np.unique(uids),
)))

for data, feature_name in tqdm(chain(
        zip(user_metadata.T, METADATA_FEATURE_NAMES_VERBOSE),
        zip(user_sentiment.T, SENTIMENT_FEATURE_NAMES_VERBOSE),
), desc='Plotting histograms'):
    title = f'Histogram of {feature_name}'
    fig = plot_compare_histogram(
        data, user_types,
        normalize_support=True,
        x_label=feature_name,
        title=title,
    )
    fig.savefig(OUTPUT_DIR / f'{title}.png')
    plt.close(fig)

word_counters = [
    Counter(chain.from_iterable(normalized_posts[types == type_]))
    for type_ in unique_types
]
normalized_word_counters = [
    Counter({
        k: v / s
        for k, v, s in zip(
            counter.keys(),
            counter.values(),
            repeat(sum(counter.values())),
        )
    })
    for counter in word_counters
]
# noinspection PyTypeChecker
common_counter = reduce(Counter.__and__, normalized_word_counters)
common_top_k = 10
# noinspection PyTypeChecker
common_top_k_words = reduce(set.__and__, (
    set(map(itemgetter(0), counter.most_common(common_top_k)))
    for counter in word_counters
))
fig = plot_compare_histogram(
    *zip(*chain.from_iterable(
        zip(counter.values(), repeat(type_))
        for counter, type_ in zip(word_counters, unique_types)
    )),
    x_label='Word Frequency',
    y_label='Logarithm of # of Words',
    title='Histogram of Word Frequencies',
    hist_kws={'log': True},
)
fig.savefig(OUTPUT_DIR / 'Word Frequencies Histogram.png')
plt.close(fig)
bar = tqdm(
    zip(unique_types, word_counters, normalized_word_counters),
    desc='Working on words', total=unique_types.shape[0],
)
for type_, word_counter, normalized_word_counter in bar:
    bar.set_postfix(file='Word List')
    with (OUTPUT_DIR / f'Word List {type_}.txt').open('wt', encoding='utf-8') as f:
        printf = partial(print, file=f)
        for w, c in word_counter.most_common():
            printf(f'{w:15}   {c}')
    bar.set_postfix(file='Word Cloud')
    make_word_cloud(word_counter).to_file(OUTPUT_DIR / f'Word Cloud - {type_}.png')
    bar.set_postfix(file='Word Cloud (No All Common)')
    make_word_cloud(normalized_word_counter - common_counter) \
        .to_file(OUTPUT_DIR / f'Word Cloud (No All Common) - {type_}.png')
    bar.set_postfix(file=f'Word Cloud (No Top-{common_top_k} Common)')
    make_word_cloud({k: v for k, v in word_counter.items() if k not in common_top_k_words}) \
        .to_file(OUTPUT_DIR / f'Word Cloud (No Top-{common_top_k} Common) - {type_}.png')
del bar

print('Extracting TF-IDF features.')
vectorizer = TfidfVectorizer(
    stop_words=list(STOPWORDS),
    ngram_range=(1, 2),
    max_df=0.8,
    min_df=5,
    norm=None,
)
post_tfidf = vectorizer.fit_transform(posts.tolist())
tfidf_feature_names = vectorizer.get_feature_names()
user_tfidf = post_feature_to_user_feature(uids, post_tfidf)

print('Selecting features.')
feature_selector = SelectKBest(chi2, k=SELECT_K)
user_tfidf = feature_selector.fit_transform(user_tfidf, user_types)
# noinspection PyTypeChecker
tfidf_feature_names = list(map(
    tfidf_feature_names.__getitem__,
    feature_selector.get_support(indices=True),
))
with (OUTPUT_DIR / 'TF-IDF Feature Scores.txt').open('wt', encoding='utf-8') as f:
    printf = partial(print, file=f)
    for feature_name, score, p in sorted(
            zip(tfidf_feature_names, feature_selector.scores_, feature_selector.pvalues_),
            key=itemgetter(1), reverse=True,
    ):
        printf(f'{feature_name:20}   {score:12.6f}   {p:.6f}')

for feature_group, feature in zip(
        ['Metadata', 'Sentiment', 'TF-IDF'],
        [user_metadata, user_sentiment, user_tfidf],
):
    print(f'Plotting T-SNE of {feature_group}.')
    plot_compare_tsne(feature, user_types, verbose=True) \
        .savefig(OUTPUT_DIR / f'T-SNE of {feature_group}.png')

print('Clustering on metadata.')
with (OUTPUT_DIR / 'Clustering Result on Metadata.txt').open('wt', encoding='utf-8') as f:
    print(test_clustering(user_metadata, user_types, norm='max', axis=0).to_string(), file=f)
print('Clustering on sentiment.')
with (OUTPUT_DIR / 'Clustering Result on Sentiment.txt').open('wt', encoding='utf-8') as f:
    print(test_clustering(user_sentiment, user_types, norm='max', axis=0).to_string(), file=f)
print('Clustering on TF-IDF.')
with (OUTPUT_DIR / 'Clustering Result on TF-IDF.txt').open('wt', encoding='utf-8') as f:
    print(test_clustering(user_tfidf, user_types, norm='l2', axis=1).to_string(), file=f)

x = np.concatenate([
    user_metadata,
    user_sentiment,
    user_tfidf,
], axis=1)
y = user_types
del uids, posts, types
del post_metadata, post_sentiment, post_tfidf
del user_metadata, user_sentiment, user_tfidf, user_types,
feature_names = [
    *(f'Metadata: {name}' for name in METADATA_FEATURE_NAMES),
    *(f'Sentiment: {name}' for name in SENTIMENT_FEATURE_NAMES),
    *(f'TF-IDF: {name}' for name in tfidf_feature_names),
]
feature_group_names = ['Metadata', 'Sentiment', 'TF-IDF']
feature_group_masks = np.array([
    [True] * len(METADATA_FEATURE_NAMES) + [False] * len(SENTIMENT_FEATURE_NAMES) + [False] * len(tfidf_feature_names),
    [False] * len(METADATA_FEATURE_NAMES) + [True] * len(SENTIMENT_FEATURE_NAMES) + [False] * len(tfidf_feature_names),
    [False] * len(METADATA_FEATURE_NAMES) + [False] * len(SENTIMENT_FEATURE_NAMES) + [True] * len(tfidf_feature_names),
])

print('Evaluating classification.')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
classifiers = [
    {
        'name': 'Gaussian Naive Bayes',
        'estimator': GaussianNB(),
    },
    {
        'name': 'Bernoulli Naive Bayes',
        'estimator': BernoulliNB(),
        'param_grid': {
            'alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0],
            'binarize': [0.0, 0.1, 0.15, 0.5, 1.0, 5.0],
            'fit_prior': [True, False],
        },
    },
    {
        'name': 'K-Neighbors',
        'estimator': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [5, 10, 20, 50],
        },
    },
    {
        'name': 'Ridge',
        'estimator': RidgeClassifier(),
        'param_grid': {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'normalize': [True, False],
            'tol': [1e-3, 1e-5],
        },
    },
    {
        'name': 'Linear SGD',
        'estimator': SGDClassifier(),
        'param_grid': {
            'penalty': ['l2', 'elasticnet'],
            'alpha': [1e-5, 1e-6],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [1e-3, 1e-4, 1e-5],
        },
    },
    {
        'name': 'Linear SVC',
        'estimator': LinearSVC(dual=False, max_iter=50000),
        'param_grid': {
            'penalty': ['l2', 'l1'],
            'C': [0.01, 0.1, 0.5, 1.0, 2.0],
        },
    },
    {
        'name': 'RBF SVC',
        'estimator': SVC(),
        'param_grid': {
            'C': [0.01, 0.1, 0.5, 1.0, 2.0],
            'gamma': ['scale', 'auto', 0.5, 1.0, 2.0],
        },
    },
    {
        'name': 'Decision Tree',
        'estimator': DecisionTreeClassifier(),
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 10, 20, 50, None],
            'min_samples_split': [2, 5, 10],
            'max_features': [2, 5, 10, 'auto', None],
        },
    },
    {
        'name': 'Random Forest',
        'estimator': RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [10, 20, 50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 10, 20, 50, None],
            'min_samples_split': [2, 5, 10],
            'max_features': [2, 5, 10, 'auto'],
            'bootstrap': [True, False],
        },
    },
    {
        'name': 'Multi-layer Perceptron',
        'estimator': MLPClassifier(max_iter=10000),
        'param_grid': {
            'hidden_layer_sizes': [(10,), (50,), (100,), (50, 50), (20, 20, 20)],
            'activation': ['logistic', 'tanh', 'relu'],
            'alpha': [1e-4, 1e-2, 1, 10, 100],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        },
    },
]
feature_group_results = []
for classifier in classifiers:
    classifier_name = classifier['name']
    print(f'Classifier: {classifier_name}')
    searcher = GridSearchCV(
        estimator=classifier['estimator'],
        param_grid=classifier.get('param_grid', {}),
        scoring='balanced_accuracy',
        # n_jobs=4,
        cv=5,
        error_score=np.nan,
        verbose=1,
    )
    searcher.fit(x_train, y_train)
    best_estimator = searcher.best_estimator_
    y_pred = best_estimator.predict(x_test)
    with (OUTPUT_DIR / f'Classification Report - {classifier_name}.txt').open('wt', encoding='utf-8') as f:
        print(classification_report(y_test, y_pred, digits=5), file=f)
    with (OUTPUT_DIR / f'Classification Params - {classifier_name}.json').open('wt', encoding='utf-8') as f:
        json.dump(searcher.best_params_, f, indent=2)
    fig = plot_confusion_matrix(best_estimator, x_test, y_test, title=f'Confusion Matrix of {classifier_name}')
    fig.savefig(OUTPUT_DIR / f'Confusion Matrix - {classifier_name}.png')
    plt.close(fig)

    try:
        for feature_combination in [
            (0,), (1,), (2,),
            (0, 1), (0, 2), (1, 2),
            (0, 1, 2),
        ]:
            combination_name = ' + '.join(map(feature_group_names.__getitem__, feature_combination))
            combination_mask = reduce(np.logical_or, map(feature_group_masks.__getitem__, feature_combination))
            x_train_subset = x_train[:, combination_mask]
            x_test_subset = x_test[:, combination_mask]
            # noinspection PyUnresolvedReferences
            y_pred = classifier['estimator'] \
                .set_params(**searcher.best_params_) \
                .fit(x_train_subset, y_train) \
                .predict(x_test_subset)
            feature_group_results.append({
                'Classifier': classifier_name,
                'Combination': combination_name,
                'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
            })
    except ValueError:
        pass
feature_group_results = pd.DataFrame.from_records(feature_group_results)
sns.catplot(
    x='Balanced Accuracy', y='Classifier', hue='Combination',
    data=feature_group_results,
    orient='h', kind='bar',
).savefig(OUTPUT_DIR / 'Classification on Feature Groups.png')
with (OUTPUT_DIR / 'Classification on Feature Groups.txt').open('wt', encoding='utf-8') as f:
    print(feature_group_results.to_string(), file=f)
