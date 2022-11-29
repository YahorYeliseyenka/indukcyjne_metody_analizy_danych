import pandas as pd
import numpy
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import model_selection
import matplotlib.pyplot as plt
import os
import warnings
import csv
warnings.filterwarnings("ignore")


def get_ds_name(ds):
    if (ds.equals(datasets[0])):
        name = "glass"
    elif (ds.equals(datasets[1])):
        name = "pima"
    else:
        name = "wine"
    return name


def standardize(df):
    df.iloc[:, :-1] = StandardScaler().fit_transform(df.iloc[:, :-1])


def normalize(df):
    df.iloc[:, :-1] = Normalizer().fit_transform(df.iloc[:, :-1])


def load_dataset(filename, scale_data):
    dataset = pd.read_csv(filename + '.csv', sep=',')
    if scale_data == 1:
        standardize(dataset)
    elif scale_data == 2:
        normalize(dataset)
    return dataset


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write2file(folder_name, param, results, p=True):
    if p:
        x = folder_name + " " + param
    else:
        x = folder_name
    with open("results/" + x + "/" + param + ".csv", 'w', newline='') as myfile:
        fieldnames = [param, 'acc_glass', 'f1_glass', 'rec_glass', 'prec_glass',
                      'acc_pima', 'f1_pima', 'rec_pima', 'prec_pima',
                      'acc_wine', 'f1_wine', 'rec_wine', 'prec_wine']
        writer = csv.DictWriter(myfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(results[0])):
            writer.writerow({param: results[0][i],
                             'acc_glass': round(results[1][i], 2), 'f1_glass': round(results[2][i], 2), 'rec_glass': round(results[3][i], 2), 'prec_glass': round(results[4][i], 2),
                             'acc_pima': round(results[5][i], 2), 'f1_pima': round(results[6][i], 2), 'rec_pima': round(results[7][i], 2), 'prec_pima': round(results[8][i], 2),
                             'acc_wine': round(results[9][i], 2), 'f1_wine': round(results[10][i], 2), 'rec_wine': round(results[11][i], 2), 'prec_wine': round(results[12][i], 2),
                             })


def write_plots4results(alg_name, ds_name, label_x_name, x, acc, f_score, recall, prec):
    write_plot(alg_name=alg_name, result=[x, acc], plot_title=ds_name,
               label_x=label_x_name, label_y="accuracy")

    write_plot(alg_name=alg_name, result=[x, f_score], plot_title=ds_name,
               label_x=label_x_name, label_y="f_score")

    write_plot(alg_name=alg_name, result=[x, recall], plot_title=ds_name,
               label_x=label_x_name, label_y="recall")

    write_plot(alg_name=alg_name, result=[x, prec], plot_title=ds_name,
               label_x=label_x_name, label_y="precision")


def write_bar_plots4results(alg_name, ds_name, label_x_name, x, acc, f_score, recall, prec):
    write_bar_plot(alg_name=alg_name, result=[x, acc], plot_title=ds_name,
               label_x=label_x_name, label_y="accuracy")

    write_bar_plot(alg_name=alg_name, result=[x, f_score], plot_title=ds_name,
               label_x=label_x_name, label_y="f_score")

    write_bar_plot(alg_name=alg_name, result=[x, recall], plot_title=ds_name,
               label_x=label_x_name, label_y="recall")

    write_bar_plot(alg_name=alg_name, result=[x, prec], plot_title=ds_name,
               label_x=label_x_name, label_y="precision")


def write_plot(alg_name, result, label_x, label_y, plot_title=""):
    make_dir("results/" + alg_name + " " + label_x)
    x = alg_name + " " + plot_title + " " + label_x

    plt.clf()
    plt.plot(result[0], result[1])
    plt.plot(result[0], result[1], 'ro')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(x.upper() + "\n" + label_y.upper())
    plt.yticks(numpy.arange(0, 1.1, 0.1))
    plt.xticks(result[0])
    plt.savefig("results/" + alg_name + " " + label_x + "/" + plot_title + " " + label_y + '.png')


def write_bar_plot(alg_name, result, label_x, label_y, plot_title=""):
    make_dir("results/" + alg_name + " " + label_x)
    x = alg_name + " " + plot_title + " " + label_x

    plt.clf()
    plt.bar(result[0], result[1])
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(x.upper() + "\n" + label_y.upper())
    plt.yticks(numpy.arange(0, 1.1, 0.1))
    plt.xticks([0, 1], result[0])
    plt.savefig("results/" + alg_name + " " + label_x + "/" + plot_title + " " + label_y + '.png')


def test_bagging():

    # params
    seed = 1
    max_samples = [0.2, 0.4, 0.6, 0.8, 1.0]
    n_estimators = [20, 50, 100, 200, 400]
    max_features = [0.2, 0.4, 0.6, 0.8, 1.0]
    warm_start = [False, True]
    bootstrap = [False, True]
    bootstrap_features = [False, True]

    # cross validation
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    scores = ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro']

    # N ESTIMATORS

    m_result = [n_estimators]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in n_estimators:
            model = BaggingClassifier(base_estimator=cart, n_estimators=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("BaggingC_cart", get_ds_name(dataset), "n_estimators", n_estimators, acc, f_score, recall, prec)
    write2file('BaggingC_cart', "n_estimators", m_result)

    # MAX FEATURES

    m_result = [max_features]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in max_features:
            model = BaggingClassifier(base_estimator=cart, max_features=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("BaggingC_cart", get_ds_name(dataset), "max_features", max_features, acc, f_score, recall, prec)
    write2file('BaggingC_cart', "max_features", m_result)

    # MAX SAMPLES

    m_result = [max_samples]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in max_samples:
            model = BaggingClassifier(base_estimator=cart, max_samples=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("BaggingC_cart", get_ds_name(dataset), "max_samples", max_samples, acc, f_score, recall, prec)
    write2file('BaggingC_cart', "max_samples", m_result)

    # BOOTSTRAP

    m_result = [bootstrap]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in bootstrap:
            model = BaggingClassifier(base_estimator=cart, bootstrap=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_bar_plots4results("BaggingC_cart", get_ds_name(dataset), "bootstrap", bootstrap, acc, f_score, recall, prec)
    write2file('BaggingC_cart', "bootstrap", m_result)

    # WARM START

    m_result = [warm_start]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in warm_start:
            model = BaggingClassifier(base_estimator=cart, warm_start=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_bar_plots4results("BaggingC_cart", get_ds_name(dataset), "warm_start", warm_start, acc, f_score, recall, prec)
    write2file('BaggingC_cart', "warm_start", m_result)

    # BOOTSTRAP FEATURES

    m_result = [bootstrap_features]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in bootstrap_features:
            model = BaggingClassifier(base_estimator=cart, bootstrap_features=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_bar_plots4results("BaggingC_cart", get_ds_name(dataset), "bootstrap_features", bootstrap_features, acc, f_score, recall, prec)
    write2file('BaggingC_cart', "bootstrap_features", m_result)


def test_ada_boost():
    # params
    seed = 1
    n_estimators = [10, 20, 50, 100, 200]
    learning_rate = [.5, 1., 5., 10., 15.]

    # cross validation
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    # N ESTIMATORS

    m_result = [n_estimators]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in n_estimators:
            model = AdaBoostClassifier(base_estimator=cart, n_estimators=i, algorithm='SAMME.R', random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("AdaBoost_cart", get_ds_name(dataset), "n_estimators", n_estimators, acc, f_score, recall, prec)
    write2file('AdaBoost_cart', "n_estimators", m_result)

    # LEARNING RATE

    m_result = [learning_rate]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in learning_rate:
            model = AdaBoostClassifier(base_estimator=cart, learning_rate=i, algorithm='SAMME.R', random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("AdaBoost_cart", get_ds_name(dataset), "learning_rate", learning_rate, acc, f_score, recall, prec)
    write2file('AdaBoost_cart', "learning_rate", m_result)


def test_rf():

    # params
    seed = 1
    n_estimators = [10, 20, 50, 100, 200]
    criterion = ["gini", "entropy"]
    max_depth = [1, 2, 5, 10, 20]
    min_samples_split = [.2, .4, .6, .8, 1.]
    min_samples_leaf = [.1, .3, .5, 1]
    max_features = [.2, .4, .6, .8, 1.]
    max_leaf_nodes = [2, 5, 10, 20, 40]

    # cross validation
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    # N ESTIMATORS

    m_result = [n_estimators]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in n_estimators:
            model = RandomForestClassifier(n_estimators=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("RF", get_ds_name(dataset), "n_estimators", n_estimators, acc, f_score, recall, prec)
    write2file('RF', "n_estimators", m_result)

    # CRITERION

    m_result = [criterion]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in criterion:
            model = RandomForestClassifier(criterion=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_bar_plots4results("RF", get_ds_name(dataset), "criterion", criterion, acc, f_score, recall, prec)
    write2file('RF', "criterion", m_result)

    # MAX DEPTH

    m_result = [max_depth]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in max_depth:
            model = RandomForestClassifier(max_depth=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("RF", get_ds_name(dataset), "max_depth", max_depth, acc, f_score, recall, prec)
    write2file('RF', "max_depth", m_result)

    # MAX SAMPLES SPLIT

    m_result = [min_samples_split]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in min_samples_split:
            model = RandomForestClassifier(min_samples_split=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("RF", get_ds_name(dataset), "min_samples_split", min_samples_split, acc, f_score, recall, prec)
    write2file('RF', "min_samples_split", m_result)

    # MAX SAMPLES LEAF

    m_result = [min_samples_leaf]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in min_samples_leaf:
            model = RandomForestClassifier(min_samples_leaf=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("RF", get_ds_name(dataset), "min_samples_leaf", min_samples_leaf, acc, f_score, recall, prec)
    write2file('RF', "min_samples_leaf", m_result)

    # MAX FEATURES

    m_result = [max_features]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in max_features:
            model = RandomForestClassifier(max_features=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("RF", get_ds_name(dataset), "max_features", max_features, acc, f_score, recall,
                            prec)
    write2file('RF', "max_features", m_result)

    # MAX LEAF NODES

    m_result = [max_leaf_nodes]
    for dataset in datasets:
        acc, recall, prec, f_score = [], [], [], []
        for i in max_leaf_nodes:
            model = RandomForestClassifier(max_leaf_nodes=i, random_state=seed)
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        write_plots4results("RF", get_ds_name(dataset), "max_leaf_nodes", max_leaf_nodes, acc, f_score, recall,
                            prec)
    write2file('RF', "max_leaf_nodes", m_result)


########################################################################################################################
datasets = [load_dataset("glass", 0), load_dataset("pima", 0), load_dataset("wine", 0)]
cart = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=.1, max_features=.7,
                              random_state=1)

#test_bagging()
#test_ada_boost()
#test_rf()

####################################################################
# gridBC = ParameterGrid({"max_samples": [0.2, , 1.0],
#                           "max_features": [1, 2, 4],
#                           "bootstrap": [True, False],
#                           "bootstrap_features": [True, False]})


# def dtc():
#     seed = 1
#     criterion = ['gini', 'entropy']
#     splitter = ['best', 'random']
#     max_depth = [1, 2, 5, 10, 15]
#     min_samples_split = [.1, .2, .6, .8, 1.]
#     max_features = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
#
#     # cross validation
#     kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
#     scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
#
#     # N ESTIMATORS
#
#     m_result = [["ds_0", "ds_1", "ds_2"]]
#     for d in range(3):
#         acc, recall, prec, f_score = [], [], [], []
#         for s in range(3):
#             model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=.1,
#                                            max_features=.7, random_state=seed)
#             results = model_selection.cross_validate(model, datasets[s][d].iloc[:, :-1].values, datasets[s][d].class_dataset,
#                                                      cv=kfold, scoring=scores)
#
#             acc.append(results['test_accuracy'].mean())
#             f_score.append(results['test_f1_macro'].mean())
#             recall.append(results['test_recall_macro'].mean())
#             prec.append(results['test_precision_macro'].mean())
#
#         m_result.append(acc)
#         m_result.append(f_score)
#         m_result.append(recall)
#         m_result.append(prec)
#
#         write_plots4results("CART", get_ds_name(datasets[0][d]), "ds", ["ds_0", "ds_1", "ds_2"], acc, f_score, recall, prec)
#     write2file('CART', "ds", m_result)

def voting():
    seed = 1
    bagging = BaggingClassifier(base_estimator=cart, n_estimators=50, max_samples=1.0, max_features=0.7,
                                bootstrap_features=True, bootstrap=False, random_state=seed)
    boosting = AdaBoostClassifier(base_estimator=cart, learning_rate=1.0, n_estimators=26, random_state=seed)
    rf = RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=100, max_leaf_nodes=40,
                                min_samples_leaf=1, min_samples_split=0.2, random_state=seed)

    ex = ["EXMPL0", "EXMPL1", "EXMPL2", "EXMPL3", "EXMPL4", "EXMPL5", "EXMPL6"]

    exmpl0 = [('bagging', bagging)]
    exmpl1 = [('boosting', boosting)]
    exmpl2 = [('rf', rf)]
    exmpl3 = [('bagging', bagging), ('boosting', boosting)]
    exmpl4 = [('bagging', bagging), ('rf', rf)]
    exmpl5 = [('boosting', boosting), ('rf', rf)]
    exmpl6 = [('bagging', bagging), ('boosting', boosting), ('rf', rf)]

    exmpls = [exmpl0, exmpl1, exmpl2, exmpl3, exmpl4, exmpl5, exmpl6]

    # cross validation
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    voting = ['hard', 'soft']
    x = 0

    for e in exmpls:
        m_result = [voting]
        for dataset in datasets:
            acc, recall, prec, f_score = [], [], [], []
            for i in voting:
                model = VotingClassifier(estimators=e, voting=i)
                results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                         cv=kfold, scoring=scores)

                acc.append(results['test_accuracy'].mean())
                f_score.append(results['test_f1_macro'].mean())
                recall.append(results['test_recall_macro'].mean())
                prec.append(results['test_precision_macro'].mean())

            m_result.append(acc)
            m_result.append(f_score)
            m_result.append(recall)
            m_result.append(prec)

            write_bar_plots4results(ex[x], get_ds_name(dataset), "voting", voting, acc, f_score, recall,
                                prec)
        write2file(ex[x], "voting", m_result)
        x += 1


#voting()


# def testqwe():
#     seed = 1
#     bagging = BaggingClassifier(base_estimator=cart, n_estimators=50, max_samples=1.0, max_features=0.7,
#                                 bootstrap_features=True, bootstrap=False, random_state=seed)
#     boosting = AdaBoostClassifier(base_estimator=cart, learning_rate=1.0, n_estimators=26, random_state=seed)
#     rf = RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=100, max_leaf_nodes=40,
#                                 min_samples_leaf=1, min_samples_split=0.2, random_state=seed)
#
#     exmpls = [bagging, boosting, rf]
#     qwe = ['bagging', 'boosting', 'rf']
#     x = 0
#
#     # cross validation
#     kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
#     scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
#
#     m_result = [["Bagging", "Boosting", "RF"]]
#     for model in exmpls:
#         acc, recall, prec, f_score = [], [], [], []
#         for dataset in datasets:
#             results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
#                                                      cv=kfold, scoring=scores)
#
#             acc.append(results['test_accuracy'].mean())
#             f_score.append(results['test_f1_macro'].mean())
#             recall.append(results['test_recall_macro'].mean())
#             prec.append(results['test_precision_macro'].mean())
#
#         m_result.append(acc)
#         m_result.append(f_score)
#         m_result.append(recall)
#         m_result.append(prec)
#
#         x += 1
#     write2file("_RES", "qwe", m_result, False)
#
#
# testqwe()

def testqwe():
    seed = 1
    bagging = BaggingClassifier(base_estimator=cart, n_estimators=50, max_samples=1.0, max_features=0.7,
                                bootstrap_features=True, bootstrap=False, random_state=seed)
    boosting = AdaBoostClassifier(base_estimator=cart, learning_rate=1.0, n_estimators=26, random_state=seed)
    rf = RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=100, max_leaf_nodes=40,
                                min_samples_leaf=1, min_samples_split=0.2, random_state=seed)

    exmpls = [bagging, boosting, rf]
    qwe = ['bagging', 'boosting', 'rf']
    x = 0

    # cross validation
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    m_result = [["Bagging", "Boosting", "RF"]]
    for model in exmpls:
        acc, recall, prec, f_score = [], [], [], []
        for dataset in datasets:
            results = model_selection.cross_validate(model, dataset.iloc[:, :-1].values, dataset.class_dataset,
                                                     cv=kfold, scoring=scores)

            acc.append(results['test_accuracy'].mean())
            f_score.append(results['test_f1_macro'].mean())
            recall.append(results['test_recall_macro'].mean())
            prec.append(results['test_precision_macro'].mean())

        m_result.append(acc)
        m_result.append(f_score)
        m_result.append(recall)
        m_result.append(prec)

        x += 1
    write2file("_RES", "qwe", m_result, False)


testqwe()
