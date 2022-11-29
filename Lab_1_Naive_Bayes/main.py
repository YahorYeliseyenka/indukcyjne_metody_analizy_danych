import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import KBinsDiscretizer


def discretization_by_length(p_data, p_buckets_num):
    v_data = p_data.copy()
    for i in v_data.keys()[:-1]:
        min = v_data[i].min()
        max = v_data[i].max()+0.1
        one_bucket_range = (max - min) / p_buckets_num

        bins = [min, max]

        start_val = min
        for _ in range(p_buckets_num - 1):
            start_val += one_bucket_range
            bins.append(start_val)

        bins.sort()
        v_data[i] = np.digitize(v_data[i], bins)
    return v_data


def discretization_by_density(p_data, p_buckets_num):
    v_data = p_data.copy()
    v_bucket_size = math.floor(len(v_data) / p_buckets_num)

    for i in v_data.keys()[:-1]:
        v_data = v_data.sort_values(by=[i])
        var = 1
        for j in range(len(v_data)):
            v_data.at[j, i] = var
            if (j+1) % v_bucket_size == 0:
                if (j+1) / v_bucket_size != p_buckets_num:
                    var += 1

    v_data.sort_index(inplace=True)
    return v_data


def discretization_KMeans(p_data, p_buckets_num):
    v_data = p_data.copy()
    est = KBinsDiscretizer(n_bins=p_buckets_num, encode='ordinal', strategy='uniform')
    v_data.iloc[:,:-1] = est.fit_transform(v_data.iloc[:,:-1])
    return v_data


def mean_variance(p_data, p_target):
    n_features = p_data.shape[1]
    classes = set(p_target)
    n_classes = len(classes)
    bin_count = np.bincount(p_target)[np.nonzero(np.bincount(p_target))[0]]

    mean = np.ones((n_classes, n_features))
    variance = np.ones((n_classes, n_features))

    for i in range(n_classes):
        n_class = classes.pop()
        x = np.ones((bin_count[i], n_features))

        count = 0
        for j in range(p_data.shape[0]):
            if p_target[j] == n_class:
                x[count] = p_data[j]
                count += 1

        for j in range(n_features):
            mean[i][j] = np.mean(x.T[j])
            variance[i][j] = np.var([x.T[j]])

    return mean+1, variance+1


def prior_probability(p_target):
    classes = set(p_target)
    target_dict = collections.Counter(p_target)
    prior_pr = np.ones(len(target_dict))
    for i in range(len(target_dict)):
        prior_pr[i] = target_dict[list(classes)[i]]/p_target.shape[0]
    return prior_pr


def posterior_probability(p_mean, p_variance, p_test_data):
    n_features = p_mean.shape[1]
    n_classes = p_mean.shape[0]
    posterior_pr = np.ones(n_classes)
    for i in range(n_classes):
        product = 1
        for j in range(0, n_features):
            product = product * (1/np.sqrt(2 * math.pi * (p_variance[i][j]))) * np.exp(-0.5 * pow((p_test_data[j] - p_mean[i][j]),2)/(p_variance[i][j]))
        posterior_pr[i] = product
    return posterior_pr


def gaussian_naive_bayes(p_data, p_test_data):
    v_data = p_data.iloc[:,:-1].values
    v_target = p_data.class_dataset.values
    v_test_data = p_test_data.iloc[:,:-1].values

    mean, variance = mean_variance(v_data, v_target)
    classes = set(v_target)
    n_classes = mean.shape[0]
    prior_pr = prior_probability(v_target)

    result = []

    for j in range(len(p_test_data)):
        pcf = np.ones(n_classes)
        total_prob = 0

        posterior_pr = posterior_probability(mean, variance, v_test_data[j])

        for i in range(n_classes):
            total_prob += (posterior_pr[i] * prior_pr[i])
        for i in range(n_classes):
            pcf[i] = (posterior_pr[i] * prior_pr[i])/total_prob
        prediction = list(classes)[int(pcf.argmax())]
        result.append(prediction)

    return np.array(result)


def cross_validation(title, img_title, p_data, p_fold_num, p_stratified, p_shuffle):
    classes = len(set(p_data.class_dataset.values))

    m_confusion_matrix = np.zeros((classes, classes))
    m_accurancy_score = 0
    m_recall_score = 0
    m_precision_score = 0
    m_f_score = 0

    if p_stratified:
        kf = model_selection.StratifiedKFold(n_splits=p_fold_num, shuffle=p_shuffle)
        for train_index, test_index in kf.split(p_data, p_data.class_dataset.values):
            result = gaussian_naive_bayes(p_data.iloc[train_index], p_data.iloc[test_index])

            matrix_of_truth = p_data.iloc[test_index].class_dataset.values

            t_confusion_matrix = confusion_matrix(p_data.iloc[test_index].class_dataset.values, result)
            if len(t_confusion_matrix) < len(m_confusion_matrix):
                t_confusion_matrix = np.insert(t_confusion_matrix, len(t_confusion_matrix), 1, 0)
                t_confusion_matrix = np.insert(t_confusion_matrix, len(t_confusion_matrix) - 1, 1, 1)

            m_confusion_matrix += t_confusion_matrix
            m_accurancy_score += accuracy_score(matrix_of_truth, result)
            m_recall_score += recall_score(matrix_of_truth, result, average='macro')
            m_precision_score += precision_score(matrix_of_truth, result, average='macro')
            m_f_score += f1_score(matrix_of_truth, result, average='macro')
    else:
        kf = model_selection.KFold(n_splits=p_fold_num, shuffle=p_shuffle)
        for train_index, test_index in kf.split(p_data):
            result = gaussian_naive_bayes(p_data.iloc[train_index], p_data.iloc[test_index])

            matrix_of_truth = p_data.iloc[test_index].class_dataset.values

            t_confusion_matrix = confusion_matrix(p_data.iloc[test_index].class_dataset.values, result)
            while len(t_confusion_matrix) < len(m_confusion_matrix):
                t_confusion_matrix = np.insert(t_confusion_matrix, len(t_confusion_matrix), 0, 0)
                t_confusion_matrix = np.insert(t_confusion_matrix, len(t_confusion_matrix) - 1, 0, 1)

            m_confusion_matrix += t_confusion_matrix
            m_accurancy_score += accuracy_score(matrix_of_truth, result)
            m_recall_score += recall_score(matrix_of_truth, result, average='macro')
            m_precision_score += precision_score(matrix_of_truth, result, average='macro')
            m_f_score += f1_score(matrix_of_truth, result, average='macro')

    # m_confusion_matrix /= len(test_index)
    m_accurancy_score /= p_fold_num
    m_recall_score /= p_fold_num
    m_precision_score /= p_fold_num
    m_f_score /= p_fold_num

    m_confusion_matrix = np.round(m_confusion_matrix, 2)

    sns.heatmap(m_confusion_matrix, square=True, annot=True, xticklabels=set(p_data.class_dataset.values), yticklabels=set(p_data.class_dataset.values))
    plt.title(title)
    plt.xlabel('TRUE')
    plt.ylabel('PREDICTED')
    plt.savefig(img_title + '.png')
    plt.show()

    # print("CONFUSION MATRIX:\n", m_confusion_matrix)
    # print("ACCURANCY: ", m_accurancy_score)
    # print("RECALL SCORE: ", m_recall_score)
    # print("PERCISION: ", m_precision_score)
    # print("F-SCORE: ", m_f_score)

    return m_accurancy_score, m_recall_score, m_precision_score, m_f_score


def f_main(dataset_name, discretization_on, d_type, d_buckets_num, cv_fold_num, cv_stratified, cv_shuffle):
    title = dataset_name.upper() + "\n"
    img_title = dataset_name.upper()

    dataset = pd.read_csv(dataset_name + '.csv', sep=',')
    v_data = dataset.copy()

    if discretization_on:
        title += " Discretization: TYPE = " + d_type + ", BUCKETS= " + str(d_buckets_num) + "\n"
        img_title += "_Discretization_TYPE_" + d_type + "_BUCKETS_" + str(d_buckets_num)
        if d_type == 'L':
            v_data = discretization_by_length(dataset, d_buckets_num)
        elif d_type == 'D':
            v_data = discretization_by_density(dataset, d_buckets_num)
        elif d_type == 'K':
            v_data = discretization_KMeans(dataset, d_buckets_num)

    title += "Cross Validation: STRATIFIED = " + str(cv_stratified) + ", SHUFFLE = " + str(cv_shuffle) + ", FOLDS = " + str(cv_fold_num)
    img_title += "_Cross_Validation_STRATIFIED_" + str(cv_stratified) + "_SHUFFLE_" + str(cv_shuffle) + "_FOLDS_" + str(cv_fold_num)
    return cross_validation(title, img_title, v_data, cv_fold_num, cv_stratified, cv_shuffle)


################################################################################################################################################

def save_results(discretization_on, stratified, shuffle):
    datasetnames = ['pima', 'wine', 'glass']
    discretization_type = ['L', 'D', 'K']
    folds = [2, 3, 5, 10]
    buckets = [2,5,10,20,50,100]
    y = [0.,.2,.4,.6,.8,1]

    if discretization_on:
        for dtset_nms in datasetnames:
            for t in discretization_type:
                acc_score = []
                rec_score = []
                pre_score = []
                f_score = []
                for b in buckets:
                    acc, rec, pre, fsc = f_main(dtset_nms, discretization_on, t, b, 10, stratified, shuffle)
                    acc_score.append(acc)
                    rec_score.append(rec)
                    pre_score.append(pre)
                    f_score.append(fsc)
                ttl = dtset_nms.upper() + "\nDiscretization: TYPE = " + t + "\nCross Validation: STRATIFIED = " + str(
                    stratified) + ", SHUFFLE = " + str(shuffle) + ", FOLDS = " + str(10)
                img_ttl = dtset_nms.upper() + '_Discretization_TYPE_' + t + '_Cross_Validation_STRATIFIED_' + str(
                    stratified) + '_SHUFFLE_' + str(shuffle) + "_FOLDS_" + str(10) + '_FINAL'
                plt.plot(buckets, acc_score, label='acc')
                plt.plot(buckets, rec_score, label='rec')
                plt.plot(buckets, pre_score, label='pre')
                plt.plot(buckets, f_score, label='fsc')
                plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)
                plt.xlabel('fold_num')
                plt.ylabel('precision')
                plt.title(ttl)
                plt.yticks(y)
                plt.xticks(buckets)
                plt.savefig(img_ttl + '.png')
                plt.show()
    else:
        for dtset_nms in datasetnames:
            acc_score = []
            rec_score = []
            pre_score = []
            f_score = []
            for f in folds:
                acc, rec, pre, fsc = f_main(dtset_nms, discretization_on, 'D', 25, f, stratified, shuffle)
                acc_score.append(acc)
                rec_score.append(rec)
                pre_score.append(pre)
                f_score.append(fsc)
            ttl = dtset_nms.upper() + "\nCross Validation: STRATIFIED = " + str(stratified) + ", SHUFFLE = " + str(shuffle)
            img_ttl = dtset_nms.upper() + '_Cross_Validation_STRATIFIED_' + str(stratified) + '_SHUFFLE_' + str(shuffle) + '_FINAL'

            plt.plot(folds, acc_score, label='acc')
            plt.plot(folds, rec_score, label='rec')
            plt.plot(folds, pre_score, label='pre')
            plt.plot(folds, f_score, label='fsc')
            plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)
            plt.xlabel('fold_num')
            plt.ylabel('precision')
            plt.title(ttl)
            plt.yticks(y)
            plt.xticks(folds)
            plt.savefig(img_ttl + '.png')
            plt.show()


#dicretization OFF  stratified FALSE  shuffle FALSE
save_results(False, False, False)
#dicretization OFF  stratified FALSE  shuffle TRUE
save_results(False, False, True)
#dicretization OFF  stratified TRUE  shuffle FAlSE
save_results(False, True, False)
#dicretization OFF  stratified TRUE  shuffle TRUE
save_results(False, True, True)

#dicretization On
# save_results(True, True, True)