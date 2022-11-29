import os
import sys
import csv
import math
import numpy
import operator
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def standardize(df):
    df.iloc[:, :-1] = StandardScaler().fit_transform(df.iloc[:, :-1])


def normalize(df):
    df.iloc[:, :-1] = Normalizer().fit_transform(df.iloc[:, :-1])


def load_dataset(filename, scale_data=0):
    dataset = pd.read_csv(filename + '.csv', sep=',')
    if scale_data == 1:
        standardize(dataset)
    elif scale_data == 2:
        normalize(dataset)
    return dataset


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def manhattan_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(instance1[x] - instance2[x])
    return distance


def get_neighbors(training_set, test_instance, k, odleglosc=0):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        if odleglosc==0:
            dist = euclidean_distance(test_instance, training_set.iloc[x], length)
        else:
            dist = manhattan_distance(test_instance, training_set.iloc[x], length)
        distances.append((training_set.iloc[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors


def get_response(neighbors, response_type):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0][-1]
        distance = neighbors[x][1]
        if response_type == 0:
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        elif response_type == 1:
            if response in class_votes:
                if neighbors[x][1] == 0:
                    class_votes[response] += sys.maxsize
                else:
                    class_votes[response] += 1 / distance
            else:
                class_votes[response] = 1 / distance
        elif response_type == 2:
            if response in class_votes:
                if neighbors[x][1] == 0:
                    class_votes[response] += sys.maxsize
                else:
                    class_votes[response] += 1 / pow(distance, 2)
            else:
                class_votes[response] = 1 / pow(distance, 2)
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def knn(training_set, test_set, k, odleglosc=0, voting=0):
    predictions = []
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set.iloc[x], k, odleglosc)
        result = get_response(neighbors, voting)
        predictions.append(result)
    return predictions


def cross_validation(p_data, p_fold_num, p_stratified):

    temp_val = math.floor(len(p_data) / p_fold_num / 10)
    k_range = [i * temp_val + 2 for i in range(10)]

    acc = []
    recall = []
    prec = []
    f_score = []

    for k in k_range:
        m_accuracy = 0
        m_recall = 0
        m_precision = 0
        m_f_score = 0

        if p_stratified:
            kf = model_selection.StratifiedKFold(n_splits=p_fold_num, shuffle=True)
            for train_index, test_index in kf.split(p_data, p_data.class_dataset.values):
                matrix_of_truth = p_data.iloc[test_index].class_dataset.values
                result = knn(p_data.iloc[train_index], p_data.iloc[test_index], k)

                m_accuracy += accuracy_score(matrix_of_truth, result)
                m_recall += recall_score(matrix_of_truth, result, average='macro')
                m_precision += precision_score(matrix_of_truth, result, average='macro')
                m_f_score += f1_score(matrix_of_truth, result, average='macro')
        else:
            kf = model_selection.KFold(n_splits=p_fold_num, shuffle=True)
            for train_index, test_index in kf.split(p_data):
                matrix_of_truth = p_data.iloc[test_index].class_dataset.values
                result = knn(p_data.iloc[train_index], p_data.iloc[test_index], k)

                m_accuracy += accuracy_score(matrix_of_truth, result)
                m_recall += recall_score(matrix_of_truth, result, average='macro')
                m_precision += precision_score(matrix_of_truth, result, average='macro')
                m_f_score += f1_score(matrix_of_truth, result, average='macro')

        acc.append(m_accuracy / p_fold_num)
        recall.append(m_recall / p_fold_num)
        prec.append(m_precision / p_fold_num)
        f_score.append(m_f_score / p_fold_num)

    return k_range, acc, recall, prec, f_score


def main(filename):
    direc = filename + "_scale"
    create_folder(direc)
    for scale in (0, 1, 2):
        arr2 = []
        arr3 = []
        arr5 = []
        arr10 = []
        dataset = load_dataset(filename, scale)
        for folds in (2, 3, 5, 10):
            for stratified in (False, True):
                print("scale: " + str(scale) + ", folds: " + str(folds) + ", stratified: " + str(stratified))
                k_range, acc, rec, prec, fsc = cross_validation(dataset, folds, stratified)
                if folds == 2:
                    if len(arr2) == 0:
                        arr2.append(k_range)
                    arr2.append(acc)
                    arr2.append(rec)
                    arr2.append(prec)
                    arr2.append(fsc)
                elif folds == 3:
                    if len(arr3) == 0:
                        arr3.append(k_range)
                    arr3.append(acc)
                    arr3.append(rec)
                    arr3.append(prec)
                    arr3.append(fsc)
                elif folds == 5:
                    if len(arr5) == 0:
                        arr5.append(k_range)
                    arr5.append(acc)
                    arr5.append(rec)
                    arr5.append(prec)
                    arr5.append(fsc)
                elif folds == 10:
                    if len(arr10) == 0:
                        arr10.append(k_range)
                    arr10.append(acc)
                    arr10.append(rec)
                    arr10.append(prec)
                    arr10.append(fsc)

        write_arr(direc, "scale=" + str(scale) + "_folds=2", arr2, scale, 2)
        write_arr(direc, "scale=" + str(scale) + "_folds=3", arr3, scale, 3)
        write_arr(direc, "scale=" + str(scale) + "_folds=5", arr5, scale, 5)
        write_arr(direc, "scale=" + str(scale) + "_folds=10", arr10, scale, 10)


def write_arr(direc, fname, result, scale, folds):
    ylist = numpy.arange(0, 1.1, 0.2)
    write_results_to_file(direc, fname, result)

    plt.clf()
    plt.plot(result[0], result[1], label='acc_n_strat')
    plt.plot(result[0], result[5], label='acc_strat')
    plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)
    plt.xlabel('k')
    plt.ylabel('acc')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nACCURACY")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_acc" + '.png')

    plt.clf()
    plt.plot(result[0], result[2], label='rec_n_strat')
    plt.plot(result[0], result[6], label='rec_strat')
    plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nRECALL")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_rec" + '.png')

    plt.clf()
    plt.plot(result[0], result[3], label='prec_n_strat')
    plt.plot(result[0], result[7], label='prec_strat')
    plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nPRECISION")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_prec" + '.png')

    plt.clf()
    plt.plot(result[0], result[4], label='fsc_n_strat')
    plt.plot(result[0], result[8], label='fsc_strat')
    plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nF-SCORE")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_fsc" + '.png')


def create_folder(folder_name):
    os.mkdir("results/" + folder_name)


def write_results_to_file(folder_name, file_name, results):
    with open("results/" + folder_name + "/" + file_name + ".csv", 'w', newline='') as myfile:
        fieldnames = ['k', 'acc_n_stat', 'acc_stat', 'rec_n_stat', 'rec_stat', 'prec_n_stat', 'prec_stat',
                      'fsc_n_stat', 'fsc_stat']
        writer = csv.DictWriter(myfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(results[0])):
            writer.writerow({'k': results[0][i],
                             'acc_n_stat': round(results[1][i], 2), 'acc_stat': round(results[5][i], 2),
                             'rec_n_stat': round(results[2][i], 2), 'rec_stat': round(results[6][i], 2),
                             'prec_n_stat': round(results[3][i], 2), 'prec_stat': round(results[7][i], 2),
                             'fsc_n_stat': round(results[4][i], 2), 'fsc_stat': round(results[8][i], 2)
                             })


#################################################################################################################

def cross_validation_2(p_data, p_fold_num, p_stratified, odleglosc):

    temp_val = math.floor(len(p_data) / p_fold_num / 10)
    k_range = [i * temp_val + 2 for i in range(10)]

    acc = []
    recall = []
    prec = []
    f_score = []

    for k in k_range:
        m_accuracy = 0
        m_recall = 0
        m_precision = 0
        m_f_score = 0

        print(k)

        if p_stratified:
            kf = model_selection.StratifiedKFold(n_splits=p_fold_num, shuffle=True)
            for train_index, test_index in kf.split(p_data, p_data.class_dataset.values):
                matrix_of_truth = p_data.iloc[test_index].class_dataset.values
                result = knn(p_data.iloc[train_index], p_data.iloc[test_index], k, odleglosc)

                m_accuracy += accuracy_score(matrix_of_truth, result)
                m_recall += recall_score(matrix_of_truth, result, average='macro')
                m_precision += precision_score(matrix_of_truth, result, average='macro')
                m_f_score += f1_score(matrix_of_truth, result, average='macro')
        else:
            kf = model_selection.KFold(n_splits=p_fold_num, shuffle=True)
            for train_index, test_index in kf.split(p_data):
                matrix_of_truth = p_data.iloc[test_index].class_dataset.values
                result = knn(p_data.iloc[train_index], p_data.iloc[test_index], k, odleglosc)

                m_accuracy += accuracy_score(matrix_of_truth, result)
                m_recall += recall_score(matrix_of_truth, result, average='macro')
                m_precision += precision_score(matrix_of_truth, result, average='macro')
                m_f_score += f1_score(matrix_of_truth, result, average='macro')

        acc.append(m_accuracy / p_fold_num)
        recall.append(m_recall / p_fold_num)
        prec.append(m_precision / p_fold_num)
        f_score.append(m_f_score / p_fold_num)

    return k_range, acc, recall, prec, f_score


def write_arr_2(direc, fname, result, scale, folds):
    ylist = numpy.arange(0, 1.1, 0.2)
    write_results_to_file_2(direc, fname, result)

    plt.clf()
    plt.plot(result[0], result[1], label='acc_euclidean')
    plt.plot(result[0], result[5], label='acc_manhattan')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('acc')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nACCURACY")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_acc" + '.png')

    plt.clf()
    plt.plot(result[0], result[2], label='rec_euclidean')
    plt.plot(result[0], result[6], label='rec_manhattan')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nRECALL")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_rec" + '.png')

    plt.clf()
    plt.plot(result[0], result[3], label='prec_euclidean')
    plt.plot(result[0], result[7], label='prec_manhattan')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nPRECISION")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_prec" + '.png')

    plt.clf()
    plt.plot(result[0], result[4], label='fsc_euclidean')
    plt.plot(result[0], result[8], label='fsc_manhattan')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nF-SCORE")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_fsc" + '.png')


def write_results_to_file_2(folder_name, file_name, results):
    with open("results/" + folder_name + "/" + file_name + ".csv", 'w', newline='') as myfile:
        fieldnames = ['k', 'acc_eucl', 'acc_manh', 'rec_eucl', 'rec_manh', 'prec_eucl', 'prec_manh',
                      'fsc_eucl', 'fsc_manh']
        writer = csv.DictWriter(myfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(results[0])):
            writer.writerow({'k': results[0][i],
                             'acc_eucl': round(results[1][i], 2), 'acc_manh': round(results[5][i], 2),
                             'rec_eucl': round(results[2][i], 2), 'rec_manh': round(results[6][i], 2),
                             'prec_eucl': round(results[3][i], 2), 'prec_manh': round(results[7][i], 2),
                             'fsc_eucl': round(results[4][i], 2), 'fsc_manh': round(results[8][i], 2)
                             })


def main_2(filename):
    odl = [0, 1]
    direc = filename + "_odleglosc_2"
    create_folder(direc)
    #for scale in (0, 1, 2):
    scale = 0
    folds = 10
    stratified = True
    # arr2 = []
    # arr3 = []
    # arr5 = []
    arr10 = []
    dataset = load_dataset(filename, scale)
    #for folds in (10):
    #for stratified in (False, True):
    for odleg in odl:
        print(filename + "scale: " + str(scale) + ", folds: " + str(folds) + ", stratified: " + str(stratified) +
              ", odleg: " + str(odleg))
        k_range, acc, rec, prec, fsc = cross_validation_2(dataset, folds, stratified, odleg)

        if len(arr10) == 0:
            arr10.append(k_range)
        arr10.append(acc)
        arr10.append(rec)
        arr10.append(prec)
        arr10.append(fsc)

    write_arr_2(direc, "scale=" + str(scale) + "_folds=10", arr10, scale, 10)


if not sys.warnoptions:
    warnings.simplefilter("ignore")


###############################################################################################################


def cross_validation_3(p_data, p_fold_num, p_stratified, odleglosc, voting):

    temp_val = math.floor(len(p_data) / p_fold_num / 10)
    k_range = [i * temp_val + 2 for i in range(10)]

    acc = []
    recall = []
    prec = []
    f_score = []

    for k in k_range:
        m_accuracy = 0
        m_recall = 0
        m_precision = 0
        m_f_score = 0

        print(k)

        if p_stratified:
            kf = model_selection.StratifiedKFold(n_splits=p_fold_num, shuffle=True)
            for train_index, test_index in kf.split(p_data, p_data.class_dataset.values):
                matrix_of_truth = p_data.iloc[test_index].class_dataset.values
                result = knn(p_data.iloc[train_index], p_data.iloc[test_index], k, odleglosc, voting)

                m_accuracy += accuracy_score(matrix_of_truth, result)
                m_recall += recall_score(matrix_of_truth, result, average='macro')
                m_precision += precision_score(matrix_of_truth, result, average='macro')
                m_f_score += f1_score(matrix_of_truth, result, average='macro')
        else:
            kf = model_selection.KFold(n_splits=p_fold_num, shuffle=True)
            for train_index, test_index in kf.split(p_data):
                matrix_of_truth = p_data.iloc[test_index].class_dataset.values
                result = knn(p_data.iloc[train_index], p_data.iloc[test_index], k, odleglosc, voting)

                m_accuracy += accuracy_score(matrix_of_truth, result)
                m_recall += recall_score(matrix_of_truth, result, average='macro')
                m_precision += precision_score(matrix_of_truth, result, average='macro')
                m_f_score += f1_score(matrix_of_truth, result, average='macro')

        acc.append(m_accuracy / p_fold_num)
        recall.append(m_recall / p_fold_num)
        prec.append(m_precision / p_fold_num)
        f_score.append(m_f_score / p_fold_num)

    return k_range, acc, recall, prec, f_score


def write_arr_3(direc, fname, result, scale, folds):
    ylist = numpy.arange(0, 1.1, 0.2)
    write_results_to_file_3(direc, fname, result)

    plt.clf()
    plt.plot(result[0], result[1], label='acc_majority')
    plt.plot(result[0], result[5], label='acc_distance')
    plt.plot(result[0], result[9], label='acc_inverse_distance_sqr')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('acc')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nACCURACY")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_acc" + '.png')

    plt.clf()
    plt.plot(result[0], result[2], label='rec_majority')
    plt.plot(result[0], result[6], label='rec_distance')
    plt.plot(result[0], result[10], label='rec_inverse_distance_sqr')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nRECALL")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_rec" + '.png')

    plt.clf()
    plt.plot(result[0], result[3], label='prec_majority')
    plt.plot(result[0], result[7], label='prec_distance')
    plt.plot(result[0], result[11], label='prec_inverse_distance_sqr')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nPRECISION")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_prec" + '.png')

    plt.clf()
    plt.plot(result[0], result[4], label='fsc_majority')
    plt.plot(result[0], result[8], label='fsc_distance')
    plt.plot(result[0], result[12], label='fsc_inverse_distance_sqr')
    plt.legend(loc='lower left')
    plt.xlabel('k')
    plt.ylabel('rec')
    plt.title("scale=" + str(scale) + ", folds=" + str(folds) + "\nF-SCORE")
    plt.yticks(ylist)
    plt.xticks(result[0])
    plt.savefig("results/" + direc + "/" + fname + "_fsc" + '.png')


def write_results_to_file_3(folder_name, file_name, results):
    with open("results/" + folder_name + "/" + file_name + ".csv", 'w', newline='') as myfile:
        fieldnames = ['k', 'acc_maj', 'acc_dis', 'acc_inv_dis_sqr', 'rec_maj', 'rec_dis', 'rec_inv_dis_sqr', 'prec_maj', 'prec_dis', 'prec_inv_dis_sqr', 'fsc_maj', 'fsc_dis', 'fsc_inv_dis_sqr']
        writer = csv.DictWriter(myfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(results[0])):
            writer.writerow({'k': results[0][i],
                             'acc_maj': round(results[1][i], 2), 'acc_dis': round(results[5][i], 2), 'acc_inv_dis_sqr': round(results[9][i], 2),
                             'rec_maj': round(results[2][i], 2), 'rec_dis': round(results[6][i], 2), 'rec_inv_dis_sqr': round(results[10][i], 2),
                             'prec_maj': round(results[3][i], 2), 'prec_dis': round(results[7][i], 2), 'prec_inv_dis_sqr': round(results[11][i], 2),
                             'fsc_maj': round(results[4][i], 2), 'fsc_dis': round(results[8][i], 2), 'fsc_inv_dis_sqr': round(results[12][i], 2)
                             })


def main_3(filename):
    odleg = 1
    vots = [0, 1, 2]
    direc = filename + "_voting_2"
    create_folder(direc)
    #for scale in (0, 1, 2):
    scale = 0
    folds = 10
    stratified = True
    # arr2 = []
    # arr3 = []
    # arr5 = []
    arr10 = []
    dataset = load_dataset(filename, scale)
    #for folds in (10):
    #for stratified in (False, True):
    for voting in vots:
        print(filename + "scale: " + str(scale) + ", folds: " + str(folds) + ", stratified: " + str(stratified) +
              ", odleg: " + str(odleg) + ", dist: " + str(voting))
        k_range, acc, rec, prec, fsc = cross_validation_3(dataset, folds, stratified, odleg, voting)

        if len(arr10) == 0:
            arr10.append(k_range)
        arr10.append(acc)
        arr10.append(rec)
        arr10.append(prec)
        arr10.append(fsc)

    write_arr_3(direc, "scale=" + str(scale) + "_folds=10", arr10, scale, 10)


main_2("iris")
main_3("iris")
main_2("wine")
main_3("wine")
main_2("seeds")
main_3("seeds")
main_2("glass")
main_3("glass")


main_2("pima")
main_3("pima")
main("pima")
