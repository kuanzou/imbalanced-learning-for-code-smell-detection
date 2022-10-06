import os
import csv
import pandas as pd
from pathlib import Path

def PrintClassifierResult(result_path, balancer):
    results = []
    for classifier in os.listdir(result_path):
        if classifier == "%s_result.csv" % balancer:
            continue
        tmpList = [balancer, classifier]
        datas = pd.read_csv(result_path + "%s/metrics_result.csv" % classifier)
        datas_mean = datas.iloc[:, 1:].mean().tolist()
        tmpList.extend(datas_mean)
        results.append(tmpList)

    filename = result_path + "%s_result.csv" % balancer
    csv_file = open(filename, 'w', newline='', encoding='utf-8')
    csv_write = csv.writer(csv_file)
    csv_write.writerow(
        ['Balance', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC', 'run_time'])
    for result in results:
        csv_write.writerow(result)
    csv_file.close()

def PrintBalancerResult(result_path, dataset):
    results = []
    for balancer in os.listdir(result_path):
        if balancer == '%s_result.csv' % dataset:
            continue
        datas = pd.read_csv(result_path + "%s/%s_result.csv" % (balancer, balancer))
        datas = datas.values.tolist()
        results.append(datas)

    filename = result_path + "%s_result.csv" % dataset
    csv_file = open(filename, 'w', newline='', encoding='utf-8')
    csv_write = csv.writer(csv_file)
    csv_write.writerow(
        ['Balancer', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC', 'run_time'])
    for balancer_result in results:
        for classifier_result in balancer_result:
            csv_write.writerow(classifier_result)
    csv_file.close()

    print("%s Ending!" % dataset)


if __name__ == '__main__':
    result_path = "results/OurDataset/"
    for dataset in os.listdir(result_path):
        result_dataset_path = result_path + "%s/" % dataset
        for balancer in os.listdir(result_dataset_path):
            if balancer == '%s_result.csv' % dataset:
                continue
            result_balancer_path = result_dataset_path + "%s/" % balancer
            PrintClassifierResult(result_balancer_path, balancer)
        PrintBalancerResult(result_dataset_path, dataset)
