import os
import csv
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from cliffs_delta import cliffs_delta

def read_data(dataset, dataset_path, metric):
    resample_balancers = ['SMOTE', 'ROS', 'ADASYN', 'BSMOTE', 'RUS', 'NM', 'CNN', 'TL', 'ENN']

    datas = {}
    for balancer in os.listdir(dataset_path):
        if '_result.csv' in balancer:
            continue

        data_path = ''
        if balancer in resample_balancers:
            if dataset == 'DataClass':
                data_path = dataset_path + '%s/NB/metrics_result.csv' % balancer
            if dataset == 'FeatureEnvy':
                data_path = dataset_path + '%s/LogisticRegression/metrics_result.csv' % balancer
            if dataset == 'GodClass':
                data_path = dataset_path + '%s/NB/metrics_result.csv' % balancer
            if dataset == 'LongMethod':
                data_path = dataset_path + '%s/LogisticRegression/metrics_result.csv' % balancer
        else:
            continue

        raw_datas = pd.read_csv(data_path)
        raw_datas = raw_datas[metric].values

        datas[balancer] = raw_datas

    return datas

def process_data(datas):
    metric_datas = []
    balancers = []

    baseline_data = datas['SMOTE']
    metric_datas.append(baseline_data)

    for key, value in datas.items():
        if key == 'SMOTE':
            continue
        metric_datas.append(value)
        balancers.append(key)

    return metric_datas, balancers

def wilcoxon(l1, l2):
    w, p_value = stats.wilcoxon(l1, l2, correction=False)
    return p_value

def Wilcoxon_signed_rank_test(metric_datas, balancers, dataset, metric):
    pvalues = []
    sortpvalues = []
    bhpvalues = []
    cliffs = []
    medians = []
    labels = []

    for i in range(1, len(metric_datas)):
        print(balancers[i-1])
        pvalue = wilcoxon(metric_datas[0], metric_datas[i])
        pvalues.append(pvalue)
        sortpvalues.append(pvalue)

        cliff, _ = cliffs_delta(metric_datas[i], metric_datas[0])
        cliffs.append(abs(cliff))

        median = np.median(metric_datas[i]) - np.median(metric_datas[0])
        medians.append(median)

    sortpvalues.sort()

    for i in range(len(pvalues)):
        bhpvalue = pvalues[i]*(len(pvalues))/(sortpvalues.index(pvalues[i])+1)
        bhpvalues.append(bhpvalue)

    for i in range(len(balancers)):
        if bhpvalues[i] < 0.05 and cliffs[i] > 0.147:
            if medians[i] > 0:
                labels.append('significant improvement')
            if medians[i] < 0:
                labels.append('significant decrease')
            if medians[i] == 0:
                labels.append('non-significant')
        else:
            labels.append('non-significant')

    output_path = 'results/resampling_significant/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    file_path = output_path + '{}_{}.csv'.format(dataset, metric)

    csv_file = open(file_path, 'w', newline='', encoding='utf-8')
    csv_write = csv.writer(csv_file)
    csv_write.writerow(['balancer', 'pvalue', 'cliff', 'significant'])

    for i in range(len(balancers)):
        csv_write.writerow([balancers[i], bhpvalues[i], cliffs[i], labels[i]])

    csv_file.close()


if __name__ == '__main__':
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC']

    result_path = 'results/OurDataset/'
    for dataset in os.listdir(result_path):
        if dataset in ['LongParameterList', 'SwitchStatements']:
            continue
        dataset_path = result_path + '%s/' % dataset
        for metric in metrics:
            print("Doing Wilcoxon signed rank test in %s_%s ..." % (dataset, metric))
            datas = read_data(dataset, dataset_path, metric)
            metric_datas, balancers = process_data(datas)
            Wilcoxon_signed_rank_test(metric_datas, balancers, dataset, metric)
