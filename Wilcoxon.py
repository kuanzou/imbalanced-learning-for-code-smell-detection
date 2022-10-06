# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from DrawBoxPlot import BoxPlot
from cliffs_delta import cliffs_delta

def read_data(dataset, dataset_path, metric):
    resample_balancers = ['NotBalance', 'ROS', 'SMOTE', 'ADASYN', 'BSMOTE', 'RUS', 'NM', 'CNN', 'TL', 'ENN']

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
            # if dataset == 'LongMethod':
            #     data_path = dataset_path + '%s/NB/metrics_result.csv' % balancer
            if dataset == 'LongMethod':
                data_path = dataset_path + '%s/LogisticRegression/metrics_result.csv' % balancer
            if dataset == 'LongParameterList':
                data_path = dataset_path + '%s/RandomForest/metrics_result.csv' % balancer
            if dataset == 'SwitchStatements':
                data_path = dataset_path + '%s/RandomForest/metrics_result.csv' % balancer
        else:
            data_path = dataset_path + '%s/%s/metrics_result.csv' % (balancer, balancer)

        raw_datas = pd.read_csv(data_path)
        raw_datas = raw_datas[metric].values

        datas[balancer] = raw_datas

    return datas

def process_data(datas):
    metric_datas = []
    balancers = []

    baseline_data = datas['NotBalance']
    metric_datas.append(baseline_data)

    for key, value in datas.items():
        if key == 'NotBalance':
            continue
        metric_datas.append(value)
        balancers.append(key)

    return metric_datas, balancers

def wilcoxon(l1, l2):
    w, p_value = stats.wilcoxon(l1, l2, correction=False)
    return p_value

# win draw loss值
def wdl(l1, l2):
    win = 0
    draw = 0
    loss = 0
    for i in range(len(l1)):
        if l1[i] < l2[i]:
            loss = loss+1
            # print('从0列开始，第', i, '列，我们的方法输了')
        if l1[i] == l2[i]:
            draw = draw+1
        if l1[i] > l2[i]:
            win = win+1

    return win, draw, loss

def average_improvement(l1, l2):
    avgl1 = round(np.average(l1), 3)
    avgl2 = round(np.average(l2), 3)
    imp = round((avgl1-avgl2)/avgl2, 4)

    return avgl1, avgl2, imp

# def cliff(l1, l2):
#     total = 0
#     for i in l1:
#         temp = 0
#         for j in l2:
#             if i < j:
#                 temp -= 1
#             elif i > j:
#                 temp += 1
#         total += temp
#     return total / (len(l1) * len(l2))

def Wilcoxon_signed_rank_test(metric_datas, balancers, metric):
    pvalues = []
    sortpvalues = []
    bhpvalues = []
    cliffs = []

    for i in range(1, len(metric_datas)):
        print(balancers[i-1])
        pvalue = wilcoxon(metric_datas[0], metric_datas[i])
        pvalues.append(pvalue)
        sortpvalues.append(pvalue)

        cliff, _ = cliffs_delta(metric_datas[i], metric_datas[0])
        cliffs.append(abs(cliff))

        # print("compute p-value between %s and NotBalance: %s" % (balancers[i-1], pvalue))
        # print("compute W/D/L between %s and NotBalance: %s" % (balancers[i-1], wdl(metric_datas[0], metric_datas[i])))
        # print("compute average improvement between %s and NotBalance: %s" % (balancers[i-1],
        #                                                                      average_improvement(metric_datas[0], metric_datas[i])))

    sortpvalues.sort()

    for i in range(len(pvalues)):
        bhpvalue = pvalues[i]*(len(pvalues))/(sortpvalues.index(pvalues[i])+1)
        bhpvalues.append(bhpvalue)
        # print("compute Benjamini—Hochberg p-value between %s and NotBalance: %s" % (balancers[i-1], bhpvalue))

    Path('statistics/Wilcoxon/csv/').mkdir(parents=True, exist_ok=True)
    output_path = 'statistics/Wilcoxon/csv/%s_%s.csv' % (dataset, metric)

    output = pd.DataFrame(data=[bhpvalues, cliffs], columns=balancers)
    output.to_csv(output_path, encoding='utf-8')


if __name__ == '__main__':
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC']

    result_path = 'results/OurDataset/'
    for dataset in os.listdir(result_path):
        dataset_path = result_path + '%s/' % dataset
        for metric in metrics:
            print("Doing Wilcoxon signed rank test in %s_%s ..." % (dataset, metric))
            datas = read_data(dataset, dataset_path, metric)
            metric_datas, balancers = process_data(datas)
            Wilcoxon_signed_rank_test(metric_datas, balancers, metric)

    # If the BH corrected p-value is less than 0.05,
    # it means that there is a statistically significant difference between the two methods.

    # BoxPlot(result_path, 'Wilcoxon')
