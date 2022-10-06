# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(dataset, dataset_path, metric):
    """
    loading the specific metric datas from a code smell dataset result
    :param dataset: a code smell dataset result
    :param dataset_path: result folder path
    :param metric: (accuracy, precision, recall, f1-score, kappa, auc, mcc)
    :return: metric datas
    """
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
            if dataset == 'LongMethod':
                data_path = dataset_path + '%s/LogisticRegression/metrics_result.csv' % balancer
            # if dataset == 'LongParameterList':
            #     data_path = dataset_path + '%s/RandomForest/metrics_result.csv' % balancer
            # if dataset == 'SwitchStatements':
            #     data_path = dataset_path + '%s/RandomForest/metrics_result.csv' % balancer
        else:
            data_path = dataset_path + '%s/%s/metrics_result.csv' % (balancer, balancer)

        # read all result datas
        raw_datas = pd.read_csv(data_path)
        # get the metric collun datas
        raw_datas = raw_datas[metric].values
        # save metric datas
        datas[balancer] = raw_datas

    return datas

def process_data(datas):
    resample_balancers = ['ROS', 'SMOTE', 'ADASYN', 'BSMOTE', 'RUS', 'NM', 'CNN', 'TL', 'ENN']
    cost_balancers = ['Adacost', 'AsymBoost', 'AdaUBoost', 'CSSVM', 'CSDT']
    ensemble_balancers = ['Bagging', 'AdaBoost', 'CatBoost', 'XGBoost', 'DeepForest', 'StackingLR', 'StackingDT', 'StackingSVM']
    imbalance_balancers = ['SMOTEBoost', 'ROSBoost', 'SMOTEBagging', 'ROSBagging', 'RUSBoost', 'RUSBagging', 'EEC', 'BCC', 'BRF']

    baseline_data = datas['NotBalance']

    metric_datas = []
    balancers = []

    for balancer in resample_balancers:
        metric_datas.append(datas[balancer] - baseline_data)
        balancers.append(balancer)

    for balancer in ensemble_balancers:
        metric_datas.append(datas[balancer] - baseline_data)
        balancers.append(balancer)

    for balancer in cost_balancers:
        metric_datas.append(datas[balancer] - baseline_data)
        balancers.append(balancer)

    for balancer in imbalance_balancers:
        metric_datas.append(datas[balancer] - baseline_data)
        balancers.append(balancer)

    return metric_datas, balancers

def load_color_Wilcoxon(dataset, metric, balancers, medians):
    colors_path = 'statistics/Wilcoxon/csv/%s_%s.csv' % (dataset, metric)
    datas = pd.read_csv(colors_path)

    colors = []
    for balancer in balancers:
        if datas[balancer][0] < 0.05 and datas[balancer][1] > 0.147:
            if medians[balancers.index(balancer)] > 0:
                colors.append('red')
            elif medians[balancers.index(balancer)] < 0:
                colors.append('green')
            else:
                colors.append('black')
        else:
            colors.append('black')

    return colors

# def load_color_SKESD(dataset, metric, balancers):
#     colors_path = 'statistics/SKESD/rank/%s_%s.txt' % (dataset, metric)
#     with open(colors_path, 'r', encoding='utf-8') as f:
#         datas = f.readlines()
#
#     rank_dict = {}
#     for i in range(1, len(datas)):
#         rank = datas[i].strip('\n').split(' ')
#         rank[0] = rank[0].strip('"')
#         rank_dict[rank[0]] = int(rank[1])
#
#     color_dict = {1: 'red', 2: "green", 3: 'blue', 4: 'yellow', 5: 'purple', 6: 'orange', 7: 'pink', 8: 'lightcoral',
#                   9: 'olive', 10: 'plum', 11: 'c', 12: 'aqua', 13: 'cyan', 14: 'teal', 15: 'skyblue', 16: 'darkblue',
#                   17: 'deepskyblue', 18: 'indigo', 19: 'darkorange', 20: 'gray'}
#
#     colors = []
#     for balancer in balancers:
#         colors.append(color_dict[rank_dict[balancer]])
#
#     return colors

def draw_box(metric_datas, balancers, metric, dataset, test):
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(6, 2))  # figsize:指定figure的宽和高，单位为英寸；
    ax.tick_params(direction='in')

    xticks = np.arange(1, len(balancers) * 2, 2)
    figure = ax.boxplot(metric_datas,
                        notch=False,  # notch shape
                        sym='r+',  # blue squares for outliers
                        vert=True,  # vertical box aligmnent
                        meanline=True,
                        showmeans=False,
                        patch_artist=False,
                        showfliers=False,
                        positions=xticks,
                        boxprops={'color': 'red'}
                        )

    medians = np.median(metric_datas, axis=1)

    colors = []
    if test == 'Wilcoxon':
        colors = load_color_Wilcoxon(dataset, metric, balancers, medians)
    # elif test == 'SKESD':
    #     colors = load_color_SKESD(dataset, metric, balancers)

    median_fold = 'statistics/%s/median/' % test
    Path(median_fold).mkdir(parents=True, exist_ok=True)
    median_path = median_fold + '%s_%s.csv' % (dataset, metric)
    csv_file = open(median_path, 'w', newline='', encoding='utf-8')
    csv_write = csv.writer(csv_file)

    csv_write.writerow(['balancer', 'median', 'difference'])
    for i in range(len(medians)):
        if medians[i] > 0 and colors[i] == 'red':
            csv_write.writerow([balancers[i], medians[i], 'signficantly improved'])
        if medians[i] > 0 and colors[i] == 'black':
            csv_write.writerow([balancers[i], medians[i], 'non-signficantly improved'])
        if medians[i] == 0:
            csv_write.writerow([balancers[i], medians[i], 'equailvant'])
        if medians[i] < 0 and colors[i] == 'green':
            csv_write.writerow([balancers[i], medians[i], 'signficantly reduced'])
        if medians[i] < 0 and colors[i] == 'black':
            csv_write.writerow([balancers[i], medians[i], 'non-signficantly reduced'])

    csv_file.close()

    for i in range(len(balancers)):
        if balancers[i] == 'AdaBoost':
            balancers[i] = 'ADB'
        if balancers[i] == 'CatBoost':
            balancers[i] = 'CBoost'
        if balancers[i] == 'XGBoost':
            balancers[i] = 'XGB'
        if balancers[i] == 'DeepForest':
            balancers[i] = 'DF'
        if balancers[i] == 'StackingLR':
            balancers[i] = 'SLR'
        if balancers[i] == 'StackingDT':
            balancers[i] = 'SDT'
        if balancers[i] == 'StackingSVM':
            balancers[i] = 'SSVM'
        if balancers[i] == 'SMOTEBagging':
            balancers[i] = 'SMOTEBAG'
        if balancers[i] == 'ROSBagging':
            balancers[i] = 'ROSBAG'
        if balancers[i] == 'RUSBagging':
            balancers[i] = 'RUSBAG'

    for i in range(len(colors)):
        k = figure['boxes'][i]
        k.set(color=colors[i])

        # k = figure['means'][i]
        # k.set(color='green', linewidth=1)

        k = figure['medians'][i]
        k.set(color=colors[i], linewidth=2)

        k = figure['whiskers'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i], linestyle='--')

        k = figure['caps'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i])

    # plt.xlim((0, 20)) #在Python的matplotlib.pyplot中方法xlim和ylim的使用方法相同，分别用来设置x轴和y轴的显示范围。SANER2020设置为这样
    plt.xlim((0, 63))
    # plt.ylim(ymin*1.1, ymax*1.1)
    # 设置x轴坐标,这里的y+1是为了让第一个算法的名称在(1,0)这个位置
    # yticks = np.append(np.linspace(ymin*1.1, ymax*1.1, 5)[1:-1], [0])
    # plt.xticks([y+1 for y in range(len(balancers))], balancers, rotation=45, weight='heavy', fontsize=8, ha='center')
    plt.xticks(xticks, balancers, rotation=60, weight='heavy', fontsize=7, ha='center')
    # plt.yticks([round(y, 2) for y in yticks], fontsize=8)
    plt.yticks(fontsize=8)

    if metric == 'MatthewsCoff':
        metric = 'MCC'
    if metric == 'Roc measure':
        metric = 'AUC'
    plt.ylabel(metric, fontsize=10, weight='heavy')

    # if dataset == 'DataClass':
    #     plt.xlabel('(a) Data Class', fontsize=10, weight='heavy')
    # elif dataset == 'FeatureEnvy':
    #     plt.xlabel('(c) Feature Envy', fontsize=10, weight='heavy')
    # elif dataset == 'GodClass':
    #     plt.xlabel('(b) God Class', fontsize=10, weight='heavy')
    # elif dataset == 'LongMethod':
    #     plt.xlabel('(d) Long Method', fontsize=10, weight='heavy')
    # elif dataset == 'LongParameterList':
    #     plt.xlabel('(e) Long Parameter List', fontsize=10, weight='heavy')
    # elif dataset == 'SwitchStatements':
    #     plt.xlabel('(f) Switch Statements', fontsize=10, weight='heavy')

    plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    plt.axhline(y=0, color='blue', linewidth=0.5)
    plt.axvline(18, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    plt.axvline(34, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    plt.axvline(44, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    # plt.axvline(36.5, color='black', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标

    plt.title(
        "        Data Resampling                    "
        "Ensemble Learning     "
        "Cost-Sensitive Learning  "
        "Imbalanced Ensemble Learning"
        , fontsize=7, loc='left', weight='heavy')

    # plt.show()

    figure_fold = 'statistics/%s/figures/' % test
    Path(figure_fold).mkdir(parents=True, exist_ok=True)
    output_path = figure_fold + '%s_%s.pdf' % (dataset, metric)

    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='pdf', dpi=1000, bbox_inches='tight')

    plt.clf()
    plt.close()

def BoxPlot(results_path, test):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC']

    print("Draw %s box plot" % test)

    for dataset in os.listdir(results_path):
        if dataset in ['LongParameterList', 'SwitchStatements']:
            continue

        for metric in metrics:
            print('Drawing %s_%s .....' % (dataset, metric))

            dataset_path = results_path + '%s/' % dataset

            datas = load_data(dataset, dataset_path, metric)

            metric_datas, balancers = process_data(datas)

            draw_box(metric_datas, balancers, metric, dataset, test)


if __name__ == '__main__':
    BoxPlot(r'results/OurDataset/', 'Wilcoxon')

