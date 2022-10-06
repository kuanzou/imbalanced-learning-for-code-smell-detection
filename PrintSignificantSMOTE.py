import pandas as pd
import numpy as np
import csv
from cliffs_delta import cliffs_delta
from Wilcoxon import wilcoxon
from pathlib import Path

if __name__ == '__main__':

    SMOTE_results_path = 'results/SMOTE/'

    None_results_path = 'results/NotBalance/First/'

    datasets = ['DataClass', 'GodClass', 'FeatureEnvy', 'LongMethod']

    classifiers = ['DecisionTree', 'KNN', 'LogisticRegression', 'MLP', 'NB', 'RandomForest', 'SVM']

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC']

    for dataset in datasets:
        for metric in metrics:
            pvalues = []
            cliffs = []
            labels = []
            for classifier in classifiers:
                SMOTE_classifier_path = SMOTE_results_path + '{}/{}/metrics_result.csv'.format(dataset, classifier)
                SMOTE_results = pd.read_csv(SMOTE_classifier_path)
                None_classifier_path = None_results_path + '{}/{}/metrics_result.csv'.format(dataset, classifier)
                None_results = pd.read_csv(None_classifier_path)

                SMOTE_metric_result = SMOTE_results[metric].values
                None_metric_result = None_results[metric].values

                median = np.median(SMOTE_metric_result) - np.median(None_metric_result)

                pvalue = wilcoxon(SMOTE_metric_result, None_metric_result)
                pvalues.append(pvalue)

                cliff, _ = cliffs_delta(SMOTE_metric_result, None_metric_result)
                cliffs.append(abs(cliff))

                if pvalue < 0.05 and abs(cliff) > 0.147:
                    if median > 0:
                        labels.append('significant improvement')
                    if median < 0:
                        labels.append('significant decrease')
                    if median == 0:
                        labels.append('non-significant')
                else:
                    labels.append('non-significant')

            print('{} {}'.format(dataset, metric))

            output_path = SMOTE_results_path + '{}/'.format('significant')
            Path(output_path).mkdir(parents=True, exist_ok=True)
            file_path = output_path + '{}_{}.csv'.format(dataset, metric)

            csv_file = open(file_path, 'w', newline='', encoding='utf-8')
            csv_write = csv.writer(csv_file)
            csv_write.writerow(['classifier', 'pvalue', 'cliff', 'significant'])

            for i in range(len(classifiers)):
                csv_write.writerow([classifiers[i], pvalues[i], cliffs[i], labels[i]])

            csv_file.close()


