import os
import csv
import pandas as pd
import time
from pathlib import Path
from Processing import load_data
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import ModelEvaluater as model_eva

# basic classifier
import basic_classifiers.DecisionTree as dt
import basic_classifiers.KNN as knn
import basic_classifiers.LogisticRegression as lgr
import basic_classifiers.MLP as mlp
import basic_classifiers.NaiveBayes as nb
import basic_classifiers.RandomForest as rf
import basic_classifiers.SVM as svm

def Training_crossValidation(dataset, dataset_path, output_path):
    for i in range(10):
        # load dataset
        datas = load_data(dataset_path).values

        # 10-fold cross validation
        kfold = KFold(10, True, 1)
        for train, test in kfold.split(datas):
            print('train: %s, test: %s' % (len(datas[train]), len(datas[test])))
            # training data of one fold
            x_train = datas[train]
            y_train = x_train[:, -1]
            x_train = x_train[:, :-1]
            # testing data of one fold
            x_test = datas[test]
            y_test = x_test[:, -1]
            x_test = x_test[:, :-1]

            # normalizing training data and testing data
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            x, y = x_train, y_train

            # Decision Tree
            start_time = time.time()
            dtModel = dt.DT_training_tuning(x, y)
            dtAcc, run_time = model_eva.evaluate_model(x, y, x_test, y_test, dtModel, 'DecisionTree', start_time,
                                                       output_path)
            print('Decision Tree  accuracy  %.3f  run_time %.3fs ' % (dtAcc, run_time))

            # KNN
            start_time = time.time()
            knnModel = knn.KNN_training_tuning(x, y)
            knnAcc, run_time = model_eva.evaluate_model(x, y, x_test, y_test, knnModel, 'KNN', start_time, output_path)
            print('KNN  accuracy  %.3f  run_time %.3fs ' % (knnAcc, run_time))

            # LogisticRegression
            start_time = time.time()
            lgrModel = lgr.LogisticRegression_training_tuning(x, y)
            lgrAcc, run_time = model_eva.evaluate_model(x, y, x_test, y_test, lgrModel, 'LogisticRegression',
                                                        start_time, output_path)
            print('LogisticRegression  accuracy  %.3f  run_time %.3fs ' % (lgrAcc, run_time))

            # MLP
            start_time = time.time()
            mlpModel = mlp.MLP_training_tuning(x, y)
            mlpAcc, run_time = model_eva.evaluate_model(x, y, x_test, y_test, mlpModel, 'MLP', start_time, output_path)
            print('MLP  accuracy  %.3f  run_time %.3fs ' % (mlpAcc, run_time))

            # Naive Bayes
            start_time = time.time()
            nbModel = nb.NB_training_tuning(x, y)
            nbAcc, run_time = model_eva.evaluate_model(x, y, x_test, y_test, nbModel, 'NB', start_time, output_path)
            print('NB  accuracy  %.3f  run_time %.3fs ' % (nbAcc, run_time))

            # Random Forest
            start_time = time.time()
            rfModel = rf.RandomForest_training_tuning(x, y)
            rfAcc, run_time = model_eva.evaluate_model(x, y, x_test, y_test, rfModel, 'RandomForest', start_time,
                                                       output_path)
            print('RandomForest  accuracy  %.3f  run_time %.3fs ' % (rfAcc, run_time))

            # SVM
            start_time = time.time()
            svmModel = svm.Svm_training_tuning(x, y)
            svmAcc, run_time = model_eva.evaluate_model(x, y, x_test, y_test, svmModel, 'SVM', start_time, output_path)
            print('SVM  accuracy  %.3f  run_time %.3fs ' % (svmAcc, run_time))


def PrintClassifierResult(result_path, dataset):
    results = []
    for classifier in os.listdir(result_path):
        if classifier == "%s_result.csv" % dataset:
            continue
        tmpList = [classifier]
        datas = pd.read_csv(result_path + "%s/metrics_result.csv" % classifier)
        datas_median = datas.iloc[:, 1:].median().tolist()
        tmpList.extend(datas_median)
        results.append(tmpList)

    filename = result_path + "%s_result.csv" % dataset
    csv_file = open(filename, 'w', newline='', encoding='utf-8')
    csv_write = csv.writer(csv_file)
    csv_write.writerow(
        ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC', 'run_time'])
    for result in results:
        csv_write.writerow(result)
    csv_file.close()


if __name__ == "__main__":
    First = {'DataClass': "datasets/firstDataset/DataClass.csv",
             'FeatureEnvy': "datasets/firstDataset/FeatureEnvy.csv",
             'GodClass': "datasets/firstDataset/GodClass.csv",
             'LongMethod': "datasets/firstDataset/LongMethod.csv",
             'LongParameterList': "datasets/firstDataset/LongParameterList.csv",
             'SwitchStatements': "datasets/firstDataset/SwitchStatements.csv"
             }

    Merge = {'DataClass': "datasets/MergeDataset/DataClass.csv",
             'FeatureEnvy': "datasets/MergeDataset/FeatureEnvy.csv",
             'GodClass': "datasets/MergeDataset/GodClass.csv",
             'LongMethod': "datasets/MergeDataset/LongMethod.csv",
             'LongParameterList': "datasets/MergeDataset/LongParameterList.csv",
             'SwitchStatements': "datasets/MergeDataset/SwitchStatements.csv"
             }

    OurDataset = {'DataClass': "datasets/OurDataset/DataClass.csv",
                  'FeatureEnvy': "datasets/OurDataset/FeatureEnvy.csv",
                  'GodClass': "datasets/OurDataset/GodClass.csv",
                  'LongMethod': "datasets/OurDataset/LongMethod.csv",
                  'LongParameterList': "datasets/OurDataset/LongParameterList.csv",
                  'SwitchStatements': "datasets/OurDataset/SwitchStatements.csv"
                  }

    inputDatasets = {'First': First, 'Merge': Merge, 'OurDataset': OurDataset}
    for datasets_name, datasets in inputDatasets.items():
        for dataset, dataset_path in datasets.items():
            output_path = 'results/NotBalance/{}/{}/'.format(datasets_name, dataset)
            Path(output_path).mkdir(parents=True, exist_ok=True)

            Training_crossValidation(dataset, dataset_path, output_path)

            PrintClassifierResult(output_path, dataset)
