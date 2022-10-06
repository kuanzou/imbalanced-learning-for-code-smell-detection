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

# Data Balance
import DataBalance as balance

def Training_crossValidation(dataset, dataset_path, balancers):
    balances_dataset = {'NotBalance': '',
                        'ROS': balance.data_balancing_ROS,
                        'SMOTE': balance.data_balancing_SMOTE,
                        'ADASYN': balance.data_balancing_ADASYN,
                        'BSMOTE': balance.data_balancing_BSMOTE,
                        'RUS': balance.data_balancing_RUS,
                        'NM': balance.data_balancing_NM,
                        'CNN': balance.data_balancing_CNN,
                        'TL': balance.data_balancing_TL,
                        'ENN': balance.data_balancing_ENN
                        }
    balances_classifier = {'Adacost': balance.data_balancing_Adacost,
                           'AsymBoost': balance.data_balancing_AsymBoost,
                           'AdaUBoost': balance.data_balancing_AdaUBoost,
                           'CSSVM': balance.data_balancing_CSSVM,
                           'CSDT': balance.data_balancing_CSDT,
                           'Bagging': balance.data_balancing_Bagging,
                           'AdaBoost': balance.data_balancing_AdaBoost,
                           'CatBoost': balance.data_balancing_CatBoost,
                           'XGBoost': balance.data_balancing_XGBoost,
                           'DeepForest': balance.data_balancing_DeepForest,
                           'StackingLR': balance.data_balancing_StackingLR,
                           'StackingDT': balance.data_balancing_StackingDT,
                           'StackingSVM': balance.data_balancing_StackingSVM,
                           'SMOTEBoost': balance.data_balancing_SMOTEBoost,
                           'ROSBoost': balance.data_balacing_ROSBoost,
                           'SMOTEBagging': balance.data_balacing_SMOTEBagging,
                           'ROSBagging': balance.data_balacing_ROSBagging,
                           'RUSBoost': balance.data_balancing_RUSBoost,
                           'RUSBagging': balance.data_balacing_RUSBagging,
                           'EEC': balance.data_balancing_EasyEnsemble,
                           'BCC': balance.data_balacing_BalanceCascade,
                           'BRF': balance.data_balancing_BalancedRandomForest
                           }

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

            for balancer in balancers:
                # create result folder
                output_path = "results/OurDataset/%s/%s/" % (dataset, balancer)
                Path(output_path).mkdir(parents=True, exist_ok=True)

                # Data Balance
                if balancer in balances_dataset.keys():
                    if balancer != 'NotBalance':
                        x, y = balances_dataset[balancer](x_train, y_train)
                    else:
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

                    # RandomForest
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

                elif balancer in balances_classifier.keys():
                    start_time = time.time()
                    classifier = balances_classifier[balancer](x_train, y_train)
                    Acc, run_time = model_eva.evaluate_model(x_train, y_train, x_test, y_test, classifier, balancer,
                                                             start_time, output_path)
                    print('%s  accuracy  %.3f  run_time %.3fs ' % (balancer, Acc, run_time))


if __name__ == "__main__":
    datasets = {'DataClass': "datasets/OurDataset/DataClass.csv",
                'FeatureEnvy': "datasets/OurDataset/FeatureEnvy.csv",
                'GodClass': "datasets/OurDataset/GodClass.csv",
                'LongMethod': "datasets/OurDataset/LongMethod.csv",
                'LongParameterList': "datasets/OurDataset/LongParameterList.csv",
                'SwitchStatements': "datasets/OurDataset/SwitchStatements.csv"
                }

    balancers = ['NotBalance', 'ROS', 'SMOTE', 'ADASYN', 'BSMOTE', 'RUS', 'NM', 'CNN', 'TL', 'ENN',
                 'Adacost', 'AsymBoost', 'AdaUBoost', 'CSSVM', 'CSDT',
                 'Bagging', 'AdaBoost', 'CatBoost', 'XGBoost', 'DeepForest', 'StackingLR', 'StackingDT', 'StackingSVM',
                 'SMOTEBoost', 'ROSBoost', 'SMOTEBagging', 'ROSBagging',
                 'RUSBoost', 'RUSBagging', 'EEC', 'BCC', 'BRF'
                 ]

    for dataset, dataset_path in datasets.items():
        Training_crossValidation(dataset, dataset_path, balancers)
