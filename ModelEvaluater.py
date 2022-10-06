import pandas as pd
import numpy as np
import csv
import time
from sklearn.metrics import *
from pathlib import Path

def evaluate_model(x_train, y_train, x_test, y_test, model, model_name, start_time, output_path):
    # some classifiers do not need fit again, because they require parameters.
    some_classifiers = ['CSDT', 'DeepForest']
    # fit the model
    if model_name not in some_classifiers:
        model.fit(x_train, y_train)

    # make predictions
    yhat = model.predict(x_test)
    yhat = np.array(yhat).reshape(len(yhat), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)

    # evaluate metrics
    accuracy = accuracy_score(y_test, yhat)
    accuracy = accuracy * 100.0

    precision = precision_score(y_test, yhat, average='weighted')
    recall = recall_score(y_test, yhat, average='weighted')
    f1 = f1_score(y_test, yhat, average='weighted')

    kappa = cohen_kappa_score(y_test, yhat)
    kappa = kappa * 100.0

    auc = roc_auc_score(y_test, yhat, average='weighted')
    mcc = matthews_corrcoef(y_test, yhat)

    # print result
    Path(output_path + "%s/" % model_name).mkdir(parents=True, exist_ok=True)
    file_name = output_path + '%s/metrics_result.csv' % model_name
    myfile = Path(file_name)
    if myfile.exists():
        True
    else:
        csv_file = open(file_name, 'w', newline='', encoding='utf-8')
        csv_write = csv.writer(csv_file)
        csv_write.writerow(['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUC', 'MCC', 'run_time'])
        csv_file.close()

    csv_file = open(file_name, 'a+', newline='', encoding='utf-8')
    csv_write = csv.writer(csv_file)
    end_time = time.time()
    run_time = end_time - start_time
    csv_write.writerow([model_name, str(accuracy), str(precision), str(recall), str(f1), str(kappa), str(auc),
                        str(mcc), str(run_time)])
    csv_file.close()

    return accuracy, run_time
