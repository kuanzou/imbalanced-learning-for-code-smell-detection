import csv
import pandas as pd
import numpy as np
from info_gain import info_gain
from Processing import load_data, find_Null_Samples
from pathlib import Path

def Merge_and_Delete(datasets, input_path, output_path):
    for dataset in datasets:
        # origin dataset path
        dataset_path_in = input_path + dataset
        # preprocessing dataset path
        dataset_path_out = output_path + dataset
        # create output path if this path not exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # load dataset
        columns = load_data(dataset_path_in).columns
        inputData_A = load_data(dataset_path_in).values
        # load the corresponding dataset
        inputData_B = ''
        if dataset == 'DataClass.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "GodClass.csv").values
        elif dataset == 'GodClass.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "DataClass.csv").values
        elif dataset == 'FeatureEnvy.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "LongMethod.csv").values
        elif dataset == 'LongMethod.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "FeatureEnvy.csv").values
        elif dataset == 'LongParameterList.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "SwitchStatements.csv").values
        elif dataset == 'SwitchStatements.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "LongParameterList.csv").values

        # remove self same datas
        inputData_A = np.array(list(set([tuple(t) for t in inputData_A])))
        inputData_B = np.array(list(set([tuple(t) for t in inputData_B])))

        # find the same data between inputData_A and inputData_B
        same_index = []
        for dataA in inputData_A:
            for i in range(len(inputData_B)):
                if (inputData_B[i] == dataA).all():
                    same_index.append(i)
                    break

        # remove redundant datas and conflict_data in inputData_B
        inputData_B = np.array([inputData_B[i, :] for i in range(len(inputData_B)) if i not in same_index])
        if len(same_index) != 0:
            print('%s have %s redundant datas and conflict datas.' % (dataset.split('.')[0], len(same_index)))

        # split smell and non-smell from inputdata_A
        smellDatas = np.array([inputData_A[i, :] for i in range(len(inputData_A)) if inputData_A[i, -1] == 1.0])
        nonsmellDatas = np.array([inputData_A[i, :] for i in range(len(inputData_A)) if inputData_A[i, -1] == 0.0])

        # merge non-smell and inputData_B as a whole non-smell
        nonsmellDatas = np.vstack((nonsmellDatas, inputData_B))

        # replace 0 at label colunm in non-smell
        nonsmellDatas[:, -1] = [0 for i in range(len(nonsmellDatas))]

        # merge smell and non-smell
        datas = np.vstack((smellDatas, nonsmellDatas))

        # shuffle datas
        np.random.shuffle(datas)

        # fill missing values
        datas = find_Null_Samples(datas)

        df = pd.DataFrame(data=datas, columns=columns)
        FeatureSelect(df, dataset_path_out)

    print('Merge and Delete Ending!')


def Merge(datasets, input_path, output_path):
    for dataset in datasets:
        # origin dataset path
        dataset_path_in = input_path + dataset
        # preprocessing dataset path
        dataset_path_out = output_path + dataset
        # create output path if this path not exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # load dataset
        columns = load_data(dataset_path_in).columns
        inputData_A = load_data(dataset_path_in).values
        # load the corresponding dataset
        inputData_B = ''
        if dataset == 'DataClass.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "GodClass.csv").values
        elif dataset == 'GodClass.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "DataClass.csv").values
        elif dataset == 'FeatureEnvy.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "LongMethod.csv").values
        elif dataset == 'LongMethod.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "FeatureEnvy.csv").values
        elif dataset == 'LongParameterList.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "SwitchStatements.csv").values
        elif dataset == 'SwitchStatements.csv':
            inputData_B = load_data(dataset_path_in.split(dataset)[0] + "LongParameterList.csv").values

        # merge A and B
        datas = np.vstack((inputData_A, inputData_B))

        # shuffle datas
        np.random.shuffle(datas)

        # fill missing values
        datas = find_Null_Samples(datas)

        df = pd.DataFrame(data=datas, columns=columns)
        FeatureSelect(df, dataset_path_out)

    print('Merge Ending!')


def Preprocessing(datasets, input_path, output_path):
    for dataset in datasets:
        # origin dataset path
        dataset_path_in = input_path + dataset
        # preprocessing dataset path
        dataset_path_out = output_path + dataset
        # create output path if this path not exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # load dataset
        columns = load_data(dataset_path_in).columns
        datas = load_data(dataset_path_in).values

        # fill missing values
        datas = find_Null_Samples(datas)

        df = pd.DataFrame(data=datas, columns=columns)
        FeatureSelect(df, dataset_path_out)

    print('First Ending!')

def FeatureSelect(df, output_file_path):
    columns = df.columns

    ratios = []
    col = df.shape[1]
    label_data = list(df[columns[col - 1]])
    for i in range(col - 1):
        col_data = list(df[columns[i]])

        gain_ratio = info_gain.info_gain_ratio(col_data, label_data)

        ratios.append([columns[i], gain_ratio])

    ratios.sort(key=lambda ele: ele[1], reverse=True)

    for ratio in ratios:
        if ratio[1] < 0.1:
            df.drop(columns=ratio[0], axis=1, inplace=True)

    df.to_csv(output_file_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    input_path = 'datasets/origin/'
    output_path = ['datasets/FirstDataset/', 'datasets/MergeDataset/', 'datasets/OurDataset/']
    datasets = ['DataClass.csv', 'FeatureEnvy.csv', 'GodClass.csv', 'LongMethod.csv', 'LongParameterList.csv',
                'SwitchStatements.csv']

    Preprocessing(datasets, input_path, output_path[0])

    Merge(datasets, input_path, output_path[1])

    Merge_and_Delete(datasets, input_path, output_path[2])
