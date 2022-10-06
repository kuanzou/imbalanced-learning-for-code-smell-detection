import os
import numpy as np
import pandas as pd
from pathlib import Path
from DataBalance import data_balancing_SMOTE


if __name__ == '__main__':
    datasets_path = 'datasets/FirstDataset/'

    for filename in os.listdir(datasets_path):
        dataset_path = datasets_path + filename
        datas = pd.read_csv(dataset_path).values
        columns = pd.read_csv(dataset_path).columns

        x1 = datas[:, :-1]
        y1 = datas[:, -1]

        x, y = data_balancing_SMOTE(x1, y1)

        newdatas = []
        for i in range(len(x)):
            newdata = np.hstack((x[i], y[i]))
            newdatas.append(newdata)

        newdatas = np.array(newdatas)

        Path('datasets/smote/').mkdir(parents=True, exist_ok=True)
        output_path = 'datasets/smote/' + filename

        output = pd.DataFrame(newdatas, columns=columns)
        output.to_csv(output_path, encoding='utf-8', index=0)
