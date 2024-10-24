from src.CsvReader import CsvReader
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import simplejson as json

if __name__ == '__main__':

    file = '../../preprocessedData.csv'
    # reader = CsvReader(file)
    # data_generator = reader.read_file()
    dataframe = pd.read_csv(file, index_col='No')
    dataset_size = 0

    features = {}
    exclude_features = ('No', 'key', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'protocol', 'Target')
    metrics = ['sum', 'min', 'max', 'mean', 'std']

    #pd.options.display.float_format = "{:.4f}".format
    dataframe.sort_values(['key', 'timestamp'], ascending=[True, True], inplace=True)

    dataframe.drop('key', inplace=True)
    dataframe.to_csv('../../preprocessedData_sorted.csv', index=False)
    #print(dataframe.head(20).to_string())
    '''
    for col in dataframe.columns:
        if col not in exclude_features:
            print(col)
            plt.figure()
            dataframe[col].plot(kind='kde')
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

    for feature in dataframe:
     
        if feature not in exclude_features:
            if feature not in features:
                features[feature] = {x: 0.0 for x in metrics}

            features[feature]['sum'] = dataframe[feature].sum()
            features[feature]['min'] = dataframe[feature].min()
            features[feature]['max'] = dataframe[feature].max()
            features[feature]['mean'] = dataframe[feature].mean()
            features[feature]['std'] = dataframe[feature].std()

    dataset_size = dataframe.shape[0]

    res = json.dumps(features, indent=4, sort_keys=False, default=str)

    print(dataset_size)
    print(res)
    '''