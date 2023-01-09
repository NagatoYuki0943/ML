import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def var_thr():
    '''
    特征选择,低方差特征过滤
    '''
    data = pd.read_csv('../data/factor_returns.csv')
    print(data)
    print('*' * 50)

    print(data.shape)
    print('*' * 50)

    # 特征降维
    transfer = VarianceThreshold(threshold=1)
    transfer_data = transfer.fit_transform(data.iloc[:, 1:10])
    print(transfer_data)
    print('*' * 50)

    # 通过形状查看数据是否相同
    print(data.iloc[:, 1:10].shape)     # (2318, 9)
    print(transfer_data.shape)          # (2318, 8)
    print('*' * 50)



if __name__ == '__main__':
    var_thr()