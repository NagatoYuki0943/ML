'''
用法和 特征值标准化 类似

sklearn.feature_extraction.DictVectorizer(sparse=True,…)
    DictVectorizer.fit_transform(X)
        X:字典或者包含字典的迭代器返回值
        sparse=True  返回sparse矩阵    可以提高效率和节省内存
        sparse=False 返回非sparse矩阵

    DictVectorizer.get_feature_names() 返回类别名称
'''

from sklearn.feature_extraction import DictVectorizer


def dict_demo():
    """
    对字典类型的数据进行特征抽取
    :return: None
    """

    data = [
        {'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}]

    # 1、实例化一个转换器类
    #  sparse=True  返回sparse矩阵    可以提高效率和节省内存
    #  sparse=False 返回非sparse矩阵
    transfer = DictVectorizer(sparse=False)
    # 2、调用fit_transform
    data = transfer.fit_transform(data)

    print("返回的结果\n", data)
    #  [[  0.   1.   0. 100.]
    #  [  1.   0.   0.  60.]
    #  [  0.   0.   1.  30.]]

    # 打印特征名字
    print("特征名字：\n", transfer.get_feature_names())
    # ['city=上海', 'city=北京', 'city=深圳', 'temperature']

    return None



if __name__ == '__main__':
    dict_demo()