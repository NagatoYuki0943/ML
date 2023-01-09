'''

sklearn.feature_extraction.text.CountVectorizer(stop_words=[])

    返回词频矩阵
    stop_words 结束词,填写之后这个词不会被计数
    不返回单个字母的词和标点符号
    CountVectorizer.fit_transform(X)
        X:文本或者包含文本字符串的可迭代对象
        返回值:返回sparse矩阵
    CountVectorizer.get_feature_names() 返回值:单词列表

'''
from sklearn.feature_extraction.text import CountVectorizer


def text_count_demo():
    """
    对文本进行特征抽取，countvetorizer
    :return: None
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]

    # 1、实例化一个转换器类
    transfer = CountVectorizer()    # 注意,没有sparse这个参数

    # 2、调用fit_transform
    data = transfer.fit_transform(data)

    # 3.打印数据
    print('文本特征抽取的结果:\n', data)
    #    (0, 2)	1
    #   (0, 1)	1
    #   (0, 6)	1
    #   (0, 3)	2
    #   (0, 5)	1
    #   (1, 2)	1
    #   (1, 1)	1
    #   (1, 5)	1
    #   (1, 7)	1
    #   (1, 4)	1
    #   (1, 0)	1

    print('文本特征抽取的结果(转换为数组):\n', data.toarray())
    #  [[0 1 1 2 0 1 1 0]
    #  [1 1 1 0 1 1 0 1]]

    print("返回特征名字：\n", transfer.get_feature_names())
    # ['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']

    return None


if __name__ == '__main__':
    text_count_demo()
