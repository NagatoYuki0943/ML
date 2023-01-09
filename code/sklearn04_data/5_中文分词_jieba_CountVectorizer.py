'''
不支持单个中文,要用空格分开

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
import jieba


def cut_word(text):
    """
    对中文进行分词
    "我爱北京天安门"————>"我 爱 北京 天安门"
    :param text:
    :return: text
    """

    # jieba返回对象,使用list转换    使用空格分开字词
    text = " ".join(list(jieba.cut(text)))

    return text


def text_chinese_count_demo():
    """
    对中文进行特征抽取
    :return: None
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    # 将原始数据转换成分好词的形式
    text_list = []
    for sent in data:
        # 返回来的加了空格的字符串,再放进列表中
        text_list.append(cut_word(sent))

    print(text_list)

    # 1、实例化一个转换器类
    transfer = CountVectorizer()
    # 2、调用fit_transform
    data = transfer.fit_transform(text_list)

    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names())

    return None


if __name__ == '__main__':
    print(cut_word('我爱你python,人生苦短,我用python'))
    # 分成功了,用空格分开了
    # 我爱你 python , 人生 苦短 , 我用 python

    text_chinese_count_demo()

