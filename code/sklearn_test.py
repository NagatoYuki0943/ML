# https://zhuanlan.zhihu.com/p/641201586

from sklearn.datasets import load_iris
from sklearn.datasets import (
    make_blobs,         # 生成由聚类组成的非线性数据集
    make_classification,# 准备类别不平衡数据
    make_moons,         # 生成形状类似于月牙的数据集的函数
    make_regression,    # 生成回归数据
    make_s_curve,       # 生成形状类似于曲线的数据集
)

from sklearn.feature_extraction import DictVectorizer       # 字典特征抽取
from sklearn.feature_extraction.text import CountVectorizer # 文本特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer # 文本特征抽取

from sklearn.preprocessing import (
    OneHotEncoder,                                          # one-hot
    MinMaxScaler,                                           # MinMax
    StandardScaler,                                         # 特征数据标准化
    normalize,                                              # 是按照向量空间模型（Vector Space Model）对特征向量进行转换，
                                                            # 使得每个特征向量的欧几里得长度（L2范数）等于1，或者每个元素的
                                                            # 绝对值之和（L1范数）等于1。换句话说：和标准化不同，
                                                            # Scikit-learn中的归一化特指将单个样本（一行数据）放缩为单位范数
                                                            # （1范数或者2范数为单位范数）的过程，该操作常见于核方法或者衡量样
                                                            # 本之间相似性的过程中。
    LabelEncoder,                                           # 标签转换为数字
    PolynomialFeatures,                                     # 多项式回归
    Binarizer,                                              # 二值化
    KBinsDiscretizer,                                       # 分箱
)

from sklearn.impute import (
    SimpleImputer,                                          # 缺失值填充,支持均值,中位数,众数,常量
    IterativeImputer,                                       # 差值填充
    KNNImputer,                                             # K近邻填充
    MissingIndicator,                                       # 缺失值标记
)

# 类别不平衡
from imblearn.under_sampling import RandomUnderSampler      # 随机下采样
from imblearn.over_sampling import RandomOverSampler        # 随机过采样

# 特征降维
from sklearn.feature_selection import VarianceThreshold     # 特征选择,低方差特征过滤
from sklearn.decomposition import PCA                       # PCA

from sklearn.model_selection import (
    train_test_split,                                       # 分割数据
    StratifiedShuffleSplit,                                 # 分割数据
    ShuffleSplit,                                           # 分割数据
    TimeSeriesSplit,                                        # 时间序列分割
    # https://zhuanlan.zhihu.com/p/642060346
    # https://zhuanlan.zhihu.com/p/642920484
    GridSearchCV,                                           # 网格搜索+交叉验证
    RandomizedSearchCV,                                     # 随机搜索+交叉验证
    HalvingGridSearchCV,                                    # 减半网格搜索+交叉验证 使用一小部分数据对所有的参数组合进行快速评估，然后仅保留表现最好的一部分参数组合，对其使用更多的数据进行进一步的评估
    KFold,
    cross_validate,     # cross_val_score与cross_validate的区别就是：cross_validate能够输出训练集分数
    cross_val_score,
    cross_val_predict,
)

# 贝叶斯
from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB,
    MultinomialNB,
    ComplementNB,
    CategoricalNB,
)

# 线性模型
from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor,
    SGDClassifier,
    LogisticRegression,     # 逻辑回归
    LogisticRegressionCV,
    Ridge,                  # 岭回归
    RidgeCV,
    Lasso,                  # Lasso回归
    LassoCV,
)

# KNN
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, kneighbors_graph

# 树
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import export_graphviz

# 随机森林
# https://zhuanlan.zhihu.com/p/648898531
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,   # 极端随机森林
    ExtraTreesRegressor,    # 1.对于每个决策树的训练集，RF采用的是随机采样bootstrap来选择采样集作为每个决策树的训练集。
                            # 而extra trees一般不采用随机采样，即每个决策树采用原始训练集。
                            # 2.在选定了划分特征后，RF的决策树会基于基尼系数，均方差之类的原则，选择一个最优的特征值划分点，这和传统的决策树相同。
                            # 但是extra trees比较的激进，他会随机的选择一个特征值来划分决策树。
    BaggingClassifier,
    BaggingRegressor,
)

# 聚类
from sklearn.datasets import make_blobs # 为聚类产生数据集，产生一个数据集和相应的标签
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    BisectingKMeans,            # 二分kmeans聚类算法
    DBSCAN,                     # 基于密度的聚类算法
    OPTICS,                     # 基于密度的聚类算法
    AffinityPropagation,        # 亲和力传播
    AgglomerativeClustering,    # 层次聚类
    Birch,
    FeatureAgglomeration,
    MeanShift,                  # 均值漂移
    SpectralClustering,         # 谱聚类
    SpectralBiclustering,       # 光谱双聚算法
    SpectralCoclustering,
)
# 评价结果
from sklearn.metrics.cluster import (
    silhouette_score,                   # bigger is better
    silhouette_samples,                 # bigger is better
    calinski_harabasz_score,            # bigger is better
    davies_bouldin_score,               # smaller is better
    homogeneity_completeness_v_measure, # bigger is better
)

# SVM                   分类,回归
from sklearn.svm import SVC, SVR

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    average_precision_score,
    recall_score,
    precision_recall_curve, # 精确率召回率曲线
    roc_curve,              # 根据 y_true, y_score 得到roc曲线
                            # x: 假正类率 (false postive rate, FPR); y: Recall(true postive rate, TPR)
    auc,                    # 根据 x,y 坐标得到roc曲线下的面积
    roc_auc_score,          # 根据 y_true,y_score 得到roc曲线下的面积
                            # roc_auc_score = roc_curve + auc
)

# 管道
from sklearn.pipeline import Pipeline


from joblib import dump # 保存模型
from joblib import load # 加载模型
