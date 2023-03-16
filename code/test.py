from sklearn.datasets import load_iris
from sklearn.datasets import make_classification            # 准备类别不平衡数据

from sklearn.feature_extraction import DictVectorizer       # 字典特征抽取
from sklearn.feature_extraction.text import CountVectorizer # 文本特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer # 文本特征抽取

from sklearn.preprocessing import (
    OneHotEncoder,                                          # one-hot
    MinMaxScaler,                                           # MinMax
    StandardScaler,                                         # 特征数据标准化StandardScaler
    LabelEncoder,                                           # 标签转换为数字
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
    GridSearchCV,                                           # 网格搜索+交叉验证
    RandomizedSearchCV,                                     # 随机搜索+交叉验证
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
)

# 树
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import export_graphviz

# 随机森林
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from joblib import dump # 保存模型
from joblib import load # 加载模型
