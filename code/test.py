from sklearn.datasets import load_iris

from sklearn.feature_extraction import DictVectorizer       # 字典特征抽取
from sklearn.model_selection import train_test_split        # 分割数据
from sklearn.model_selection import StratifiedShuffleSplit  # 分割数据
from sklearn.model_selection import GridSearchCV            # 网格搜索+交叉验证
from sklearn.preprocessing import OneHotEncoder             # one-hot
from sklearn.preprocessing import StandardScaler            # 特征数据标准化StandardScaler
from sklearn.preprocessing import LabelEncoder              # 标签转换为数字
from sklearn.decomposition import PCA                       # PCA

from imblearn.under_sampling import RandomUnderSampler      # 随机下采样

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
