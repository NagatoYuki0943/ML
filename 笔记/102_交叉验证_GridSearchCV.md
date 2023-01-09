# 交叉验证，网格搜索    GridSearchCV

## 10.1 什么是交叉验证(cross validation)

交叉验证：将拿到的训练数据，分为训练和验证集。以下图为例：将数据分成4份，其中一份作为验证集。然后经过4次(组)的测试，每次都更换不同的验证集。即得到4组模型的结果，取平均值作为最终结果。又称4折交叉验证。

**训练集分成n份,就当做n折交叉验证;测试集不发生变化**

### 1.1 分析

我们之前知道数据分为训练集和测试集，但是为了让从训练得到模型结果更加准确。做以下处理

**训练集分成n份,就当做n折交叉验证;测试集不发生变化**

- **训练集：训练集+验证集**
- **测试集：测试集**

![image-20210616175847490](102_交叉验证_GridSearchCV.assets/image-20210616175847490.png)

### 1.2 为什么需要交叉验证

交叉验证目的：为了让被评估的模型更加准确可信
问题：**这个只是让被评估的模型更加准确可信**，那么怎么选择或者调优参数呢？



## 10.2 什么是网格搜索(Grid Search)

### 超参数: 需要手动指定的参数

通常情况下，有**很多参数是需要手动指定的（如k-近邻算法中的K值），这种叫超参数**。但是手动过程繁杂，所以需要对模型预设几种超参数组合。每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。

| K值  | k=3   | k=5   | k=7   |
| ---- | ----- | ----- | ----- |
| 模型 | 模型1 | 模型2 | 模型3 |

## 10.3 交叉验证，网格搜索（模型选择与调优）API：	model_selection.GridSearchCV()

- sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)
  - 对估计器的指定参数值进行详尽搜索
    - estimator：估计器对象
    - param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
    - cv：指定几折交叉验证
    - n_jobs:  几个处理器运行,-1为全部运行
  - 方法
    - fit：输入训练数据
    - predict: 预测数据
    - score：准确率
  - 结果分析：
    - best_score_:在交叉验证中验证的最好结果
    - best_estimator_：最好的参数模型
    - cv_results_:每次交叉验证后的验证集准确率结果和训练集准确率结果

## 10.4 鸢尾花案例增加K值调优

- 使用GridSearchCV构建估计器

```python
# 引入文件
from sklearn.datasets import load_iris
# 划分训练测试
from sklearn.model_selection import train_test_split
# 标准化处理
from sklearn.preprocessing import StandardScaler
# 邻居
from sklearn.neighbors import KNeighborsClassifier
# 交叉
from sklearn.model_selection import GridSearchCV


# 1、获取数据集
iris = load_iris()
# 2、数据基本处理 -- 划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
# 3、特征工程：标准化
# 实例化一个转换器类
transfer = StandardScaler()
# 调用fit_transform
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4、KNN预估器流程
#  4.1 实例化预估器类
estimator = KNeighborsClassifier()
# 4.2 模型选择与调优——网格搜索和交叉验证
# 准备要调的超参数
param_dict = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
# 4.3 fit数据进行训练
estimator.fit(x_train, y_train)
# 5、评估模型效果
# 方法a：比对预测结果和真实值
y_predict = estimator.predict(x_test)
print("比对预测结果和真实值：\n", y_predict == y_test)
# 方法b：直接计算准确率
score = estimator.score(x_test, y_test)
print("直接计算准确率：\n", score)
```

- 然后进行评估查看最终选择的结果和交叉验证的结果

```python
print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)
```

- 最终结果

```python
比对预测结果和真实值：
 [ True  True  True  True  True  True  True False  True  True  True  True
  True  True  True  True  True  True False  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True]
直接计算准确率：
 0.947368421053
在交叉验证中验证的最好结果：
 0.973214285714
最好的参数模型：
 KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
每次交叉验证后的准确率结果：
 {'mean_fit_time': array([ 0.00114751,  0.00027037,  0.00024462]), 'std_fit_time': array([  1.13901511e-03,   1
.25300249e-05,   1.11011951e-05]), 'mean_score_time': array([ 0.00085751,  0.00048693,  0.00045625]), 'std_scor
e_time': array([  3.52785082e-04,   2.87650037e-05,   5.29673344e-06]), 'param_n_neighbors': masked_array(data 
= [1 3 5],
             mask = [False False False],
       fill_value = ?)
, 'params': [{'n_neighbors': 1}, {'n_neighbors': 3}, {'n_neighbors': 5}], 'split0_test_score': array([ 0.973684
21,  0.97368421,  0.97368421]), 'split1_test_score': array([ 0.97297297,  0.97297297,  0.97297297]), 'split2_te
st_score': array([ 0.94594595,  0.89189189,  0.97297297]), 'mean_test_score': array([ 0.96428571,  0.94642857, 
 0.97321429]), 'std_test_score': array([ 0.01288472,  0.03830641,  0.00033675]), 'rank_test_score': array([2, 3
, 1], dtype=int32), 'split0_train_score': array([ 1.        ,  0.95945946,  0.97297297]), 'split1_train_score':
 array([ 1.        ,  0.96      ,  0.97333333]), 'split2_train_score': array([ 1.  ,  0.96,  0.96]), 'mean_trai
n_score': array([ 1.        ,  0.95981982,  0.96876877]), 'std_train_score': array([ 0.        ,  0.00025481,  
0.0062022 ])}
```

## 10.5 总结

- 交叉验证【知道】
  - 定义：
    - 将拿到的训练数据，分为训练和验证集
    - n折交叉验证
  - 分割方式：
    - 训练集：训练集+验证集
    - 测试集：测试集
  - 为什么需要交叉验证
    - 为了让被评估的模型更加准确可信
    - 注意：交叉验证不能提高模型的准确率
- 网格搜索【知道】
  - 超参数:
    - sklearn中,需要手动指定的参数,叫做超参数
  - 网格搜索就是把这些超参数的值,通过字典的形式传递进去,然后进行选择最优值
- api【知道】
  - sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)
    - estimator -- 选择了哪个训练模型
    - param_grid -- 需要传递的超参数
    - cv -- 几折交叉验证





# 案例

## lightGBM案例介绍 day12

接下来，通过鸢尾花数据集对lightGBM的基本使用，做一个介绍。

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
```

### 1 加载数据，对数据进行基本处理

```python
# 加载数据
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)
```

### 2 模型训练 lgb.LGBMRegressor

```python
# regression 回归
gbm = lgb.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=20)
gbm.fit(x_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)

gbm.score(x_test, y_test)
# 0.8118770336882779
```

### 3 网格搜索 GridSearchCV

```python
#  网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid, cv=4)
gbm.fit(x_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)
# Best parameters found by grid search are: {'learning_rate': 0.1, 'n_estimators': 40}
```

### 4 模型调优训练

```python
gbm = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.1, n_estimators=40)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
gbm.score(X_test, y_test)
# 0.9536626296481988
```

