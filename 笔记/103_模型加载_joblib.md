# 模型的保存和加载

## 学习目标

- 知道sklearn中模型的保存和加载

------

## 11.1 sklearn模型的保存和加载API	joblib.dump	joblib.load

- import joblib
  - 保存：joblib.dump(estimator, 'test.pkl')
  - 加载：estimator = joblib.load('test.pkl')
- **可以将训练分为多次,训练一会儿就保存一次,再加载训练**

## 11.2 线性回归的模型保存加载案例

## joblib.dump(estimator, 'test.pkl')

```python
'''
import joblib
    保存：joblib.dump(estimator, 'test.pkl')
    加载：estimator = joblib.load('test.pkl')
'''

# 数据保存
import joblib

# 数据引入
from sklearn.datasets import load_boston
# 数据分隔
from sklearn.model_selection import train_test_split
# 标准化
from sklearn.preprocessing import StandardScaler
# 机器学习 线性回归api  岭回归
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

# 模型评估
from sklearn.metrics import mean_squared_error

# 1.获取数据
data = load_boston()


# 2.数据集划分
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=25)


# 3.特征值标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# 4.机器学习-线性回归(正规方程)
# alpha: 正则化力度
#estimator = Ridge(alpha=1)

estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10))
estimator.fit(x_train, y_train)


# 5.保存模型
joblib.dump(estimator, './data/boston.pkl')



# 6.模型评估
# 6.1 获取系数等值
y_predict = estimator.predict(x_test)
print("预测值为:\n", y_predict)
print("模型中的系数为:\n", estimator.coef_)
#   [-0.53782282  0.75275911  0.14874627  0.1135628  -1.69363856  2.70595671
#  -0.13069087 -2.7364033   2.35848648 -2.26750395 -2.05249462  0.98584826
#  -3.42164338]
print("模型中的偏置为:\n", estimator.intercept_)  # 22.23693931398419


# 6.2 评价
# 均方误差  越小越好
error = mean_squared_error(y_test, y_predict)
print("误差为:\n", error)  # 18.150788844939925

```

## estimator = joblib.load('test.pkl')

```python
'''
import joblib
    保存：joblib.dump(estimator, 'test.pkl')
    加载：estimator = joblib.load('test.pkl')
'''

# 数据保存
import joblib

# 数据引入
from sklearn.datasets import load_boston
# 数据分隔
from sklearn.model_selection import train_test_split
# 标准化
from sklearn.preprocessing import StandardScaler
# 机器学习 线性回归api  岭回归
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

# 模型评估
from sklearn.metrics import mean_squared_error


# 1.获取数据
data = load_boston()


# 2.数据集划分
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=25)


# 3.特征值标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)



# 4.模型加载
estimator = joblib.load('./data/boston.pkl')



# 5.模型评估
# 5.1 获取系数等值
y_predict = estimator.predict(x_test)
print("预测值为:\n", y_predict)
print("模型中的系数为:\n", estimator.coef_)
#   [-0.53782282  0.75275911  0.14874627  0.1135628  -1.69363856  2.70595671
#  -0.13069087 -2.7364033   2.35848648 -2.26750395 -2.05249462  0.98584826
#  -3.42164338]
print("模型中的偏置为:\n", estimator.intercept_)  # 22.23693931398419


# 5.2 评价
# 均方误差  越小越好
error = mean_squared_error(y_test, y_predict)
print("误差为:\n", error)  # 18.150788844939925
```



## 11.3 小结

- sklearn.externals import joblib【知道】
  - 保存：joblib.dump(estimator, 'test.pkl')
  - 加载：estimator = joblib.load('test.pkl')
  - 注意：
    - 1.保存文件，后缀名是**.pkl
    - 2.加载模型是需要通过一个变量进行承接