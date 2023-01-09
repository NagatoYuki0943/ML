# 特征预处理	特征值处理,目标值不处理

**都有fit, transform, fit_transform,使用了fit之后就能用transform**

```
xx.fit(??)
xx.transform(??)
xx.transform(??)
```

or

```
xx.fit_transform(??)
xx.transform(??)
```

----

## 1 什么是特征预处理 


### 1.1 特征预处理定义

通过一些**转换函数**将特征数据**转换成更加适合算法模型**的特征数据过程

![image-20210616173757205](101_特征标准化_StandardScaler.assets/image-20210616173757205.png)

为什么我们要进行归一化/标准化？
特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标
结果，使得一些算法无法学习到其它的特征

### 1.2 包含内容(数值型数据的无量纲化)

- 归一化
- 标准化

### 1.3 特征预处理API	sklearn.preprocessing

```python
sklearn.preprocessing
```



## 2 归一化	[0,1]

### 2.1 定义

通过对原始数据进行变换把数据映射到(默认为[0,1])之间

### 2.2 公式

![image-20210616174024805](101_特征标准化_StandardScaler.assets/image-20210616174024805.png)

![image-20210616174033958](101_特征标准化_StandardScaler.assets/image-20210616174033958.png)

### 2.3 API	preprocessing.MinMaxScaler ()

- sklearn.preprocessing.MinMaxScaler (feature_range=(0,1)... )
  - feature_range=(0,1)    指的是归一化的数字范围,随便写
  - MinMaxScalar.fit_transform(X)
    - X 指的是要转换的列名
    - X:numpy array格式的数据[n_samples,n_features]
  - 返回值：转换后的形状相同的array



### 2.4 数据计算

我们对以下数据进行运算，在dating.txt中。保存的就是之前的约会对象数据

```python
#公里数,冰淇淋消耗数,游戏时间,目标数(目标不能转换)
milage,Liters,Consumtime,target
40920,8.326976,0.953952,3
14488,7.153469,1.673904,2
26052,1.441871,0.805124,1
75136,13.147394,0.428964,1
38344,1.669788,0.134296,1
```

- 分析

1、实例化MinMaxScalar

2、通过fit_transform转换

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def minmax_demo():
    """
    归一化演示
    :return: None
    """
    data = pd.read_csv('../data/dating.txt')
    print(data)
    # 1.实例化一个转换器类               归一化的数值范围
    transfer = MinMaxScaler(feature_range=(2, 3))
    # 2.调用fit_transform             范围是特征值,不是目标值	pandas是先列后行
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print("最小值最大值归一化处理的结果：\n", data)
    return None
```

返回结果：

```python
     milage     Liters  Consumtime  target
0     40920   8.326976    0.953952       3
1     14488   7.153469    1.673904       2
2     26052   1.441871    0.805124       1
3     75136  13.147394    0.428964       1
..      ...        ...         ...     ...
998   48111   9.134528    0.728045       3
999   43757   7.882601    1.332446       3
[1000 rows x 4 columns]
最小值最大值归一化处理的结果：
 [[ 2.44832535  2.39805139  2.56233353]
 [ 2.15873259  2.34195467  2.98724416]
 [ 2.28542943  2.06892523  2.47449629]
 ..., 
 [ 2.29115949  2.50910294  2.51079493]
 [ 2.52711097  2.43665451  2.4290048 ]
 [ 2.47940793  2.3768091   2.78571804]]
```

### 2.5 归一化总结

注意最大值最小值是变化的，另外，最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。



## 3 标准化	均值为0,标准差为1范围

### 3.1 定义

通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内

### 3.2 公式

![image-20210616174252935](101_特征标准化_StandardScaler.assets/image-20210616174252935.png)

对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变
对于标准化来说：如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变
较小。

### 3.3 API	preprocessing.StandardScaler( )

- sklearn.preprocessing.StandardScaler( )
  - 处理之后每列来说所有数据都聚集在均值0附近标准差差为1
  - StandardScaler.fit_transform(X)
    - X:numpy array格式的数据[n_samples,n_features]
  - 返回值：转换后的形状相同的array

### 3.4 数据计算

同样对上面的数据进行处理

- 分析

1、实例化StandardScaler
2、通过fit_transform转换

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
def stand_demo():
    """
    标准化演示
    :return: None
    """
    data = pd.read_csv("../data/dating.txt")
    print(data)
    # 1、实例化一个转换器类
    transfer = StandardScaler()
    # 2、调用fit_transform				范围是特征值,不是目标值	pandas是先列后行
    data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
    print("标准化的结果:\n", data)
    print("每一列特征的平均值：\n", transfer.mean_)
    print("每一列特征的方差：\n", transfer.var_)
    return None
```

返回结果：

```python
     milage     Liters  Consumtime  target
0     40920   8.326976    0.953952       3
1     14488   7.153469    1.673904       2
2     26052   1.441871    0.805124       1
..      ...        ...         ...     ...
997   26575  10.650102    0.866627       3
998   48111   9.134528    0.728045       3
999   43757   7.882601    1.332446       3
[1000 rows x 4 columns]
标准化的结果:
 [[ 0.33193158  0.41660188  0.24523407]
 [-0.87247784  0.13992897  1.69385734]
 [-0.34554872 -1.20667094 -0.05422437]
 ..., 
 [-0.32171752  0.96431572  0.06952649]
 [ 0.65959911  0.60699509 -0.20931587]
 [ 0.46120328  0.31183342  1.00680598]]
每一列特征的平均值：
 [  3.36354210e+04   6.55996083e+00   8.32072997e-01]
每一列特征的方差：
 [  4.81628039e+08   1.79902874e+01   2.46999554e-01]
```



### 3.5 标准化总结

在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。



## 4 总结

- 什么是特征工程【知道】

  - 定义
    - 通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程
  - 包含内容:
    - 归一化
    - 标准化
  - 归一化【知道】
    - 定义:
      - 对原始数据进行变换把数据映射到(默认为[0,1])之间
    - api:
      - sklearn.preprocessing.MinMaxScaler (feature_range=(0,1)... )
      - 参数:feature_range -- 自己指定范围,默认0-1
    - 总结:
      - 鲁棒性比较差(容易受到异常点的影响)
      - 只适合传统精确小数据场景(以后不会用你了)

  - 标准化【掌握】
    - 定义:
      - 对原始数据进行变换把数据变换到均值为0,标准差为1范围内
    - api:
      - sklearn.preprocessing.StandardScaler( )
    - 总结:
      - 异常值对我影响小
      - 适合现代嘈杂大数据场景(以后就是用你了)