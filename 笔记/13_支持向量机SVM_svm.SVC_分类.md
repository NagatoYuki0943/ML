# 支持向量机
## 学习目标

- 了解什么是SVM算法
- 掌握SVM算法的原理
- 知道SVM算法的损失函数
- 知道SVM算法的核函数
- 了解SVM算法在回归问题中的使用
- 应用SVM算法实现手写数字识别器



# 1 SVM算法简介


![image-20210626174352410](13_支持向量机SVM.assets/image-20210626174352410.png)


![image-20210626174404988](13_支持向量机SVM.assets/image-20210626174404988.png)

![image-20210626174414469](13_支持向量机SVM.assets/image-20210626174414469.png)

![image-20210626174426962](13_支持向量机SVM.assets/image-20210626174426962.png)

![image-20210626174437793](13_支持向量机SVM.assets/image-20210626174437793.png)

案例来源：http://bytesizebio.net/2014/02/05/support-vector-machines-explained-well/
支持向量机直观感受：https://www.youtube.com/watch?v=3liCbRZPrZA

## 2 SVM算法定义
### 2.1 定义
SVM：**SVM全称是supported vector machine（支持向量机），即寻找到一个超平面使样本分成两类，并且间隔最大。**
SVM能够执行线性或非线性分类、回归，甚至是异常值检测任务。它是机器学习领域最受欢迎的模型之一。SVM特别适用于中小型复杂数据集的分类。

![image-20210626174633925](13_支持向量机SVM.assets/image-20210626174633925.png)

### 2.2 超平面最大间隔介绍

![image-20210626174702997](13_支持向量机SVM.assets/image-20210626174702997.png)

上左图显示了三种可能的线性分类器的决策边界：
虚线所代表的模型表现非常糟糕，甚至都无法正确实现分类。其余两个模型在这个训练集上表现堪称完美，但是**它们的决策边界与实例过于接近，导致在面对新实例时，表现可能不会太好。**
**右图中的实线代表SVM分类器的决策边界**，不仅分离了两个类别，且尽可能远离最近的训练实例。

### 2.3 硬间隔和软间隔

#### 2.3.1 硬间隔分类
在上面我们使用超平面进行分割数据的过程中，如果我们严格地让所有实例都不在最大间隔之间，并且位于正确的一边，这就是硬间隔分类。
**硬间隔分类有两个问题，首先，它只在数据是线性可分离的时候才有效；其次，它对异常值非常敏感。**
当有一个额外异常值的鸢尾花数据：左图的数据根本找不出硬间隔，而右图最终显示的决策边界与我们之前所看到的无异常值时的决策边界也大不相同，可能无法很好地泛化。

![image-20210626174747860](13_支持向量机SVM.assets/image-20210626174747860.png)

#### 2.3.2 软间隔分类
要避免这些问题，最好使用更灵活的模型。目标是尽可能在保持最大间隔宽阔和限制间隔违例（即位于最大间隔之上，**甚至在错误的一边的实例）之间找到良好的平衡，这就是软间隔分类。**
要避免这些问题，最好使用更灵活的模型。目标是尽可能在保持间隔宽阔和限制间隔违例之间找到良好的平衡，这就是软间隔分类。

![image-20210626174814037](13_支持向量机SVM.assets/image-20210626174814037.png)

在Scikit-Learn的SVM类中，可以通过超参数C来控制这个平衡：**C值越小，则间隔越宽，但是间隔违例也会越多。**上图显示了在一个非线性可分离数据集上，两个软间隔SVM分类器各自的决策边界和间隔。
左边使用了高C值，分类器的错误样本（间隔违例）较少，但是间隔也较小。
右边使用了低C值，间隔大了很多，但是位于间隔上的实例也更多。看起来第二个分类器的泛化效果更好，因为大多数间隔违例实际上都位于决策边界正确的一边，所以即便是在该训练集上，它做出的错误预测也会更少。



## 3 小结
- SVM算法定义【了解】

    - 寻找到一个超平面使样本分成两类，并且间隔最大。

- 硬间隔和软间隔【知道】

    - 硬间隔
        - 只有在数据是线性可分离的时候才有效
        - 对异常值非常敏感

    - 软间隔
        - 尽可能在保持最大间隔宽阔和限制间隔违例之间找到良好的平衡





# 2 SVM算法api初步使用	svm.SVC


- 知道SVM算法API的用法

```python
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
 max_iter=-1, probability=False, random_state=None, shrinking=True,
 tol=0.001, verbose=False)
```

在拟合后, 这个模型可以用来预测新的值:

```python
>>> clf.predict([[2., 2.]])
array([1])
```



```python
from sklearn import svm
```



```python
# 特征值
x = [[0, 0], [1, 1]]
# 目标值
y = [0, 1]
```



```python
clf = svm.SVC()
clf.fit(x, y)
# 预测,值必须是二维的
clf.predict([[2, 2]])
# 打印 array([1])
```





# 3 SVM算法原理


## 1 定义输入数据
![image-20210626175059521](13_支持向量机SVM.assets/image-20210626175059521.png)

## 2 线性可分支持向量机

给定了上面提出的线性可分训练数据集，通过间隔最大化得到分离超平面为 :$y(x) = w Φ(x)+b$
相应的分类决策函数为： $f(x) = sign(w Φ(x)+b)$
以上决策函数就称为线性可分支持向量机。
这里解释一下$Φ(x)$这个东东。
这是某个确定的特征空间转换函数，它的作用是将x映射到更高的维度，它有一个以后我们经常会⻅到的专有称号”**核函数**“。

> 比如我们看到的特征有2个：
> ![image-20210627193217186](13_支持向量机SVM.assets/image-20210627193217186.png)

以上就是线性可分支持向量机的模型表达式。我们要去求出这样一个模型，或者说这样一个超平面y(x),它能够最优地分离两个集合。
**其实也就是我们要去求一组参数（w,b),使其构建的超平面函数能够最优地分离两个集合。**
如下就是一个最优超平面：

![image-20210626175155226](13_支持向量机SVM.assets/image-20210626175155226.png)

又比如说这样：

![image-20210626175206068](13_支持向量机SVM.assets/image-20210626175206068.png)

阴影部分是一个“过渡带”，“过渡带”的边界是集合中离超平面最近的样本点落在的地方。

## 3 SVM的计算过程与算法步骤
### 3.1 推导目标函数
我们知道了支持向量机是个什么东⻄了。现在我们要去寻找这个支持向量机，也就是寻找一个最优的超平面。
于是我们要建立一个目标函数。那么如何建立呢？

![image-20210626175244676](13_支持向量机SVM.assets/image-20210626175244676.png)

![image-20210626175320439](13_支持向量机SVM.assets/image-20210626175320439.png)

![image-20210626175328389](13_支持向量机SVM.assets/image-20210626175328389.png)

![image-20210626175338050](13_支持向量机SVM.assets/image-20210626175338050.png)

### 3.2 目标函数的求解
到这一步，终于把目标函数给建立起来了。
那么下一步自然是去求目标函数的最优值.
因为目标函数带有一个约束条件，所以**我们可以用拉格朗日乘子法求解。**

#### 3.2.1 朗格朗日乘子法

啥是拉格朗日乘子法呢？
拉格朗日乘子法 (Lagrange multipliers)是**一种寻找多元函数在一组约束下的极值的方法.**
通过引入拉格朗日乘子，可将有 d 个变量与 k 个约束条件的最优化问题转化为具有 d + k 个变量的无约束优化问题求解。
朗格朗日乘子法举例复习

![image-20210626175434458](13_支持向量机SVM.assets/image-20210626175434458.png)

#### 3.2.2 对偶问题
我们要将其转换为对偶问题，变成极大极小值问题：

![image-20210626175451685](13_支持向量机SVM.assets/image-20210626175451685.png)

参考资料：https://wenku.baidu.com/view/7bf945361b37f111f18583d049649b6649d70975.html

![image-20210626175509178](13_支持向量机SVM.assets/image-20210626175509178.png)

![image-20210626175527228](13_支持向量机SVM.assets/image-20210626175527228.png)

#### 3.2.3 整体流程确定
我们用数学表达式来说明上面的过程：

![image-20210626175552267](13_支持向量机SVM.assets/image-20210626175552267.png)

![image-20210626175601818](13_支持向量机SVM.assets/image-20210626175601818.png)

#### 3.2.4 举例

![image-20210626175625321](13_支持向量机SVM.assets/image-20210626175625321.png)

![image-20210626175633236](13_支持向量机SVM.assets/image-20210626175633236.png)

![image-20210626175641457](13_支持向量机SVM.assets/image-20210626175641457.png)

![image-20210626175647357](13_支持向量机SVM.assets/image-20210626175647357.png)

ps:参考的另一种计算方式： https://blog.csdn.net/zhizhjiaodelaoshu/article/details/97112073

## 4 小结

![image-20210626175809477](13_支持向量机SVM.assets/image-20210626175809477.png)

![image-20210626175848497](13_支持向量机SVM.assets/image-20210626175848497.png)

# 4 SVM的损失函数
![image-20210626175928768](13_支持向量机SVM.assets/image-20210626175928768.png)

## 小结
- SVM的损失函数
    - 0/1损失函数
    - Hinge损失函数
    - Logistic损失函数



# 5 SVM的核方法
## 学习目标

- 知道SVM的核方法
- 了解常⻅的核函数

【SVM + 核函数】 具有极大威力。
核函数并不是SVM特有的，核函数可以和其他算法也进行结合，只是核函数与SVM结合的优势非常大。

## 1 什么是核函数

### 1.1 核函数概念

核函数，是将原始输入空间映射到新的特征空间，从而，使得原本线性不可分的样本可能在核空间可分。

![image-20210626181120587](13_支持向量机SVM.assets/image-20210626181120587.png)

下图所示的两类数据，分别分布为两个圆圈的形状，这样的数据本身就是线性不可分的，此时该如何把这两类数据分开呢?

![image-20210626181145350](13_支持向量机SVM.assets/image-20210626181145350.png)

### 1.2 核函数举例
#### 1.2.1 核方法举例1：

![image-20210626181246826](13_支持向量机SVM.assets/image-20210626181246826.png)

![image-20210626181256828](13_支持向量机SVM.assets/image-20210626181256828.png)

![image-20210626181304907](13_支持向量机SVM.assets/image-20210626181304907.png)

#### 1.2.2 核方法举例2：

- 下面这张图位于第一、二象限内。我们关注红色的⻔，以及“北京四合院”这几个字和下面的紫色的字母。
- 我们把红色的⻔上的点看成是“+”数据，字母上的点看成是“-”数据，它们的横、纵坐标是两个特征。
- 显然，在这个二维空间内，“+”“-”两类数据不是线性可分的。

![image-20210626181343687](13_支持向量机SVM.assets/image-20210626181343687.png)

![image-20210626181351747](13_支持向量机SVM.assets/image-20210626181351747.png)

![image-20210626181405297](13_支持向量机SVM.assets/image-20210626181405297.png)

## 2 常⻅核函数

![image-20210626181434695](13_支持向量机SVM.assets/image-20210626181434695.png)

> 1.多项核中，d=1时，退化为线性核；
>
> 2.高斯核亦称为RBF核。

### 线性核和多项式核：

- 这两种核的作用也是首先在属性空间中找到一些点，把这些点当做base，核函数的作用就是找与该点距离和⻆度满足某种关系的样本点。
- 当样本点与该点的夹⻆近乎垂直时，两个样本的欧式⻓度必须非常⻓才能保证满足线性核函数大于0；而当样
    本点与base点的方向相同时，⻓度就不必很⻓；而当方向相反时，核函数值就是负的，被判为反类。即，它在空间上划分出一个梭形，按照梭形来进行正反类划分。

### RBF核(推荐)

- 高斯核函数就是在属性空间中找到一些点，这些点可以是也可以不是样本点，把这些点当做base，以这些
    base为圆心向外扩展，扩展半径即为带宽，即可划分数据。
- 换句话说，在属性空间中找到一些超圆，用这些超圆来判定正反类。

### oid核：

- 同样地是定义一些base，
- 核函数就是将线性核函数经过一个tanh函数进行处理，把值域限制在了-1到1上。

## 3 核的选择,推荐RBF

- 总之，都是在定义距离，大于该距离，判为正，小于该距离，判为负。至于选择哪一种核函数，要根据具体的样本分布情况来确定。
    一般有如下指导规则：
    - 1） 如果Feature的数量很大，甚至和样本数量差不多时，往往线性可分，这时选用LR或者线性核Linear；
    - 2） 如果Feature的数量很小，样本数量正常，不算多也不算少，这时选用RBF核；
    - 3） 如果Feature的数量很小，而样本的数量很大，这时手动添加一些Feature，使得线性可分，然后选用LR或者线性核Linear；
    - 4） 多项式核一般很少使用，效率不高，结果也不优于RBF；
    - 5） Linear核参数少，速度快；RBF核参数多，分类结果非常依赖于参数，需要交叉验证或网格搜索最佳参数，比较耗时；
    - 6）应用最广的应该就是RBF核，无论是小样本还是大样本，高维还是低维等情况，RBF核函数均适用。



# 6 SVM回归
## 学习目标

- 了解SVM回归的实现原理

SVM回归是让尽可能多的实例位于预测线上，同时限制间隔违例（也就是不在预测线距上的实例）。
线距的宽度由超参数ε控制。

![image-20210626181655822](13_支持向量机SVM.assets/image-20210626181655822.png)

# 7 SVM算法api再介绍

## 1 SVM算法api综述

- SVM方法既可以用于分类（二/多分类），也可用于回归和异常值检测。
- SVM具有良好的鲁棒性，对未知数据拥有很强的泛化能力，**特别是在数据量较少的情况下**，相较其他传统机器学习算法具有更优的性能。

使用SVM作为模型时，通常采用如下流程：

1. 对样本数据进行归一化
2. 应用核函数对样本进行映射**（最常采用和核函数是RBF和Linear，在样本线性可分时，Linear效果要比RBF好）**
3. 用cross-validation和grid-search对超参数进行优选
4. 用最优参数训练得到模型
5. 测试

**sklearn中支持向量分类主要有三种方法：SVC、NuSVC、LinearSVC，扩展为三个支持向量回Q归方法：SVR、NuSVR、LinearSVR。**

- SVC和NuSVC方法基本一致，唯一区别就是损失函数的度量方式不同
    - NuSVC中的nu参数和SVC中的C参数；
- LinearSVC是实现线性核函数的支持向量分类，没有kernel参数。

## 2 SVC
```python
class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3,coef0=0.0,random_state=None)
```

- **C: 惩罚系数，用来控制损失函数的惩罚系数，类似于线性回归中的正则化系数。**
    - **C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情
        况，这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。**
    - **C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。**
- **kernel: 算法中采用的核函数类型，核函数是用来将非线性问题转化为线性问题的一种方法。**
    - 参数选择有RBF, Linear, Poly, Sigmoid或者自定义一个核函数。
        - **默认的是"RBF"，即径向基核，也就是高斯核函数； (最常用)**
        - 而Linear指的是线性核函数，
        - Poly指的是多项式核，
        - Sigmoid指的是双曲正切函数tanh核；。
- **degree:**
    - **当指定kernel为'poly'时，表示选择的多项式的最高次数，默认为三次多项式；**
    - 若指定kernel不是'poly'，则忽略，即该参数只对'poly'有用。
        - 多项式核函数是将低维的输入空间映射到高维的特征空间。
- coef0: 核函数常数值(y=kx+b中的b值)，
    - 只有‘poly’和‘sigmoid’核函数有，默认值是0。

## 3 NuSVC

```python
class sklearn.svm.NuSVC(nu=0.5)
```

- nu： 训练误差部分的上限和支持向量部分的下限，取值在（0，1）之间，默认是0.5

## 4 LinearSVC  

```python
class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=1.0)
```

- **penalty:正则化参数，**
    - L1和L2两种参数可选，仅LinearSVC有。
- **loss:损失函数，**
    - **有hinge和squared_hinge两种可选，前者又称L1损失，后者称为L2损失，默认是squared_hinge，**
    - **其中hinge是SVM的标准损失，squared_hinge是hinge的平方**
- dual:是否转化为对偶问题求解，默认是True。
- **C:惩罚系数，**
    - **用来控制损失函数的惩罚系数，类似于线性回归中的正则化系数。**

## 5 小结
- SVM的核方法
    - 将原始输入空间映射到新的特征空间，从而，使得原本线性不可分的样本可能在核空间可分。
- SVM算法api
    - sklearn.svm.SVC
    - sklearn.svm.NuSVC
    - sklearn.svm.LinearSVC





# 8 案例：数字识别器
## 1 案例背景介绍

![image-20210626182313300](13_支持向量机SVM.assets/image-20210626182313300.png)

MNIST（“修改后的国家标准与技术研究所”）是计算机视觉事实上的“hello world”数据集。自1999年发布以来，这一经典的手写图像数据集已成为分类算法基准测试的基础。随着新的机器学习技术的出现，MNIST仍然是研究人员和学习者的可靠资源。本次案例中，我们的目标是**从数万个手写图像的数据集中正确识别数字。**

## 2 数据介绍
数据文件train.csv和test.csv包含从0到9的手绘数字的灰度图像。
**每个图像的高度为28个像素，宽度为28个像素，总共为784个像素。**
每个像素具有与其相关联的单个像素值，指示该像素的亮度或暗度，较高的数字意味着较暗。**该像素值是0到255之间的整数，包括0和255。**
**训练数据集（train.csv）有785列。第一列称为“标签”，是用户绘制的数字。其余列包含关联图像的像素值。**
训练集中的每个像素列都具有像pixelx这样的名称，其中x是0到783之间的整数，包括0和783。为了在图像上定位该像素，假设我们已经将x分解为x = i * 28 + j，其中i和j是0到27之间的整数，包括0和27。然后，pixelx位于28 x 28矩阵的第i行和第j列上（索引为零）。
例如，pixel31表示从左边开始的第四列中的像素，以及从顶部开始的第二行，如下面的ascii图中所示。
在视觉上，如果我们省略“像素”前缀，像素组成图像如下：

```python
000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 | | | | ...... | |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783
```

![image-20210626182402068](13_支持向量机SVM.assets/image-20210626182402068.png)

测试数据集（test.csv）与训练集相同，只是它不包含“标签”列。

## 3 案例实现

### 1 导入包

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 数据分隔
from sklearn.model_selection import train_test_split
# 支持向量机
from sklearn import svm
```

### 2 获取数据

```python
train = pd.read_csv('../data/day10/train.csv')
train.head()
# 每一行都是一副图片
```

![image-20210628103334617](13_支持向量机SVM.assets/image-20210628103334617.png)

```python
train.shape
# 785个特征值
```

![image-20210628103404050](13_支持向量机SVM.assets/image-20210628103404050.png)

#### 2.1 获取特征值/目标值

```python
# 获取图像,去除第一列的标签
train_image = train.iloc[:, 1:]
train_image.head()
```

![image-20210628103443433](13_支持向量机SVM.assets/image-20210628103443433.png)

```python
# 获取label
train_label = train.iloc[:, :1]
train_label.head()
```

![image-20210628103505513](13_支持向量机SVM.assets/image-20210628103505513.png)

#### 2.2 查看具体图像

```python
# 每一行都是一副图片
train_image.iloc[0].values
```

![image-20210628103522964](13_支持向量机SVM.assets/image-20210628103522964.png)

```python
# 修改成默认形状
num = train_image.iloc[0].values.reshape(28, 28)
plt.imshow(num)
# 关掉轴
plt.axis('off')
plt.show()
```

![image-20210628103555861](13_支持向量机SVM.assets/image-20210628103555861.png)

```python
# 定义画图函数
def to_plot(n):
    # 每一行都是一副图片
    num = train_image.iloc[n].values.reshape(28, 28) 
    plt.imshow(num)
    # 关掉轴
    plt.axis('off')
    plt.show()
    
to_plot(1)
to_plot(100)
```

![image-20210628103611266](13_支持向量机SVM.assets/image-20210628103611266.png)



### 3 数据基本处理

```python
train_image.head()
```

![image-20210628103633834](13_支持向量机SVM.assets/image-20210628103633834.png)

#### 3.1 数据归一化,整体除以255

```python
# 特征值处理
train_image1 = train_image.values / 255
train_image1
```

![image-20210628103708908](13_支持向量机SVM.assets/image-20210628103708908.png)

```python
# 目标值处理,二维转为一维数据
train_label1 = train_label.values
train_label1
```

![image-20210628103720960](13_支持向量机SVM.assets/image-20210628103720960.png)

#### 3.2 数据集分隔

```python
x_train, x_val, y_train, y_val = train_test_split(train_image1, train_label1, train_size=0.8, random_state=25)
x_train.shape, x_val.shape
# 列数太多,特征太多
```

![image-20210628103800979](13_支持向量机SVM.assets/image-20210628103800979.png)

### 4 特征降维和模型训练

多次使用PCA降维,确定最后的最优模型

sklearn.decomposition.PCA(n_components=None)

- 将数据分解为较低维数空间
- n_components:
  - **小数：表示保留百分之多少的信息**
  - **整数：减少到多少特征**
- PCA.fit_transform(X) X:numpy array格式的数据[n_samples,n_features]
- 返回值：转换后指定维度的array

#### 4.1 特征降维

```python
import time
# PCA特征降维
from sklearn.decomposition import PCA

def n_components_analysis(n, x_train, x_val, y_train, y_val):
    # 记录开始时间
    start = time.time()
    
    # 构造PCA降维实现
    pca = PCA(n_components=n)
    print('特征降维传递的参数为:{}'.format(n))
    # fit和transform可以分开也可以一起使用,分开使用的情况是针对多个数据进行相同的处理
    pca.fit(x_train)
    
    # 在训练集和测试集进行降维
    x_train_pca = pca.transform(x_train)
    x_val_pca = pca.transform(x_val)
    
    # 利用svm.SVC进行训练
    print('开始使用SVC训练')
    ss = svm.SVC()
    ss.fit(x_train_pca, y_train)
    
    # 获取accuracy结果
    accuracy = ss.score(x_val_pca, y_val)
    
    # 记录结束时间
    end = time.time()
    
    print('准确率是:{},消耗时间:{}'.format(accuracy, int(end - start)))
    
    return accuracy

# 传递多个n_components值,寻找合理的n_components
n_components = np.linspace(0.70, 0.85, num=5)
accuracy = []
for n in n_components:
    temp = n_components_analysis(n,  x_train, x_val, y_train, y_val)
    accuracy.append(temp)
```

![image-20210628103853241](13_支持向量机SVM.assets/image-20210628103853241.png)

#### 4.2 准确率可视化显示

```python
plt.plot(n_components, accuracy, 'red')
plt.grid()
plt.show()
```

![image-20210628103915443](13_支持向量机SVM.assets/image-20210628103915443.png)

经过图形展示,选择合理的n_components,最后的综合考虑结果是0.80

### 5 确定最优模型

```python
pca = PCA(n_components=0.80)
# fit和transform可以分开也可以一起使用,分开使用的情况是针对多个数据进行相同的处理
pca.fit(x_train)
# 显示训练后的特征值
pca.n_components_
# 剩余43列
```

43

```
# 修改特征值
x_train_pca = pca.transform(x_train)
x_val_pca = pca.transform(x_val)
x_train_pca.shape, x_val_pca.shape
# 都只剩余43列了
```

![image-20210628103949031](13_支持向量机SVM.assets/image-20210628103949031.png)



```python
# 训练比较优的模型,计算accuracy
ss1 = svm.SVC()
ss1.fit(x_train_pca, y_train)
ss1.score(x_val_pca, y_val)
# 0.9808333333333333
```

### 6 test训练集测试

```python
test = pd.read_csv('../data/day10/test.csv')
test.head()
# 只有特征值,没有目标值
```

![image-20210628105306221](13_支持向量机SVM.assets/image-20210628105306221.png)

```python
# 特征降维
test_pca = pca.transform(test)
test_pca.shape
# (28000, 43)
```

```python
# 预测数据
y_pre = ss1.predict(test_pca)
y_pre
```

![image-20210628105349685](13_支持向量机SVM.assets/image-20210628105349685.png)

```python
# 转换为df
res = pd.DataFrame(y_pre, columns=['number'])
res
```

![image-20210628105402801](13_支持向量机SVM.assets/image-20210628105402801.png)

```python
# 保存到文件
res.to_csv('number.csv', index=False)
```





# 9 SVM总结
## SVM基本综述
- SVM是一种二类分类模型。
- 它的基本模型是在特征空间中寻找间隔最大化的分离超平面的线性分类器。
    - 1）当训练样本线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机；
    - 2）当训练数据近似线性可分时，引入松弛变量，通过软间隔最大化，学习一个线性分类器，即线性支持向量机；
    - 3）当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。
      

## SVM的优缺点

- SVM的优点：
    - 在高维空间中非常高效；
    - 即使在数据维度比样本数量大的情况下仍然有效；
    - 在决策函数（称为支持向量）中使用训练集的子集,因此它也是高效利用内存的；
    - 通用性：不同的核函数与特定的决策函数一一对应；
- SVM的缺点：
    - 如果特征数量比样本数量大得多，在选择核函数时要避免过拟合；
    - 对缺失数据敏感;
    - 对于核函数的高维映射解释力不强
