# -*- coding:utf-8 -*-
'''
作者：helloWorld
日期：2025年11月02日
4.10 实战Kaggle比赛：预测房价
'''
import hashlib
import os
import tarfile
import zipfile
import requests
import pandas as pd
import torch
import torch.nn as nn
# from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 4.10.1 下载和缓存数据集
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


# assert 'kaggle_house_train' in DATA_HUB, f"{'kaggle_house_train'} 不存在于 {DATA_HUB}"
def download(name, cache_dir=os.path.join('.', 'data')):  
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    # url.split('/')[-1] 表示获取分割后的列表中的最后一个元素。自动提取文件名
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        # hashlib 是Python的哈希函数库，用于生成数据的"数字指纹"（哈希值）。
        """
        hashlib.sha1()：创建一个SHA1哈希计算器
        .update(data)：向计算器添加数据
        .hexdigest()：获取最终的哈希值
        """
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            # 一次读取1MB数据，但是文件可能超过1MB，所以需要用while True
            """
            如果要读取2MB数据，为什么不把2MB使用while True读取完再最后一次性添加哈希计算，
            而是读完一个数据块，比如1MB就添加一次哈希计算？
            答：  方式	   内存使用	  数据处理	    适用场景
               收集后计算	   线性增长	存储所有原始数据	 小文件
               分块累积计算	 恒定	只保留哈希状态	   任意大小文件
            """
            while True:
                data = f.read(1048576)  # 一次性读取1MB数据
                if not data:
                    break
                sha1.update(data)
        """
        下面的if语句作用：
        检查计算出的文件哈希值是否与预期哈希值匹配：
        如果匹配 → 文件完整无误，返回文件路径
        如果不匹配 → 文件可能损坏，继续执行后续代码（通常是重新下载）
        """
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    """
    下载代码：从网络获取数据并保存到本地，哈希验证完再下载
    哈希验证代码：确保数据的完整性和正确性，一般最先执行
    """
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  
    """下载并解压zip/tar文件"""
    fname = download(name)
    # 获取文件所在的目录路径
    base_dir = os.path.dirname(fname)
    """
    下面一行是 分割文件名和扩展名，例如：
    fname = "dataset.tar.gz"
    data_dir, ext = os.path.splitext(fname)  
    # data_dir = "dataset.tar"
    # ext = ".gz"

    fname = "datafile.csv"
    data_dir, ext = os.path.splitext(fname)
    # data_dir = "datafile" 
    # ext = ".csv"
    """
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar/tar.gz文件可以被解压缩'
    # 解压到指定目录
    fp.extractall(base_dir)
    """
    if folder:  # 如果folder不是None、空字符串等"假值"
        return os.path.join(base_dir, folder)
    else:
        return data_dir
    """
    return os.path.join(base_dir, folder) if folder else data_dir


# 一般要先批量下载download_all()，后按需解压download_extract(name, folder=None)
def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:  # 遍历所有数据集
        download(name)  # 只下载，不解压


DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

"""
如果不知道文件格式怎么办？
方案1：查看DATA_HUB定义
# 先查看DATA_HUB中的URL，从URL推断格式
print(DATA_HUB['kaggle_house_train'])
# 输出: ('http://.../kaggle_house_pred_train.csv', 'hash123')
# ↑ 从URL可以看到是 .csv 文件
"""
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 训练数据集包括1460个样本，每个样本80个特征和1个标签，
# 而测试数据集包含1459个样本，每个样本80个特征。
print(train_data.shape)  # (1460, 81)
print(test_data.shape)  # (1459, 80)
# 打印训练数据的前4行，以及第0、1、2、3列和倒数第3、2、1列
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 删除id，训练集：保留所有行，从第1列到倒数第2列（不包括最后一列）。测试集：保留所有行，从第1列到最后一列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features[train_data].shape)
# print(all_features[test_data].shape)
# print(all_features[train_data].iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# .iloc 是pandas中非常重要的基于位置的索引器。
print(all_features.iloc[:len(train_data)].shape)  # 训练集部分，(1460, 79)
print(all_features.iloc[len(train_data):].shape)  # 测试集部分，(1459, 79)
print(all_features.iloc[:len(train_data)].iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 4.10.4 数据预处理
# 代码功能测试
# 获取每列的数据类型
print(all_features.dtypes)
# 创建布尔掩码：True表示数值列，False表示非数值列
mask = all_features.dtypes != 'object'
print(mask)
# all_features.dtypes[mask]使用布尔掩码筛选数值列的数据类型。.index：获取这些数值列的列名
numeric_dtypes = all_features.dtypes[mask].index
print(numeric_dtypes)
"""
# 1. 选择数值列子集
selected_data = all_features[numeric_features]
# 2. 对每列应用标准化函数
transformed_data = selected_data.apply(lambda x: (x - x.mean()) / (x.std()))
# 3. 将结果赋值回原DataFrame
all_features[numeric_features] = transformed_data

apply() 对DataFrame的每一列（axis=0默认）应用函数，即每一个特征
对于 DataFrame.apply(func):
- 每次传入一列数据（Series对象）
- 对每列独立计算
- 返回转换后的列

特征标准化是按特征（列）进行的，因为：
- 每个特征有自己的量纲和分布
- 我们需要统一所有特征的尺度
- 目标：让不同特征对模型的贡献权重相当
如果按行标准化（错误！）：
- 会破坏特征之间的相对关系
- 让模型无法学习有意义的模式
"""

# 数据预处理
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 这里的 x 是每一列（每个特征）
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
"""
pd.get_dummies() 的作用：
将分类变量（字符串/对象类型）转换为独热编码（One-Hot Encoding）
dummy_na=True 参数：
为缺失值也创建一个独立的虚拟变量列
样本数不变，特征数变多了，把带缺失值的那些特征再按照每个样本的值进行细分，
有几个值存在，那么这个特征就会被再次细分为几个子特征，这样特征数就变多了
"""
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

n_train = train_data.shape[0]
# 检查all_features的类型
print(f"all_features 类型: {type(all_features)}")

# 检查训练部分
train_data1 = all_features[:n_train]
test_data1 = all_features[n_train:]
print(f"训练数据形状: {train_data1.shape}")
print(f"训练数据类型: {type(train_data1)}")
train_features = torch.tensor(all_features[:n_train].values.astype(np.float32), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(np.float32), dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 4.10.5 训练
loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


# 专门为房价预测设计的自定义评估函数。计算对数均方根误差，专门处理房价预测中的数值稳定性问题。
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    """
    下面一行代码的解析：
    # 神经网络对输入特征的原始预测
    raw_predictions = net(features)
    # 可能包含各种值：正数、负数、零，甚至非常大的数

    torch.clamp(输入张量, 最小值, 最大值)
    将所有小于最小值的元素设置为最小值
    将所有大于最大值的元素设置为最大值
    保持中间值不变
    这里，最小值: 1
    最大值: float('inf') 表示正无穷大
    所以限制范围: [1, +∞)
    """
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    # rmse张量的完整结构：rmse 不仅仅包含数值，还包含完整的计算图信息！
    print(rmse)
    # .item() 从单元素张量中提取Python标量值
    return rmse.item()


# 返回每个epoch结束后在整个训练集和测试集上计算的对数RMSE（均方根误差）指标
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    # train_iter = d2l.load_array((train_features, train_labels), batch_size)  # 返回: (特征, 标签) 的批量数据对
    # train_iter = data.DataLoader(train_features, batch_size, shuffle=True)  # 替换错误，因为只提供特征，没有标签！
    # test_iter = data.DataLoader(test_features, batch_size, shuffle=False)   # 替换错误，因为只提供特征，没有标签！
    # 训练数据加载器
    train_dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_features.shape[0]：获取测试集的样本数量
    # torch.zeros(行数, 列数)
    # 测试数据加载器（选择一种方式）
    # 方式1：使用伪标签，标签：全零的伪标签（占位符）
    test_dataset = TensorDataset(test_features, torch.zeros(test_features.shape[0], 1))
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            # 如果自己设计新优化算法，如def my_optimizer_step(model, lr=0.001, weight_decay=0):...，
            # 那么使用自定义优化器（而不是optimizer.step()）
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 4.10.6 K折交叉验证
"""
有助于模型选择和超参数调整。
我们首先需要定义一个函数，在K折交叉验证过程中返回第i折的数据。
具体地说，它选择第i个切片作为验证数据，其余部分作为训练数据。
注意，这并不是处理数据的最有效方法，如果我们的数据集大得多，会有其他解决办法。
"""


# 作用：将数据集分成K份，选择第i份作为验证集，其余K-1份作为训练集
def get_k_fold_data(k, i, X, y):
    assert k > 1
    # // 作用：执行两个数的除法。将结果向下取整到最接近的整数。
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # slice(start, end)：创建切片对象
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            # torch.cat(tensors, dim=0)：沿第0维（行方向）拼接张量，要拼接的数据除拼接维度外，其他维度大小必须相同
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# 在K折交叉验证中训练K次后，返回训练和验证误差的平均值。
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        """
        # train_ls 记录每个epoch后的RMSE，比如：
        train_ls = [0.8, 0.6, 0.5, 0.4, 0.35]  # 5个epoch
        valid_ls = [0.9, 0.7, 0.6, 0.5, 0.45]

        # train_ls[-1] = 0.35  # 最终训练RMSE
        # valid_ls[-1] = 0.45  # 最终验证RMSE
        """
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plt.plot(list(range(1, num_epochs + 1)), train_ls, label='train')
            plt.plot(list(range(1, num_epochs + 1)), valid_ls, label='valid')
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.xlim([1, num_epochs])
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)  # 添加网格线，更美观
            plt.title('训练和验证RMSE学习曲线')
            # 在绘图前设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            plt.tight_layout()  # 自动调整布局
            # plt.show()  # 关键！显示图片
            plt.show(block=False)
            plt.pause(5)  # 显示10秒
            plt.close()  # 关闭图片
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f},' f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# 4.10.7 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, ' f'平均验证log rmse: {float(valid_l):f}')


# 4.10.8 提交Kaggle预测，这段代码没有使用K折交叉训练！
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, test_features, None, num_epochs, lr, weight_decay,
                        batch_size)
    # plt.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    plt.plot(np.arange(1, num_epochs + 1), train_ls)  # 移除了方括号
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.xlim([1, num_epochs])
    plt.yscale('log')
    plt.grid(True, alpha=0.3)  # 添加网格线，更美观
    plt.title('训练和验证RMSE学习曲线')
    # 在绘图前设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.tight_layout()  # 自动调整布局
    # plt.show()  # 关键！显示图片
    plt.show(block=False)
    plt.pause(5)  # 显示10秒
    plt.close()  # 关闭图片
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    # 假设有1459个测试样本，形状可能是: (1459, 1) 或 (1459,)
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    """
    # 步骤1: reshape(1, -1)
    reshaped = preds.reshape(1, -1)  
    # 1：新的行数 = 1行
    # -1：自动计算列数（保持总元素数不变）
    # 形状变为: (1, 1459) - 1行，1459列
    # 步骤2: [0] 取第一行
    first_row = reshaped[0]  
    # 形状变为: (1459,) - 一维数组    
    # 步骤3: pd.Series() 转换为pandas Series
    price_series = pd.Series(first_row)
    # 创建带索引的Series对象   
    # 步骤4: 赋值给test_data
    test_data['SalePrice'] = price_series
    # 将预测房价添加到测试数据中
    """
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
