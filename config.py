# config.py

class Config:
    # 数据集配置

    device = "cuda:0"  # 默认使用cuda:3
    # 模型参数
    num_classes = 2          # 类别数

    bottleneck = 256         # bottleneck层维度
    dis_hidden = 128         # 判别器隐藏层维度

    drop_path = True
    drop_path_rate = 0.2  # 默认丢弃概率

    domain_num = 3    # 潜在域的数量
    batch_size = 64
    local_epochs = 1
    max_rounds = 2

    lr = 5e-5             # 基础学习率
    lr_decay1 = 0.2          # 特征提取器的学习率衰减
    lr_decay2 = 1.0          # 分类器的学习率衰减

    beta1 = 0.9
    weight_decay = 5e-3     # L2正则化系数

#alpha改成alpha2，方便写论文
    alpha1 = 0.3             # 辅助域适应的梯度反转层系数
    alpha = 0.7             # 梯度反转层系数

    lam = 0.3                # 熵损失权重
    algorithm = 'diversify'

    factor = 0.2  # 新的学习率是旧的学习率乘以该因子。
    patience = 5  # 在监测量多少个周期没有改善后调整学习率。

    # 其他配置
    seed = 42
    num_workers = 4
    dropout = 0.2