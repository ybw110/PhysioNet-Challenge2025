# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from model_Adver_network import Discriminator, ReverseLayerF
from model_base import Algorithm
from common_loss import Entropylogits
from config import Config

from model_SECNN import SECNN
# from model_SECNN_aug import SECNN


# 使用 Config 类实例化配置
config = Config()
# 定义计算设备 - 添加这一行
# 然后在model_diversify_robust.py中



class Diversify(Algorithm):

    def __init__(self, config):

        super(Diversify, self).__init__(config)

        # 将所有模型移到指定设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.featurizer = SECNN(include_metadata=True)
        feature_dim = self.featurizer.in_features

        # 第一步,获得细粒度特征
        # self.abottleneck = self._create_bottleneck(feature_dim, config.bottleneck)
        # self.aclassifier = self._create_classifier(config.num_classes * config.domain_num, config.bottleneck)
        # self.discriminator = Discriminator(config.bottleneck, config.dis_hidden, config.domain_num)

        # 直接创建分类器，去掉bottleneck
        self.aclassifier = self._create_classifier(config.num_classes * config.domain_num, feature_dim)
        self.discriminator = Discriminator(feature_dim, config.dis_hidden, config.domain_num)

        # 用于分类任务,
        # self.bottleneck = self._create_bottleneck(feature_dim, config.bottleneck)
        # self.classifier = self._create_classifier(config.num_classes, config.bottleneck)

        # 用于分类任务
        self.classifier = self._create_classifier(config.num_classes, feature_dim)

        # 用于域适应任务
        # self.dbottleneck = self._create_bottleneck(feature_dim, config.bottleneck)
        # self.ddiscriminator = Discriminator(config.bottleneck, config.dis_hidden, config.num_classes)
        # self.dclassifier = self._create_classifier(config.domain_num, config.bottleneck)

        # 用于域适应任务
        self.ddiscriminator = Discriminator(feature_dim, config.dis_hidden, config.num_classes)
        self.dclassifier = self._create_classifier(config.domain_num, feature_dim)

        # 将所有模型移到指定设备
        self.to(self.device)

    # def _create_bottleneck(self, feature_dim, bottleneck_dim):
    #     """创建瓶颈层"""
    #     return nn.Sequential(
    #         nn.Linear(feature_dim, bottleneck_dim),
    #         nn.BatchNorm1d(bottleneck_dim),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(p=config.dropout)
    #     )

    # def _create_classifier(self, class_num,  input_dim):
    #     """创建分类器"""
    #     return nn.Sequential(
    #         nn.Linear( input_dim, 64),
    #         nn.ReLU(),
    #         # nn.Dropout(p=config.dropout),
    #         nn.Linear(64, class_num)
    #     )

    def _create_classifier(self, class_num, input_dim):
        """使用高级激活函数的分类器"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),  # 使用GELU代替ReLU
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, class_num)
        )

 # c) 联合更新(update_a)：
    def update_a(self, minibatches, opt):
        signals = minibatches[0].to(self.device).float()  # 输入数据
        metadata = minibatches[1].to(self.device).float()  # 元数据 (年龄和性别)
        class_labels = minibatches[2].to(self.device).long()  # 类别标签
        domain_labels = minibatches[3].to(self.device).long()  # 域标签

        # 组合领域标签和类别标签
        combined_labels = domain_labels * config.num_classes + class_labels

        # 前向传播 - 使用元数据
        logits, features = self.featurizer(signals, metadata)

        # all_z = self.abottleneck(features)
        # all_preds = self.aclassifier(all_z)
        all_preds = self.aclassifier(features)

        # 计算损失
        classifier_loss = F.cross_entropy(all_preds, combined_labels)

        # 反向传播和优化
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()

        return {'class': classifier_loss.item()}

#a) 域判别器更新(update_d)：
    def update_d(self, minibatch, opt):
        """域判别器更新：更新域分类器"""
        signals = minibatch[0].to(self.device).float()  # 输入特征
        metadata = minibatch[1].to(self.device).float()  # 元数据
        class_labels = minibatch[2].to(self.device).long()  # 类别标签
        domain_labels = minibatch[3].to(self.device).long()  # 域标签

        # 前向传播 - 使用元数据
        _, features = self.featurizer(signals, metadata)
        # z = self.dbottleneck(features)

        # 域判别器
        # disc_in = ReverseLayerF.apply(z, config.alpha1)
        disc_in = ReverseLayerF.apply(features, config.alpha1)

        disc_out = self.ddiscriminator(disc_in)
        disc_loss = F.cross_entropy(disc_out, class_labels)

        # 域分类器
        # cd = self.dclassifier(z)
        cd = self.dclassifier(features)

        # 熵损失 + 交叉熵损失
        ent_loss = Entropylogits(cd) * config.lam + F.cross_entropy(cd, domain_labels)

        # 总损失
        loss = ent_loss + disc_loss

        # 反向传播和优化
        opt.zero_grad()
        loss.backward()
        opt.step()

        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

#b) 分类器更新(update)：
    def update(self, data, opt):
        """分类器更新：更新分类器和特征提取器"""
        signals = data[0].to(self.device).float()  # 输入特征
        metadata = data[1].to(self.device).float()  # 元数据
        class_labels = data[2].to(self.device).long()  # 类别标签
        domain_labels = data[3].to(self.device).long()  # 域标签

        # 前向传播 - 使用元数据
        _, features = self.featurizer(signals, metadata)
        # all_z = self.bottleneck(features)

        # 域判别器（梯度反转）
        # disc_input = ReverseLayerF.apply(all_z, config.alpha)
        disc_input = ReverseLayerF.apply(features, config.alpha)

        disc_out = self.discriminator(disc_input)
        disc_loss = F.cross_entropy(disc_out, domain_labels)

        # 分类器
        # all_preds = self.classifier(all_z)
        all_preds = self.classifier(features)

        classifier_loss = F.cross_entropy(all_preds, class_labels)

        # 总损失
        loss = classifier_loss + disc_loss

        # 反向传播和优化
        opt.zero_grad()
        loss.backward()
        opt.step()

        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x, metadata=None):
        """进行预测"""
        x = x.to(self.device)
        if metadata is not None:
            metadata = metadata.to(self.device).float()

        _, features = self.featurizer(x, metadata)
        # features = self.bottleneck(features)
        return self.classifier(features)

    # predict 函数使用分类器 self.classifier 进行预测
    def predict1(self, x, metadata=None):
        """使用域判别器进行预测"""
        x = x.to(self.device)
        if metadata is not None:
            metadata = metadata.to(self.device).float()

        _, features = self.featurizer(x, metadata)
        # features = self.dbottleneck(features)
        return self.ddiscriminator(features)

    # predict1 函数使用域判别器 self.ddiscriminator 进行预测。
    def forward(self, x, metadata=None):
        """前向传播"""
        x = x.to(self.device)
        if metadata is not None:
            metadata = metadata.to(self.device).float()

        _, features = self.featurizer(x, metadata)
        # features = self.bottleneck(features)
        logits = self.classifier(features)
        return features, logits

