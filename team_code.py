#!/usr/bin/env python

# 集成Diversify模型到team_code.py

import os
import numpy as np
import torch
import sys
import wfdb
from tqdm import tqdm

# 导入必要的挑战赛辅助函数
from helper_code import *

# 导入我们自己的模型和训练代码
from train_diversify import train_diversify_model,predict_with_diversify
from model_diversify_robust import Diversify
from config import Config
config = Config()

# 训练模型函数
def train_model(data_folder, model_folder, verbose):
    verbose=True
    if verbose:
        print('Finding the Challenge data...')

    # 检查数据文件夹是否存在
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Data folder not found: {data_folder}')

    # 训练Diversify模型
    if verbose:
        print('Training Diversify model...')

    # 调用我们的Diversify训练函数
    train_diversify_model(
        data_folder=data_folder,
        model_folder=model_folder,
        epochs=config.max_rounds,
        verbose=verbose
    )
    if verbose:
        print('Model training completed.')


# 加载训练好的模型
def load_model(model_folder, verbose):
    verbose=True
    if verbose:
        print('Loading model...')

    # 检查路径是否存在
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f'模型文件夹未找到: {model_folder}')

    # 检查是否是具体的模型目录路径
    if os.path.basename(model_folder).startswith('chagas_model_'):
        # 直接使用提供的具体模型目录
        model_path = os.path.join(model_folder, 'diversify_model.pt')
        checkpoint_path = os.path.join(model_folder, 'best_model.pth')
    else:
        # 原有逻辑：查找最新的模型文件夹
        model_dirs = [d for d in os.listdir(model_folder) if
                      d.startswith('chagas_model_') and os.path.isdir(os.path.join(model_folder, d))]
        if not model_dirs:
            raise FileNotFoundError(f'在 {model_folder} 中未找到模型目录')

        # 按时间戳排序，选择最新的
        latest_model_dir = sorted(model_dirs)[-1]
        model_path = os.path.join(os.path.abspath(model_folder), latest_model_dir, 'diversify_model.pt')
        checkpoint_path = os.path.join(os.path.abspath(model_folder), latest_model_dir, 'best_model.pth')

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # 加载模型
    if os.path.exists(model_path):
        # 直接加载保存的模型字典
        model_dict = torch.load(model_path, map_location=device,weights_only=False)
        algorithm = model_dict['algorithm']
    elif os.path.exists(checkpoint_path):
        # 从检查点加载
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        algorithm = Diversify(config)
        algorithm.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f'Model file not found: {model_path} or {checkpoint_path}')

    # 设置为评估模式
    algorithm.eval()
    algorithm.to(device)

    if verbose:
        print('Model loaded successfully.')

    return algorithm


# 运行模型进行预测
def run_model(record, algorithm, verbose):

    signal, metadata = extract_features_for_diversify(record)

    # 转换为张量
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    else:
        signal = signal.float()  # 确保是float类型

    if not isinstance(metadata, torch.Tensor):
        metadata = torch.tensor(metadata, dtype=torch.float32)
    else:
        metadata = metadata.float()

    # if verbose:
    #     print(f'张量转换完成，信号形状: {signal.shape}，元数据形状: {metadata.shape}')

    # 检查模型是否有forward方法
    if not hasattr(algorithm, 'forward') and verbose:
        print('警告: 模型没有forward方法')

    # 尝试直接调用
    try:
        # 预测
        binary_output, probability_output = predict_with_diversify(algorithm, signal, metadata)

        # if verbose:
        #     print(f'预测成功完成，二元输出: {binary_output}, 概率: {probability_output}')

        # 转换输出格式以符合挑战赛要求
        if isinstance(binary_output, np.ndarray) and len(binary_output) == 1:
            binary_output = binary_output[0]
        if isinstance(probability_output, np.ndarray) and len(probability_output) == 1:
            probability_output = probability_output[0]

        # 确保二元输出是0或1
        binary_output = int(binary_output)

        return binary_output, probability_output

    except Exception as e:
        # 如果直接调用失败，尝试直接使用一个简单的应急预测函数
        if verbose:
            print(f'预测过程中出错: {e}')
            import traceback
            traceback.print_exc()

        # 应急预测 - 返回默认值
        return 0, 0.5


# def predict_with_diversify(algorithm, signal, metadata):
#     """带详细调试信息的预测函数"""
#     try:
#         device = torch.device(config.device if torch.cuda.is_available() else "cpu")
#         print(f"预测设备: {device}")
#
#         # 确保算法是在评估模式
#         algorithm.eval()
#
#         # 确保信号是正确的形状
#         if not isinstance(signal, torch.Tensor):
#             signal = torch.tensor(signal, dtype=torch.float32)
#         if signal.dim() == 2:  # [channels, sequence_length]
#             signal = signal.unsqueeze(0)  # 添加批次维度 [1, channels, sequence_length]
#
#         # 确保元数据有正确的形状
#         if not isinstance(metadata, torch.Tensor):
#             metadata = torch.tensor(metadata, dtype=torch.float32)
#         if metadata.dim() == 1:
#             metadata = metadata.unsqueeze(0)  # 添加批次维度
#
#         print(f"预测前 - 信号形状: {signal.shape}, 元数据形状: {metadata.shape}")
#
#         # 将数据移动到正确的设备
#         signal = signal.to(device)
#         metadata = metadata.to(device)
#
#         print(f"设备转移后 - 信号设备: {signal.device}, 元数据设备: {metadata.device}")
#
#         # 检查algorithm的设备
#         for param in algorithm.parameters():
#             print(f"模型参数设备: {param.device}")
#             break
#
#         # 尝试多种方法调用模型
#         with torch.no_grad():
#             try:
#                 # 首先尝试通过__call__方法调用（标准方式）
#                 print("尝试通过__call__方法调用模型...")
#                 _, logits = algorithm(signal, metadata)
#                 print("__call__方法成功!")
#             except Exception as e1:
#                 print(f"通过__call__调用失败: {e1}")
#
#                 try:
#                     # 尝试使用显式forward方法
#                     print("尝试通过forward方法调用模型...")
#                     _, logits = algorithm.forward(signal, metadata)
#                     print("forward方法成功!")
#                 except Exception as e2:
#                     print(f"通过forward调用失败: {e2}")
#
#                     # 最后尝试不使用元数据
#                     print("尝试不使用元数据...")
#                     try:
#                         _, logits = algorithm(signal, None)
#                         print("不使用元数据成功!")
#                     except Exception as e3:
#                         print(f"不使用元数据也失败: {e3}")
#                         # 创建一个应急的logits
#                         print("创建应急logits...")
#                         logits = torch.tensor([[0.0, 0.0]], device=device)
#
#             # 处理输出
#             print(f"生成预测的logits形状: {logits.shape}")
#             probs = torch.softmax(logits, dim=1)
#             print(f"softmax后的概率: {probs}")
#
#             probability = probs[:, 1].cpu().numpy()[0]  # 获取正类的概率
#             predicted = 1 if probability > 0.5 else 0
#
#             print(f"最终预测: {predicted}, 概率: {probability:.4f}")
#
#         return predicted, probability
#
#     except Exception as e:
#         print(f"predict_with_diversify中出现未捕获的异常: {e}")
#         import traceback
#         traceback.print_exc()
#         # 出错时返回默认值
#         return 0, 0.5



# 为Diversify模型提取特征
def extract_features_for_diversify(record):
    try:
        # 尝试加载信号
        signal, fields = load_signals(record)

        # 同时加载元数据
        header = load_header(record)
        age = get_age(header)
        if np.isnan(age):
            age = 50.0  # 使用默认值

        sex = get_sex(header)
        sex_encoded = 1.0 if sex == 'Male' else 0.0 if sex == 'Female' else 0.5

        # 将元数据组合成一个数组
        metadata = np.array([age, sex_encoded], dtype=np.float32)

    except Exception as e:
        try:
            record_data = wfdb.rdrecord(record)
            signal = record_data.p_signal
            metadata = np.array([0.5, 0.5], dtype=np.float32)  # 默认归一化年龄和性别

        except Exception as e2:
            print(f"加载记录 {record} 时出错: {e2}")
            signal = np.zeros((4096, 12), dtype=np.float32)
            metadata = np.array([50.0, 0.5], dtype=np.float32)  # 默认值

    # 处理信号
    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, axis=1)  # 如果是一维数据，扩展维度以匹配12导联格式
    signal_12leads = signal

    # 处理信号长度
    max_length = 4096
    if signal_12leads.shape[0] > max_length:
        start = (signal_12leads.shape[0] - max_length) // 2
        signal_12leads = signal_12leads[start:start + max_length, :]
    elif signal_12leads.shape[0] < max_length:
        padding = np.zeros((max_length - signal_12leads.shape[0], 12), dtype=signal_12leads.dtype)
        signal_12leads = np.concatenate([signal_12leads, padding], axis=0)

    # 对每个导联进行标准化
    for lead in range(12):
        lead_signal = signal_12leads[:, lead]
        mean = np.mean(lead_signal)
        std = np.std(lead_signal)
        if std < 1e-6:
            std = 1.0  # 避免除以零
        signal_12leads[:, lead] = (lead_signal - mean) / std

    # 处理NaN和异常值
    signal_12leads = np.nan_to_num(signal_12leads, nan=0.0, posinf=0.0, neginf=0.0)
    signal_12leads = np.clip(signal_12leads, -5.0, 5.0)
    signal_12leads = signal_12leads.transpose(1, 0)
    # 转置为[12, seq_len]形状，与PyTorch卷积层期望的输入兼容

    # print(f"Final signal shape: {signal_12leads.shape}")  # 应为(12, 4096)
    return signal_12leads, metadata