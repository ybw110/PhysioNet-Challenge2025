# train_diversify.py - 修改版

import numpy as np
import random
import time
import torch
import json
import os
import pandas as pd
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
from model_diversify_robust import Diversify
from config import Config
from dataloader import ChagasECGDataLoader
from helper_code import compute_challenge_score, compute_auc, compute_accuracy, compute_f_measure
# 使用 Config 类实例化配置
config = Config()


def set_seed(seed_value=42):
    """设置所有随机种子以确保实验的可重复性。"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_metrics(metrics, save_dir, filename):
    """保存训练指标到文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)
    df = pd.DataFrame(metrics)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def evaluate_predictions(y_true, y_pred_prob, y_pred_binary=None):
    if y_pred_binary is None:
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
    # 计算指标
    challenge_score = compute_challenge_score(y_true, y_pred_prob)
    auroc, auprc = compute_auc(y_true, y_pred_prob)
    accuracy = compute_accuracy(y_true, y_pred_binary)
    f_measure = compute_f_measure(y_true, y_pred_binary)

    return {
        'challenge_score': challenge_score,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f_measure': f_measure
    }


def train_diversify_model(data_folder, model_folder, epochs=config.max_rounds, verbose=True):
    # 设置随机种子
    set_seed(config.seed)

    if verbose:
        print("Starting Diversify model training...")

    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 创建带时间戳的模型保存目录
    model_save_dir = os.path.join(model_folder, f'chagas_model_{timestamp}')
    os.makedirs(model_save_dir, exist_ok=True)
    # 创建日志文件夹
    log_dir = os.path.join(model_save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 创建指标记录字典
    metrics = {
        'round': [],
        'epoch': [],
        'phase': [],
        'class_loss': [],
        'dis_loss': [],
        'ent_loss': [],
        'total_loss': [],
        'train_acc': [],
        'val_auc': [],
        'val_auprc': [],
        'time_elapsed': [],
        # 添加挑战赛指标
        'challenge_score': [],
        'auroc': [],
        'auprc': [],
        'accuracy': [],
        'f_measure': []
    }
    # 设置设备
    # 然后在model_diversify_robust.py中
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using {device} device")

        # 动态查找数据子目录
        data_folders = []
        if os.path.exists(data_folder):
            # 查找所有子目录
            subdirs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]

            # 添加找到的所有子目录
            for subdir in subdirs:
                full_path = os.path.join(data_folder, subdir)

                # 特殊处理PTB-XL-500目录
                if 'ptb-xl' in subdir.lower():
                    data_folders.append(full_path)
                    if verbose:
                        print(f"找到数据子目录: {full_path}")
                    continue

                # 检查子目录是否包含数据文件
                has_hea_files = any(fname.endswith('.hea') for fname in os.listdir(full_path))

                # 如果没有直接的.hea文件，检查是否有包含.hea文件的子目录
                if not has_hea_files:
                    for sub_subdir in [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]:
                        sub_path = os.path.join(full_path, sub_subdir)
                        if any(fname.endswith('.hea') for fname in os.listdir(sub_path)):
                            has_hea_files = True
                            break

                if has_hea_files:
                    data_folders.append(full_path)
                    if verbose:
                        print(f"找到数据子目录: {full_path}")

            if not data_folders:
                # 如果没有找到有效的子目录，直接使用传入的数据文件夹
                data_folders = [data_folder]
                if verbose:
                    print(f"未找到子目录，直接使用数据文件夹: {data_folder}")

        # 设置不同数据源的权重 - 根据目录名称识别数据来源
        source_weights = {}
        for folder in data_folders:
            folder_name = os.path.basename(folder).lower()
            if 'code15' in folder_name:
                source_weights[folder] = 0.8  # CODE-15%弱标签权重低
            elif 'samitrop' in folder_name:
                source_weights[folder] = 1.5  # SaMi-Trop强标签权重高
            elif 'ptb' in folder_name:
                source_weights[folder] = 1.0  # PTB-XL正常权重
            else:
                source_weights[folder] = 1.0  # 未知来源使用默认权重

        # 设置采样率转换信息
        resample_rate = {}
        for folder in data_folders:
            folder_name = os.path.basename(folder).lower()
            if 'ptb' in folder_name:
                resample_rate[folder] = 500  # PTB-XL是500Hz，需要转换到400Hz

        if verbose:
            print(f"共找到 {len(data_folders)} 个数据目录:")
            for folder in data_folders:
                print(f"  - {folder} (权重: {source_weights.get(folder, 1.0)})")

    # 准备数据加载器
    loader = ChagasECGDataLoader(
        data_folders=data_folders,
        batch_size=config.batch_size,
        max_length=4096,
        num_workers=config.num_workers,
        pin_memory=True,
        seed=config.seed,
        source_weights=source_weights,
        resample_rate=resample_rate
    )

    # 获取训练/验证拆分
    train_loader, val_loader, train_dataset, val_dataset = loader.get_train_val_split(
        val_size=0.3,
        balance=True  # 平衡类别
    )

    if verbose:
        print(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    # 初始化算法
    algorithm = Diversify(config).to(device)
    algorithm.train()

    # 初始化优化器
    params_adv = [
        # {'params': algorithm.dbottleneck.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.dclassifier.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.ddiscriminator.parameters(), 'lr': config.lr_decay2 * config.lr}
    ]
    optimizer_adv = optim.Adam(params_adv, lr=config.lr, weight_decay=config.weight_decay,
                               betas=(config.beta1, 0.99))

    params_cls = [
        # {'params': algorithm.bottleneck.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.classifier.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.discriminator.parameters(), 'lr': config.lr_decay2 * config.lr}
    ]
    optimizer_cls = optim.Adam(params_cls, lr=config.lr, weight_decay=config.weight_decay,
                               betas=(config.beta1, 0.99))

    params_all = [
        {'params': algorithm.featurizer.parameters(), 'lr': config.lr_decay1 * config.lr},
        # {'params': algorithm.abottleneck.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.aclassifier.parameters(), 'lr': config.lr_decay2 * config.lr}
    ]
    optimizer_all = optim.Adam(params_all, lr=config.lr, weight_decay=config.weight_decay,
                               betas=(config.beta1, 0.99))

    # 用于早停的变量
    best_challenge_score = 0
    patience_counter = 0
    patience = 20

    # 训练轮数循环
    for round_idx in range(epochs):
        if verbose:
            print(f"\n====== ROUND {round_idx + 1}/{config.max_rounds} ======")

        # 1. 特征更新
        if verbose:
            print("Stage 1: Feature Update")

        for epoch in range(config.local_epochs):
            start_time = time.time()
            epoch_losses = []

            for data in tqdm(train_loader, desc=f"Feature Update (Epoch {epoch + 1}/{config.local_epochs})",
                             disable=not verbose):
                loss_dict = algorithm.update_a(data, optimizer_all)
                epoch_losses.append(loss_dict['class'])

            avg_class_loss = np.mean(epoch_losses)
            time_elapsed = time.time() - start_time

            if verbose:
                print(f"Epoch {epoch + 1}/{config.local_epochs}, Class Loss: {avg_class_loss:.4f}, "
                      f"Time: {time_elapsed:.2f}s")

            # 记录指标
            metrics['round'].append(round_idx + 1)
            metrics['epoch'].append(epoch + 1)
            metrics['phase'].append('feature_update')
            metrics['class_loss'].append(avg_class_loss)
            metrics['dis_loss'].append(None)
            metrics['ent_loss'].append(None)
            metrics['total_loss'].append(None)
            metrics['train_acc'].append(None)
            metrics['val_auc'].append(None)
            metrics['val_auprc'].append(None)
            metrics['time_elapsed'].append(time_elapsed)
            # 添加挑战赛指标（这一阶段不评估）
            metrics['challenge_score'].append(None)
            metrics['auroc'].append(None)
            metrics['auprc'].append(None)
            metrics['accuracy'].append(None)
            metrics['f_measure'].append(None)
            # 每个epoch保存一次指标
            # save_metrics(metrics, log_dir, f'metrics_{timestamp}.csv')

        # 2. 域标签更新 - 与源域标签直接使用相关的阶段
        if verbose:
            print("Stage 2: Latent Domain Characterization")

        for epoch in range(config.local_epochs):
            start_time = time.time()
            epoch_total_losses = []
            epoch_dis_losses = []
            epoch_ent_losses = []

            for data in tqdm(train_loader, desc=f"Domain Characterization (Epoch {epoch + 1}/{config.local_epochs})",
                             disable=not verbose):
                loss_dict = algorithm.update_d(data, optimizer_adv)
                epoch_total_losses.append(loss_dict['total'])
                epoch_dis_losses.append(loss_dict['dis'])
                epoch_ent_losses.append(loss_dict['ent'])

            avg_total_loss = np.mean(epoch_total_losses)
            avg_dis_loss = np.mean(epoch_dis_losses)
            avg_ent_loss = np.mean(epoch_ent_losses)
            time_elapsed = time.time() - start_time

            if verbose:
                print(f"Epoch {epoch + 1}/{config.local_epochs}, " +
                      f"Total Loss: {avg_total_loss:.4f}, " +
                      f"Disc Loss: {avg_dis_loss:.4f}, " +
                      f"Ent Loss: {avg_ent_loss:.4f}, " +
                      f"Time: {time_elapsed:.2f}s")

            # 记录指标
            metrics['round'].append(round_idx + 1)
            metrics['epoch'].append(epoch + 1)
            metrics['phase'].append('domain_characterization')
            metrics['class_loss'].append(None)
            metrics['dis_loss'].append(avg_dis_loss)
            metrics['ent_loss'].append(avg_ent_loss)
            metrics['total_loss'].append(avg_total_loss)
            metrics['train_acc'].append(None)
            metrics['val_auc'].append(None)
            metrics['val_auprc'].append(None)
            metrics['time_elapsed'].append(time_elapsed)
            # 添加挑战赛指标（这一阶段不评估）
            metrics['challenge_score'].append(None)
            metrics['auroc'].append(None)
            metrics['auprc'].append(None)
            metrics['accuracy'].append(None)
            metrics['f_measure'].append(None)
            # 每个epoch保存一次指标
            # save_metrics(metrics, log_dir, f'metrics_{timestamp}.csv')


        # 3. 域不变特征学习
        if verbose:
            print("Stage 3: Domain-invariant Feature Learning")

        for epoch in range(config.local_epochs):
            start_time = time.time()
            epoch_total_losses = []
            epoch_class_losses = []
            epoch_dis_losses = []

            for data in tqdm(train_loader, desc=f"Domain-invariant Learning (Epoch {epoch + 1}/{config.local_epochs})",
                             disable=not verbose):
                loss_dict = algorithm.update(data, optimizer_cls)
                epoch_total_losses.append(loss_dict['total'])
                epoch_class_losses.append(loss_dict['class'])
                epoch_dis_losses.append(loss_dict['dis'])

            # 计算训练指标
            avg_total_loss = np.mean(epoch_total_losses)
            avg_class_loss = np.mean(epoch_class_losses)
            avg_dis_loss = np.mean(epoch_dis_losses)

            # 评估验证集
            algorithm.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for data in tqdm(val_loader, desc="Validating", disable=not verbose):
                    signals = data[0].to(device).float()
                    metadata = data[1].to(device).float()
                    labels = data[2].to(device).long()

                    # 前向传播 - 使用元数据
                    _, logits = algorithm(signals, metadata)
                    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

                    val_preds.extend(probs)
                    val_labels.extend(labels.cpu().numpy())

            # 计算验证指标
            val_labels = np.array(val_labels)
            val_preds = np.array(val_preds)
            val_binary_preds = (val_preds > 0.5).astype(int)

            # 使用官方评估指标
            eval_metrics = evaluate_predictions(val_labels, val_preds, val_binary_preds)
            challenge_score = eval_metrics['challenge_score']
            auroc = eval_metrics['auroc']
            auprc = eval_metrics['auprc']
            accuracy = eval_metrics['accuracy']
            f_measure = eval_metrics['f_measure']

            # 计算训练集准确率
            algorithm.eval()
            train_correct = 0
            train_total = 0

            with torch.no_grad():
                for data in train_loader:
                    signals = data[0].to(device).float()
                    metadata = data[1].to(device).float()
                    labels = data[2].to(device).long()
                    # 前向传播
                    # 前向传播 - 使用元数据
                    _, logits = algorithm(signals, metadata)
                    _, predicted = torch.max(logits.data, 1)

                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

            train_acc = train_correct / train_total if train_total > 0 else 0

            # 切换回训练模式
            algorithm.train()

            time_elapsed = time.time() - start_time

            if verbose:
                print(f"Epoch {epoch + 1}/{config.local_epochs}, " +
                      f"Total Loss: {avg_total_loss:.4f}, " +
                      f"Class Loss: {avg_class_loss:.4f}, " +
                      f"Disc Loss: {avg_dis_loss:.4f}, " +
                      f"Train Acc: {train_acc:.4f}, " +
                      f"Challenge Score: {challenge_score:.4f}, " +
                      f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, " +
                      f"Accuracy: {accuracy:.4f}, F-measure: {f_measure:.4f}, " +
                      f"Time: {time_elapsed:.2f}s")
            # 记录指标
            metrics['round'].append(round_idx + 1)
            metrics['epoch'].append(epoch + 1)
            metrics['phase'].append('domain_invariant')
            metrics['class_loss'].append(avg_class_loss)
            metrics['dis_loss'].append(avg_dis_loss)
            metrics['ent_loss'].append(None)
            metrics['total_loss'].append(avg_total_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_auc'].append(auroc)
            metrics['val_auprc'].append(auprc)
            metrics['time_elapsed'].append(time_elapsed)
            # 添加挑战赛指标
            metrics['challenge_score'].append(challenge_score)
            metrics['auroc'].append(auroc)
            metrics['auprc'].append(auprc)
            metrics['accuracy'].append(accuracy)
            metrics['f_measure'].append(f_measure)

            # 每个epoch保存一次指标
            # save_metrics(metrics, log_dir, f'metrics_{timestamp}.csv')

            # 早停检查 - 使用challenge_score作为主要指标
            if challenge_score > best_challenge_score:
                best_challenge_score = challenge_score
                # 保存最佳模型
                torch.save(algorithm.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
                # 保存检查点包含更多信息
                checkpoint = {
                    'model_state_dict': algorithm.state_dict(),
                    'optimizer_all_state_dict': optimizer_all.state_dict(),
                    'optimizer_adv_state_dict': optimizer_adv.state_dict(),
                    'optimizer_cls_state_dict': optimizer_cls.state_dict(),
                    'round': round_idx + 1,
                    'epoch': epoch + 1,
                    'best_challenge_score': best_challenge_score,
                    'auroc': auroc,
                    'auprc': auprc,
                    'accuracy': accuracy,
                    'f_measure': f_measure,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(model_save_dir, 'best_checkpoint.pth'))
                patience_counter = 0

                if verbose:
                    print(f"新的最佳模型已保存！Challenge分数: {challenge_score:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at round {round_idx + 1}, epoch {epoch + 1}")
                    break

            # 如果已经触发早停，跳出轮次循环
        if patience_counter >= patience:
            break

        # 每个round结束保存一次指标
        save_metrics(metrics, log_dir, f'metrics_round_{round_idx + 1}_{timestamp}.csv')

    # 加载最佳模型
    algorithm.load_state_dict(torch.load(os.path.join(model_save_dir, 'best_model.pth')))

    # 保存最终模型(包含所有必要组件)
    model_dict = {
        'algorithm': algorithm,
        'config': config,
        'best_challenge_score': best_challenge_score,
        'timestamp': timestamp
    }

    torch.save(model_dict, os.path.join(model_save_dir, 'diversify_model.pt'))

    # 保存完整指标记录
    save_metrics(metrics, log_dir, f'complete_metrics_{timestamp}.csv')

    # 保存最终结果
    final_results = {
        'best_challenge_score': best_challenge_score,
        'timestamp': timestamp,
        'total_rounds': round_idx + 1,
        'early_stop': patience_counter >= patience
    }

    with open(os.path.join(model_save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    if verbose:
        print(f"训练完成。最佳Challenge分数: {best_challenge_score:.4f}")
        print(f"模型已保存到 {os.path.join(model_folder, 'diversify_model.pt')}")
        print(f"指标已保存到 {os.path.join(model_folder, 'complete_metrics_{timestamp}.csv')}")

    return algorithm



def predict_with_diversify(algorithm, signal, metadata):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    algorithm.eval()

    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    if signal.dim() == 2:
        signal = signal.unsqueeze(0)

    if not isinstance(metadata, torch.Tensor):
        metadata = torch.tensor(metadata, dtype=torch.float32)
    if metadata.dim() == 1:
        metadata = metadata.unsqueeze(0)

    signal = signal.to(device)
    metadata = metadata.to(device)

    with torch.no_grad():
        # 前向传播 - 使用元数据
        _, logits = algorithm(signal, metadata)

        probs = torch.softmax(logits, dim=1)
        probability = probs[:, 1].cpu().numpy()[0]  # 获取正类的概率
        predicted = logits.argmax(dim=1).item()

    return predicted, probability
