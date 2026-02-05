import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score


def accuracy(output, target, label_scale):
    """
    回归任务的统一 accuracy
    将 SBP/DBP 的 MAE 平均作为 single accuracy 指标
    """
    with torch.no_grad():
        # 分离 SBP 和 DBP
        pred_sbp = output[:, 0]
        pred_dbp = output[:, 1]
        true_sbp = target[:, 0]
        true_dbp = target[:, 1]

        # 定义损失函数
        mae_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()

        # 计算 MAE
        mae_sbp = mae_loss(pred_sbp, true_sbp) * (label_scale[0] - label_scale[1])
        mae_dbp = mae_loss(pred_dbp, true_dbp) * (label_scale[2] - label_scale[3])

        # 计算 MSE
        mse_sbp = mse_loss(pred_sbp, true_sbp) * (label_scale[0] - label_scale[1])**2
        mse_dbp = mse_loss(pred_dbp, true_dbp) * (label_scale[2] - label_scale[3])**2

        # 统一 accuracy 指标
        mae_avg = (mae_sbp + mae_dbp) / 2
        mse_avg = (mse_sbp + mse_dbp) / 2

        # 可以只返回平均 MAE 或 MSE，或者两者组合
        # 这里返回 average MAE 作为 accuracy
        return mae_avg.item()

def sbp_mae(output, target, label_scale):
    with torch.no_grad():
        pred_sbp = output[:, 0]
        true_sbp = target[:, 0]
        mae_loss = nn.L1Loss()
        mae_sbp = mae_loss(pred_sbp, true_sbp)
        # 反归一化
        return mae_sbp * (label_scale[0] - label_scale[1])

def dbp_mae(output, target, label_scale):
    with torch.no_grad():
        pred_dbp = output[:, 1]
        true_dbp = target[:, 1]
        mae_loss = nn.L1Loss()
        mae_dbp = mae_loss(pred_dbp, true_dbp)
        return mae_dbp * (label_scale[2] - label_scale[3])

def sbp_mse(output, target, label_scale):
    with torch.no_grad():
        pred_sbp = output[:, 0]
        true_sbp = target[:, 0]
        mse_loss = nn.MSELoss()
        mse_sbp = mse_loss(pred_sbp, true_sbp)
        return mse_sbp * (label_scale[0] - label_scale[1])**2

def dbp_mse(output, target, label_scale):
    with torch.no_grad():
        pred_dbp = output[:, 1]
        true_dbp = target[:, 1]
        mse_loss = nn.MSELoss()
        mse_dbp = mse_loss(pred_dbp, true_dbp)
        return mse_dbp * (label_scale[2] - label_scale[3])**2


def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')
