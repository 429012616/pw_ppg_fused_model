import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse

from data_loader.data_loaders import *  

import model.model_pw as module_arch

from parse_config import ConfigParser
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_for_test(config, checkpoint_path):

    model = config.init_obj('arch', module_arch)
    model = model.to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    
    model.eval()
    return model

def test_model(config, model, test_dataset, label_scale, batch_size=32):
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_pred = []
    all_true = []
    all_residuals = []  
    
    sbp_max, sbp_min, dbp_max, dbp_min = label_scale.tolist()
    
    with torch.no_grad():
        for data0, data1, data2, data3, label_i in test_loader:
            data0 = data0.to(device)
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            label_i = label_i.to(device)
            
            output_norm = model(data0, data1, data2, data3)
            
            sbp_pred = output_norm[:,0] * (sbp_max - sbp_min) + sbp_min
            dbp_pred = output_norm[:,1] * (dbp_max - dbp_min) + dbp_min
            y_pred = torch.stack([sbp_pred, dbp_pred], dim=1)
            
            sbp_true = label_i[:,0] * (sbp_max - sbp_min) + sbp_min
            dbp_true = label_i[:,1] * (dbp_max - dbp_min) + dbp_min
            y_true = torch.stack([sbp_true, dbp_true], dim=1)
            
            residuals = y_pred - y_true  # Residuals
            
            all_pred.append(y_pred.cpu().numpy())
            all_true.append(y_true.cpu().numpy())
    
    all_pred = np.vstack(all_pred)
    all_true = np.vstack(all_true)
    
    return all_pred, all_true #, all_residuals

import numpy as np

def plot_bland_altman(y_true, y_pred, label, save_path):
    mean_vals = np.mean([y_true, y_pred], axis=0)
    diff_vals = (y_pred - y_true)

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(6, 6))
    plt.scatter(mean_vals, diff_vals, color='blue', alpha=0.5)
    plt.axhline(mean_diff, color='gray', linestyle='--')
    plt.axhline(upper_limit, color='red', linestyle='--')
    plt.axhline(lower_limit, color='red', linestyle='--')
    plt.title(f'Bland-Altman Plot for {label}')
    plt.xlabel('Mean of SBP or DBP')
    plt.ylabel('Difference (Predicted - True)')
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    
def plot_regression_line(x, y, title, filename):

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r_squared = model.score(x.reshape(-1, 1), y)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=10, label='Data Points')
    plt.plot(x, y_pred, color='red', label=f'Regression Line (R²={r_squared:.2f})')
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    
if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', default="0",type=str,
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', default="/home/zxy/pw-project/db_data/db_data.npz",type=str,
                      help='Directory containing numpy files')
    
    args2 = args.parse_args()
    fold_id = int(args2.fold_id)
    config = ConfigParser.from_args(args, fold_id, options=[])
    
    checkpoint_path = '/home/zxy/pw-project/saved/Exp1/28_09_2025_03_36_02_fold0/model_best.pth'
    
    model = load_model_for_test(config, checkpoint_path)
    
    train_dataset, test_dataset,label_scale = generate_train_subject(args2.np_data_dir)

    all_pred, all_true = test_model(config, model, test_dataset, label_scale, batch_size=config['data_loader']['args']['batch_size'])
    
    map_predictions = (all_pred[:, 0] + 2 * all_pred[:, 1]) / 3
    map_true_values = (all_true[:, 0] + 2 * all_true[:, 1]) / 3
    
    all_pred = np.insert(all_pred, 1, map_predictions, axis=1)
    all_true = np.insert(all_true, 1, map_true_values, axis=1)
    
    all_residuals = all_pred - all_true
    
    rmse = np.sqrt(np.mean(all_residuals ** 2, axis=0))
    me = np.mean(all_residuals, axis=0)
    mae = np.mean(np.abs(all_residuals), axis=0)
    print("Pred shape:", all_pred.shape, "True shape:", all_true.shape)
    print(f" (RMSE): SBP = {rmse[0]:.2f}, DBP = {rmse[1]:.2f}, MAP = {rmse[2]:.2f}")
    print(f" (ME): SBP = {me[0]:.2f}, DBP = {me[1]:.2f}, MAP = {me[2]:.2f}")
    print(f" (MAE): SBP = {mae[0]:.2f}, DBP = {mae[1]:.2f},MAP = {mae[2]:.2f}")
    # 打印残差
    
    df = pd.DataFrame({
        'SBP_pred': all_pred[:, 0],
        'DBP_pred': all_pred[:, 1],
        'MAP_pred': all_pred[:, 2],
        'SBP_true': all_true[:, 0],
        'DBP_true': all_true[:, 1],
        'MAP_true': all_true[:, 2],
    })
    df.to_excel('/home/zxy/pw-project/model_test/28_09_2025_03_36_02_fold0/all_data.xlsx', index=False)

    plot_bland_altman(all_pred[:,0], all_true[:,0],'SBP','/home/zxy/pw-project/model_test/28_09_2025_03_36_02_fold0/ab_SBP.png')
    plot_bland_altman(all_pred[:,1], all_true[:,1],'DBP','/home/zxy/pw-project/model_test/28_09_2025_03_36_02_fold0/ab_DBP.png')
    plot_bland_altman(all_pred[:,2], all_true[:,2],'MAP','/home/zxy/pw-project/model_test/28_09_2025_03_36_02_fold0/ab_MAP.png')
    plot_regression_line(all_pred[:,0], all_true[:,0], 'SBP', '/home/zxy/pw-project/model_test/28_09_2025_03_36_02_fold0/sbp_error_regression.png')
    plot_regression_line(all_pred[:,1], all_true[:,1], 'DBP', '/home/zxy/pw-project/model_test/28_09_2025_03_36_02_fold0/dbp_error_regression.png')
    plot_regression_line(all_pred[:,2], all_true[:,2], 'MAP', '/home/zxy/pw-project/model_test/28_09_2025_03_36_02_fold0/map_error_regression.png')