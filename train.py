# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd # 确保导入 pandas
from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler # 不再需要，因为预处理已完成
# from scipy.stats.mstats import winsorize # 不再需要，因为预处理已完成
import copy # 用于保存最佳模型
import os # 用于检查文件路径

# --- 配置设定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {DEVICE}") # 打印当前使用的设备 (GPU 或 CPU)

# 模型超参数
INPUT_FEATURES = 13 # <<< 关键: 必须与下面选择的特征列数量一致
SEQ_LENGTH = 80 # 输入序列的时间步长
LSTM_HIDDEN_SIZE = 128 # LSTM 隐藏层大小 (对于双向 LSTM，每个方向的大小)
LSTM_LAYERS = 1 # LSTM 层数
DROPOUT_RATE = 0.2 # Dropout 比率
MLP_SIZES = [LSTM_HIDDEN_SIZE * 2, 256, 64, 64, 1] # MLP 各层大小

# 训练超参数
LEARNING_RATE = 1e-4 # 学习率
BATCH_SIZE = 256    # 批次大小
MAX_EPOCHS = 50 # 最大训练轮数
MIN_EPOCHS = 20 # 最小训练轮数 (用于早停逻辑)
PATIENCE = 10       # 早停耐心值
VALIDATION_SPLIT = 0.15 # 验证集占总样本的比例
SEED = 42           # 随机种子，用于保证结果可复现

# --- 数据文件路径 ---
# <<< 关键: 设置你的 CSV 文件路径 >>>
DATA_FILE_PATH = "D:/data.csv"

# <<< 关键: 定义 CSV 文件中与模型输入对应的特征列名 (假设这些列已预处理) >>>
# 你需要根据你的 data.csv 文件精确地定义这 13 个特征列的名称
feature_columns = [
    'open',         # 示例：开盘价 (假设已预处理)
    'high',         # 示例：最高价 (假设已预处理)
    'low',          # 示例：最低价 (假设已预处理)
    'close',        # 示例：收盘价 (假设已预处理)
    'vol',          # 示例：成交量 (假设已预处理)
    'amount',       # 示例：成交额 (假设已预处理)
    # --- 其他7个特征 (补全你 CSV 中的实际列名) ---
    # 'feature_7',
    # 'feature_8',
    # 'feature_9',
    # 'feature_10',
    # 'feature_11',
    # 'feature_12',
    # 'feature_13',
]
# 断言检查特征列数量是否正确
assert len(feature_columns) == INPUT_FEATURES, \
    f"期望有 {INPUT_FEATURES} 个特征列, 但实际指定了 {len(feature_columns)} 个。请检查 `feature_columns` 列表。"

# 设置随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 数据加载、目标计算和序列创建函数 ---
def load_and_prepare_data(file_path, feature_cols, seq_length, target_shift=5):
    """
    从 CSV 加载数据, 计算目标变量, 并创建时间序列样本。
    假设 feature_cols 指定的列已经是预处理过的。

    参数:
        file_path (str): CSV 文件路径。
        feature_cols (list): 用作输入的特征列名列表 (假设已预处理)。
        seq_length (int): 输入序列的长度 (例如 80)。
        target_shift (int): 计算目标收益率时向前看的天数 (例如 5)。

    返回:
        tuple: (sequences_array, targets_array)
               - sequences_array (np.array): 形状 (样本数, seq_length, 特征数) 的特征序列
               - targets_array (np.array): 形状 (样本数,) 的目标值
    """
    print(f"开始从 {file_path} 加载数据 (假设特征已预处理)...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 数据文件未找到于 '{file_path}'")

    try:
        # 加载数据，初始不解析日期，以便灵活转换
        df = pd.read_csv(file_path, parse_dates=False)
        print("CSV 文件加载完成。")

        # --- 数据清洗和格式化 ---
        # 将 trade_date 列转换为 datetime 对象
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')

        # 检查必需列是否存在 (ts_code, trade_date, open 用于计算目标, 以及所有特征列)
        required_load_cols = ['ts_code', 'trade_date', 'open'] + feature_cols
        missing_cols = [col for col in required_load_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"错误: CSV文件中缺少以下必需列: {missing_cols}")

        # 只保留需要的列
        df = df[['ts_code', 'trade_date', 'open'] + feature_cols].copy()

        # 设置并排序索引，以便高效处理
        df.set_index(['ts_code', 'trade_date'], inplace=True)
        df.sort_index(inplace=True)
        print("数据索引设置并排序完成。")

        # --- 目标变量计算 (Ret_t+5 开盘到开盘收益率) ---
        # !!! 注意: 如果你的 CSV 文件中已经有一个预先计算好的目标列，
        # !!! 你应该修改这部分代码，直接使用那个列，而不是重新计算。
        # !!! 例如: df['target'] = df['your_precalculated_target_column']
        print("正在计算目标变量 (5日远期开盘收益率)...")
        # 按股票代码分组，并将 'open' 价格向前移动 5 天
        df['open_t_plus_5'] = df.groupby(level='ts_code')['open'].shift(-target_shift)

        # 计算收益率: (Open_t+5 / Open_t) - 1
        # 如果 'open' 为 0，则避免除零错误
        df['target'] = np.where(df['open'] != 0, (df['open_t_plus_5'] / df['open']) - 1, 0)

        # 删除无法计算目标值的行 (通常是每支股票最后几天的数据)
        # 同时删除特征中可能存在的 NaN 值
        df.dropna(subset=['target'] + feature_cols, inplace=True)
        print(f"目标变量计算完成。数据量 (移除NaN后): {len(df)} 行。")

        # --- 序列创建 ---
        print(f"正在创建长度为 {seq_length} 的时间序列样本...")
        all_sequences = []
        all_targets = []
        unique_stocks = df.index.get_level_values('ts_code').unique() # 获取所有不同的股票代码

        for i, stock_code in enumerate(unique_stocks):
            if (i + 1) % 100 == 0: # 每处理 100 支股票打印一次进度
                print(f"  处理股票 {i+1}/{len(unique_stocks)}: {stock_code}")

            stock_df = df.loc[stock_code] # 获取当前股票的数据

            # 确保数据点足够创建至少一个序列
            if len(stock_df) < seq_length:
                continue

            # 获取特征数据和目标数据 (NumPy 数组)
            feature_data = stock_df[feature_cols].values # 形状: (股票数据长度, 特征数)
            target_data = stock_df['target'].values     # 形状: (股票数据长度,)

            # 使用 stride_tricks 高效创建滚动窗口 (比循环更快)
            n_features = feature_data.shape[1] # 特征数量
            # 计算输出序列数组的形状
            shape = (len(stock_df) - seq_length + 1, seq_length, n_features)
            # 计算用于创建视图的步长 (strides)
            strides = (feature_data.strides[0], feature_data.strides[0], feature_data.strides[1])
            # 创建序列视图 (不实际复制数据)
            sequences = np.lib.stride_tricks.as_strided(feature_data, shape=shape, strides=strides)

            # 目标值对应于每个序列窗口的 *结束* 时刻
            # 例如，第0个序列(天0到天79)的目标是第79天的目标值
            targets = target_data[seq_length - 1:]

            all_sequences.append(sequences)
            all_targets.append(targets)

        # 如果未能创建任何序列，则引发错误
        if not all_sequences:
             raise ValueError("错误: 未能从数据中创建任何时间序列样本。请检查数据量或 `seq_length` 设置。")

        # 将所有股票的序列和目标连接成一个大的 NumPy 数组
        sequences_array = np.concatenate(all_sequences, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)
        print(f"时间序列样本创建完成。总样本数: {sequences_array.shape[0]}")

        # 确保数据类型为 float32，以匹配 PyTorch 的默认类型
        return sequences_array.astype(np.float32), targets_array.astype(np.float32)

    except FileNotFoundError as e:
        print(e)
        exit() # 退出程序
    except ValueError as e:
        print(e)
        exit() # 退出程序
    except Exception as e:
        print(f"加载和准备数据时发生未知错误: {e}")
        exit() # 退出程序


# --- PyTorch 数据集类 (Dataset) ---
class StockDataset(Dataset):
    """用于封装特征和目标数据的 PyTorch Dataset 类"""
    def __init__(self, features, targets):
        # 将 NumPy 数组转换为 PyTorch 张量
        self.features = torch.tensor(features, dtype=torch.float32)
        # 将目标值转换为 PyTorch 张量，并增加一个维度 (形状变为 [样本数, 1]) 以匹配模型输出
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.targets)

    def __getitem__(self, idx):
        # 根据索引 idx 返回对应的特征张量和目标张量
        return self.features[idx], self.targets[idx]

# --- 模型架构 (Bi-LSTM + MLP) ---
class StockPredictor(nn.Module):
    """股票收益率预测模型 (双向LSTM + MLP)"""
    def __init__(self, input_size, lstm_hidden_size, num_lstm_layers, mlp_sizes, dropout_rate):
        super().__init__() # 调用父类构造函数
        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size=input_size,             # 输入特征数
                            hidden_size=lstm_hidden_size,     # LSTM 隐藏状态维度
                            num_layers=num_lstm_layers,       # LSTM 层数
                            batch_first=True,                 # 输入数据格式 (批次大小, 序列长度, 特征数)
                            bidirectional=True)               # 使用双向 LSTM
        # 定义 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        # 动态构建 MLP (多层感知机)
        mlp_layers = []
        # MLP 的输入层大小等于 Bi-LSTM 的输出维度 (hidden_size * 2)
        current_size = mlp_sizes[0]
        # 遍历定义的 MLP 隐藏层大小
        for size in mlp_sizes[1:-1]:
            mlp_layers.append(nn.Linear(current_size, size)) # 添加线性层
            mlp_layers.append(nn.ReLU())                     # 添加 ReLU 激活函数
            current_size = size # 更新当前层大小
        # 添加 MLP 的输出层 (大小为 1)
        mlp_layers.append(nn.Linear(current_size, mlp_sizes[-1]))
        # 将所有 MLP 层组合成一个 Sequential 模块
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        """模型的前向传播逻辑"""
        # x 形状: (批次大小, 序列长度, 输入特征数)
        # LSTM 前向传播, _ 表示我们不关心隐藏状态和细胞状态的最终值
        lstm_out, _ = self.lstm(x)
        # lstm_out 形状: (批次大小, 序列长度, lstm_hidden_size * 2)
        # 提取最后一个时间步的输出作为序列的表示
        last_time_step_output = lstm_out[:, -1, :] # 形状: (批次大小, lstm_hidden_size * 2)
        # 应用 Dropout
        dropped_out = self.dropout(last_time_step_output)
        # 将 Dropout 后的结果输入 MLP 进行最终预测
        prediction = self.mlp(dropped_out) # 输出形状: (批次大小, 1)
        return prediction

# --- R 平方 (R-squared) 计算函数 ---
def calculate_r2(model, dataloader, device):
    """在给定的 dataloader 上计算模型的 R 平方分数"""
    model.eval() # 设置模型为评估模式
    all_targets = [] # 存储所有真实目标值
    all_predictions = [] # 存储所有模型预测值
    with torch.no_grad(): # 评估时不需要计算梯度
        for features, targets in dataloader:
            # 将数据移动到指定设备
            features, targets = features.to(device), targets.to(device)
            # 获取模型预测
            outputs = model(features)
            # 收集目标和预测（移回 CPU 并转为 NumPy）
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    # 转换为 NumPy 数组并展平
    all_targets = np.array(all_targets).flatten()
    all_predictions = np.array(all_predictions).flatten()
    # 处理目标值方差接近零的情况
    if np.var(all_targets) < 1e-10:
       print("警告: 目标值的方差接近于零。R2 计算可能不稳定。")
       mean_sq_error = np.mean((all_targets - all_predictions)**2)
       # 如果 MSE 也接近零，认为是完美预测
       return 1.0 if mean_sq_error < 1e-10 else 0.0
    # 使用 scikit-learn 计算 R2
    return r2_score(all_targets, all_predictions)

# --- 模型训练函数 ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, max_epochs, min_epochs, patience):
    """训练模型的主函数"""
    best_val_r2 = -float('inf') # 初始化最佳验证集 R2
    epochs_no_improve = 0 # 记录验证集 R2 未改善的连续轮数
    best_model_state = None # 存储最佳模型参数

    print("开始模型训练...")
    for epoch in range(max_epochs): # 遍历训练轮次
        model.train() # 设置模型为训练模式
        running_loss = 0.0 # 累积当前轮次的训练损失
        for i, (features, targets) in enumerate(train_loader): # 遍历训练数据批次
            # 数据移动到设备
            features, targets = features.to(device), targets.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(features)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
            # 每 100 批次打印一次训练信息
            if (i + 1) % 100 == 0:
                print(f'轮次 [{epoch+1}/{max_epochs}], 批次 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}')

        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        # 在验证集上评估模型
        val_r2 = calculate_r2(model, val_loader, device)
        print(f'轮次 [{epoch+1}/{max_epochs}], 训练损失: {epoch_loss:.4f}, 验证集 R2: {val_r2:.4f}')

        # 早停逻辑和最佳模型保存
        if val_r2 > best_val_r2: # 如果当前 R2 更好
            best_val_r2 = val_r2 # 更新最佳 R2
            epochs_no_improve = 0 # 重置计数器
            best_model_state = copy.deepcopy(model.state_dict()) # 保存模型状态
            print(f'发现新的最佳验证集 R2: {best_val_r2:.4f}。正在保存模型状态...')
        else: # 如果没有改善
            epochs_no_improve += 1 # 增加未改善计数
            print(f'验证集 R2 未改善。耐心计数: {epochs_no_improve}/{patience}')

        # 检查是否满足早停条件
        if epochs_no_improve >= patience and (epoch + 1) >= min_epochs:
            print(f'早停条件触发，在 {epoch + 1} 轮后停止训练。')
            break # 退出训练循环

    print('模型训练完成。')
    # 加载最佳模型状态
    if best_model_state:
        print(f"加载具有最佳验证集 R2 ({best_val_r2:.4f}) 的模型状态。")
        model.load_state_dict(best_model_state)
    else:
        print("警告: 未能保存最佳模型状态（可能训练轮数不足或从未改善）。")
    # 返回训练好的模型和最佳验证 R2
    return model, best_val_r2

# --- 主程序执行入口 ---
if __name__ == "__main__":
    # 步骤 1: 加载数据, 计算目标, 创建序列 (假设特征已预处理)
    print("--- 步骤 1: 加载数据, 计算目标, 创建序列 ---")
    # features_processed 形状: (样本数, seq_length, 特征数)
    # targets_processed 形状: (样本数,)
    features_processed, targets_processed = load_and_prepare_data(DATA_FILE_PATH, feature_columns, SEQ_LENGTH)

    # 步骤 2: 创建 PyTorch Dataset 和 DataLoader
    print("\n--- 步骤 2: 创建 Dataset 和 DataLoader ---")
    dataset = StockDataset(features_processed, targets_processed) # 使用加载和处理后的数据创建 Dataset
    dataset_size = len(dataset) # 获取数据集大小
    indices = list(range(dataset_size)) # 创建索引列表
    split = int(np.floor(VALIDATION_SPLIT * dataset_size)) # 计算验证集样本数量
    # 在划分前打乱索引一次
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split] # 划分训练集和验证集索引
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    print(f"训练集样本数: {len(train_indices)}, 验证集样本数: {len(val_indices)}")

    # 步骤 3: 初始化模型、损失函数和优化器
    print("\n--- 步骤 3: 初始化模型, 损失函数, 优化器 ---")
    # 创建模型实例并移动到设备
    model = StockPredictor(input_size=INPUT_FEATURES,
                           lstm_hidden_size=LSTM_HIDDEN_SIZE,
                           num_lstm_layers=LSTM_LAYERS,
                           mlp_sizes=MLP_SIZES,
                           dropout_rate=DROPOUT_RATE).to(DEVICE)
    # 定义损失函数 (均方误差)
    criterion = nn.MSELoss()
    # 定义优化器 (Adam)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 打印模型结构
    print("模型架构:")
    print(model)

    # 步骤 4: 训练模型
    print("\n--- 步骤 4: 开始训练模型 ---")
    # 调用训练函数
    trained_model, best_validation_r2 = train_model(model, train_loader, val_loader,
                                                  criterion, optimizer, DEVICE,
                                                  MAX_EPOCHS, MIN_EPOCHS, PATIENCE)

    # 步骤 5: (可选) 保存最终训练好的模型
    print("\n--- 步骤 5: 保存最终模型 (可选) ---")
    print(f"训练过程中达到的最佳验证集 R2: {best_validation_r2:.4f}")
    # 可以取消下面一行的注释来保存模型参数
    # torch.save(trained_model.state_dict(), "best_stock_predictor_model_preprocessed.pth")
    # print("最佳模型参数已保存。")
    print("脚本执行完毕。")