import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import numpy as np


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=1)
        return torch.sum(x * attention_weights, dim=1)  # [batch_size, input_dim]


class XGBoostCardinalityEstimator(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
        super(XGBoostCardinalityEstimator, self).__init__()
        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
        self.hid_units = hid_units
        
        # 样本特征提取
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.sample_attention = AttentionLayer(hid_units)
        
        # 谓词特征提取
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_attention = AttentionLayer(hid_units)
        
        # 连接特征提取
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_attention = AttentionLayer(hid_units)
        
        # 特征融合层
        self.fusion_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.fusion_mlp2 = nn.Linear(hid_units, hid_units)
        self.fusion_mlp3 = nn.Linear(hid_units, hid_units // 2)
        
        # 残差连接
        self.residual_mlp = nn.Linear(hid_units * 3, hid_units // 2)
        
        # XGBoost模型
        self.xgb_model = None
        self.is_trained = False
        
    def _extract_features(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        batch_size = samples.shape[0]
        
        # 样本特征提取
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = self.sample_attention(hid_sample, sample_mask)

        # 谓词特征提取
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = self.predicate_attention(hid_predicate, predicate_mask)

        # 连接特征提取
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = self.join_attention(hid_join, join_mask)

        # 特征融合
        features = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        
        # 主路径
        main_path = F.relu(self.fusion_mlp1(features))
        main_path = F.relu(self.fusion_mlp2(main_path))
        main_path = self.fusion_mlp3(main_path)
        
        # 残差路径
        residual = self.residual_mlp(features)
        
        # 合并主路径和残差路径
        features = main_path + residual
        
        return features

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        features = self._extract_features(samples, predicates, joins, sample_mask, predicate_mask, join_mask)
        
        if not self.is_trained:
            return torch.sigmoid(torch.randn(features.shape[0], 1, device=features.device))
        
        features_np = features.detach().cpu().numpy()
        dtest = xgb.DMatrix(features_np)
        preds = self.xgb_model.predict(dtest)
        preds_tensor = torch.from_numpy(preds).float().to(features.device)
        return torch.sigmoid(preds_tensor.unsqueeze(1))

    def train_xgb(self, train_loader, num_epochs=10):
        """训练XGBoost模型"""
        all_features = []
        all_labels = []
        
        # 收集训练数据
        for batch in train_loader:
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = batch
            features = self._extract_features(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            all_features.append(features.detach().cpu().numpy())
            all_labels.append(targets.detach().cpu().numpy())
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        # 配置XGBoost参数
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 7,                # 增加深度
            'learning_rate': 0.03,         # 降低学习率
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,         # 减小最小子节点权重
            'gamma': 0.1,                  # 减小分裂阈值
            'reg_alpha': 0.3,              # 减小L1正则化
            'reg_lambda': 1.0,             # 减小L2正则化
            'tree_method': 'hist',
            'eval_metric': ['rmse', 'mae'],
            'max_bin': 256,               # 增加分箱数
            'grow_policy': 'lossguide'    # 使用基于损失的生长策略
        }
        
        # 创建验证集
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 早停设置
        early_stopping_rounds = 50
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        # 训练XGBoost模型
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_epochs * 100,  # 增加训练轮数
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )
        
        self.is_trained = True 