import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# 读取训练数据
train_file_path = 'used_car_train_20200313.csv'
train_df = pd.read_csv(train_file_path, sep='\s+')

# 选择一部分特征
selected_features = ['regDate', 'model', 'brand', 'bodyType', 'fuelType', 'kilometer', 'power']
X = train_df[selected_features]
y = train_df['price']

# 将非数值的缺失值标记转换为NaN
X.replace('-', np.nan, inplace=True)

# 填补缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 将数据集分为训练集和测试集，使用较小的数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# 训练和评估模型
results = []
best_model = None
best_mae = float('inf')
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': mae, 'R²': r2})
    if mae < best_mae:
        best_mae = mae
        best_model = model

# 打印结果
results_df = pd.DataFrame(results)
print(results_df)

# 读取测试数据
test_file_path = 'used_car_testB_20200421.csv'
test_df = pd.read_csv(test_file_path, sep='\s+')

# 选择相同的特征并处理
X_test_final = test_df[selected_features]
X_test_final.replace('-', np.nan, inplace=True)
X_test_final = imputer.transform(X_test_final)
X_test_final_scaled = scaler.transform(X_test_final)

# 使用最佳模型进行预测
y_test_pred = best_model.predict(X_test_final_scaled)

# 生成提交文件
submission = pd.DataFrame({'SaleID': test_df['SaleID'], 'price': y_test_pred})
submission_file_path = 'used_car_sample_submit.csv'
submission.to_csv(submission_file_path, index=False)

print("提交文件已生成：", submission_file_path)
