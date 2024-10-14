
import streamlit as st
import torch
import numpy as np
import joblib

# 加载标准化器和模型
x_scaler = joblib.load('x_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

# 定义网络结构，和训练时的一样
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, 18)
        self.bn1 = torch.nn.BatchNorm1d(18)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.hidden2 = torch.nn.Linear(18, 10)
        self.bn2 = torch.nn.BatchNorm1d(10)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.hidden3 = torch.nn.Linear(10, 5)
        self.bn3 = torch.nn.BatchNorm1d(5)
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.output = torch.nn.Linear(5, n_output)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout1(out)

        out = self.hidden2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.dropout2(out)

        out = self.hidden3(out)
        out = self.bn3(out)
        out = torch.relu(out)
        out = self.dropout3(out)

        out = self.output(out)
        return out

# 加载模型
model = Net(25, 3)
model.load_state_dict(torch.load('最终模型0801.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit app开始
st.title('烧结矿质量预测')

# 创建用户输入界面
st.write("输入原燃料数据进行预测")

# 使用表单
with st.form("input_form"):
    # 创建5列布局
    cols = st.columns(5)
    input_data = []
    features = [
        "二选", "二选（64）", "二选（62）", "天滋", "翔程", "承德鸿实", "承德天宝", "阜新源盛",
        "WPF", "超特", "混合粉", "南非", "海砂", "智利", "基姆坎", "进口钒钛", "白云石",
        "石灰石", "生石灰", "除尘灰", "钢渣", "高返", "焦粉", "混合料水分", "料层厚度"
    ]
    for i in range(25):
        if features[i] == "料层厚度":
            val = cols[i % 5].number_input(features[i], min_value=0.0, max_value=1000.0, value=600.0, step=1.0)
        else:
            val = cols[i % 5].number_input(features[i], min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        input_data.append(val)

    # 提交按钮
    submitted = st.form_submit_button("预测")

# 如果用户提交了表单
if submitted:
    # 将输入数据转换为numpy数组并标准化
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = x_scaler.transform(input_data)

    # 转换为Tensor
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    # 模型预测
    with st.spinner('正在预测...'):
        with torch.no_grad():
            pred_scaled = model(input_tensor)
            pred = y_scaler.inverse_transform(pred_scaled.numpy())

    # 显示预测结果
    st.write(f"垂直燃烧速度预测值: {pred[0][0]:.3f}")
    st.write(f"转鼓强度预测值: {pred[0][1]:.3f}")
    st.write(f"平均粒级预测值: {pred[0][2]:.3f}")
