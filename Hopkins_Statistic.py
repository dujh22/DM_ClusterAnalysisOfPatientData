import pandas as pd
import numpy as np
from sklearn import preprocessing

# 导入数据
df = pd.read_csv('.\dataset_diabetes\proprecessing_data.csv')
data_set = df.to_numpy().T

# Ordianl变量改为interval-based类型
le = preprocessing.LabelEncoder()    #获取一个LabelEncoder
le = le.fit(["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])      #训练LabelEncoder
data_set[2] = le.transform(data_set[2])                #使用训练好的LabelEncoder对原数据进行编码
scaler = preprocessing.MinMaxScaler() 
data_set[2] = scaler.fit_transform(data_set[2].reshape(-1,1)).reshape(1,-1)

# Interval-based数据归一化
idx = [6, 8, 9, 10, 11, 12, 13, 17]
idx_mix = [14, 15, 16]
scaler = scaler.fit(data_set[idx].T) 
data_set[idx] = scaler.transform(data_set[idx].T).T

for i in idx_mix:
    subdata_set = np.array([str(j) for j in data_set[i]])
    invalid_char = ['V','E','?']
    idx_nan = np.array([])
    for c in invalid_char:
        char_count = np.char.count(subdata_set, c)
        idx_nan = np.append(idx_nan, np.argwhere(char_count != 0))
    idx_nan = np.unique(idx_nan).astype(int)
    subdata_set[idx_nan] = '-1'
    subdata_set = subdata_set.astype(float)
    subdata_set[idx_nan] = 'NaN'
    data_set[i] = scaler.fit_transform(subdata_set.reshape(-1,1)).reshape(1,-1).squeeze()

T = np.array([False, False, True, False, False, False, True, False, True, True, True, True, True,
        True, True, True, True, True, False, False, False, False, False, False, False, False, False, 
        False])
idx_BN = np.argwhere(T == False).squeeze()
idx_OI = np.argwhere(T == True).squeeze()

def BNd(x1, x2):    #T:False
    res = np.ones(x2.shape)
    x1 = np.tile(x1, x2.shape[0]).reshape(-1,x2.shape[1])
    res[x1 == x2] =0
    return res
    
def OId(x1, x2):    #T:True
    return np.nan_to_num(np.abs(x1 - x2), nan= 1)

#计算向量距离函数
def edist(X1,X2):
    X1_BN = X1[idx_BN].astype(str)
    X2_BN = X2[idx_BN].astype(str)
    X1_OI = X1[idx_OI].astype(float)
    X2_OI = X2[idx_OI].astype(float)

    # 计算该属性值是否有效
    delta = np.ones(X2.shape)
    idx = np.argwhere(X1_BN == '?').squeeze()
    delta[idx_BN[idx]] = 0
    idx = np.argwhere(X2_BN == '?').squeeze()
    if len(idx.reshape(-1,1)) == 2:
        delta[idx_BN[idx[0]]][idx[1]] = 0
    else:
        for i in idx:
            delta[idx_BN[i[0]]][i[1]] = 0
    idx = np.argwhere(np.isnan(X1_OI)).squeeze()
    delta[idx_OI[idx]] = 0
    idx = np.argwhere(np.isnan(X2_OI)).squeeze()
    if len(idx.reshape(-1,1)) == 2:
        delta[idx_OI[idx[0]]][idx[1]] = 0
    else:
        for i in idx:
            delta[idx_OI[i[0]]][i[1]] = 0
    dSum = np.sum(delta, axis=0)

    # 计算不同属性间距离
    mSum = np.zeros(X2.shape[1])
    bnd = BNd(X1_BN, X2_BN.T)
    delta_BN = delta[idx_BN].T
    for i in range(X2.shape[1]):
        mSum[i] += np.dot(bnd[i], delta_BN[i])
    oid = OId(X1_OI, X2_OI.T)
    delta_oi = delta[idx_OI].T
    for i in range(X2.shape[1]):
        mSum[i] += np.dot(oid[i], delta_oi[i])

    return mSum * 10 / dSum

# 随机抽取样本计算最近邻距离
n = 100
idx_sample = np.random.randint(0, data_set.shape[1], n)
sum_X, sum_Y = 0, 0
for i in range(n):
    data = data_set.T[idx_sample[i]]
    remind_data = np.delete(np.arange(data_set.shape[1]), idx_sample[i])
    remind_data = data_set.T[remind_data]

    dist = edist(data, remind_data.T)
    sum_X += min(dist)

# 随机生成样本计算最近邻距离
random_set = []
Nominal_dict = [
            ['Caucasian', 'AfricanAmerican', '?', 'Hispanic', 'Other', 'Asian'],
            ['Female', 'Male'],
            [i for i in range(0,10)],
            [1, 2, 3, 5, 6],
            [1, 2, 3, 5, 6, 18, 22, 25],
            [1, 2, 4, 6, 7, 17],
            ['?', 'InternalMedicine', 'Family/GeneralPractice', 'Emergency/Trauma', 'Cardiology', 'Surgery-General', 'Orthopedics', 'Orthopedics-Reconstructive'
                'Radiologist', 'Nephrology'],
            ['None', '>8', 'Norm', '>7'],
            ['No', 'Steady', 'Up', 'Down'],
            ['No', 'Steady', 'Up', 'Down'],
            ['No', 'Steady', 'Up', 'Down'],
            ['No', 'Steady', 'Up', 'Down'],
            ['No', 'Steady'],
            ['No', 'Steady'],
            ['No', 'Steady', 'Up', 'Down'],
            ['No', 'Ch'],
            ['Yes', 'No']
]
T2 = T.copy()
T2[2] = False
cnt = 0
for b in T2:
    if b:
        random_set.append(np.random.random(n))
    else:
        j = np.random.randint(0,len(Nominal_dict[cnt]),(1,n)).squeeze()
        random_set.append(np.array([Nominal_dict[cnt][k] for k in j]))
        cnt += 1
scaler = preprocessing.MinMaxScaler() 
random_set[2] = scaler.fit_transform(random_set[2].reshape(-1,1)).reshape(1,-1).squeeze()
random_set = np.array(random_set).T

for i in range(n):
    data = random_set[i]
    dist = edist(data, data_set)
    sum_Y += min(dist)

# 计算二者比值
print(sum_Y / (sum_X + sum_Y))