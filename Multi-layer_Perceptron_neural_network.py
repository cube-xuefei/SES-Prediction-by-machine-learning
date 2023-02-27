import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import xlwt
'''
多层感知机回归
'''
if __name__ == '__main__':
    fearture_name = ['bank_access','education_access','entertainment_access','food_access','green_access','medical_access','shop_access','sport_access','BATHROOMS','BEDROOMS','LIVINGAREA','PRICE','PGARDEN','YARDS','BASEMENT','TRAFFIC_CO','DECORATION']
    # 1 准备数据
    file_path = './data/input.xlsx'
    df = pd.read_excel(file_path)
    data = df.values[:,1:]
    y = data[:,-1]
    x = data[:,0:-1]

    # 2 分割训练数据和测试数据
    # 随机采样25%作为测试 75%作为训练
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

    # 3 训练数据和测试数据进行标准化处理
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)

    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))

    # 4 回归模型进行训练和预测
    # 多层感知机回归
    rfr = MLPRegressor()
    # 训练
    rfr.fit(x_train, y_train)
    # 预测 保存预测结果
    rfr_y_predict = rfr.predict(x_test).reshape(-1,1)

    # 5 模型评估
    # 多层感知机回归模型评估
    print("多层感知机回归的默认评估值为：", rfr.score(x_test, y_test))
    print("多层感知机回归的R_squared值为：", r2_score(y_test, rfr_y_predict))
    print("多层感知机回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(rfr_y_predict)))
    print("多层感知机回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                                   ss_y.inverse_transform(rfr_y_predict)))
    # 5.5 保存结果
    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    Data = (('默认评估值','R_squared值','均方误差','平均绝对误差'),
            (rfr.score(x_test, y_test),r2_score(y_test, rfr_y_predict)
             ,mean_squared_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(rfr_y_predict)),
             mean_absolute_error(ss_y.inverse_transform(y_test),
                                 ss_y.inverse_transform(rfr_y_predict))
             )
    )
    for i, row in enumerate(Data):
        for j, col in enumerate(row):
            booksheet.write(i, j, col)
            workbook.save('MLPNN.xls')

    # 6 查看模型的特征重要性排序
    result = permutation_importance(rfr, x_test, y_test, scoring='neg_mean_squared_error')
    feature_importance = result.importances_mean
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(fearture_name)[sorted_idx])
    plt.title("Feature Importance (MDI)")
    plt.show()



