# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : PredictPeopleIndexByLightBGM_v1.0.py
# Time       ：2020/4/12 21:24 
# Author     ：haibiyu
# version    ：python 3.6
# Description：按区域、按小时、按天、按周、均值、方差
"""

import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib  # 将模型导出所需包

warnings.filterwarnings('ignore')


def load_data():
    """导入所有数据"""
    df1 = pd.read_csv('./data/area_passenger_index.csv', header=None
                      , names=['ID', 'time', 'people_index'])
    df2 = pd.read_csv('./data/area_passenger_info.csv', header=None
                      ,
                      names=['ID', 'area_name', 'area_type', 'Center_x',
                             'Center_y',
                             'Grid_x', 'Grid_y', 'area'])
    df3 = pd.read_csv('./data/migration_index.csv', header=None
                      , names=['date', 'departure_province', 'departure_city',
                               'arrival_province', 'arrival_city', 'index'])
    df4 = pd.read_csv('./data/datafountain_competition_od.txt', sep='\t',
                      header=None
                      , names=['hour', 'start_grid_x', 'start_grid_y',
                               'end_grid_x',
                               'end_grid_y', 'Index'])
    test_df = pd.read_csv('./data/test_submit_example.csv', header=None
                          , names=['ID', 'time', 'people_index'])
    return df1, df2, df3, df4, test_df


def processing_df1(df1):
    """对df1数据进行处理"""
    df1['time'] = pd.to_datetime(df1['time'], format='%Y%m%d%H')
    df1['day'] = df1['time'].dt.day.apply(
        lambda x: x - 17 if x >= 17 else x + 14)
    # 创建关于时间的衍生变量，如小时、周、是否周末
    df1 = createVariablesByTime(df1)

    dict_wd = dict(df1.groupby(['weekday']).mean()['people_index'])
    dict_h = dict(df1.groupby(['hour']).mean()['people_index'])
    dict_id = dict(df1.groupby(['ID']).mean()['people_index'])
    dict_isweekend = dict(df1.groupby(['is_weekend']).mean()['people_index'])

    df1 = variableToEncoded(df1, dict_isweekend, dict_wd, dict_h, dict_id)

    return df1, dict_wd, dict_h, dict_id, dict_isweekend


def processing_df3(df3):
    """对df3数据进行处理"""
    df3['date'] = pd.to_datetime(df3['date'], format='%Y%m%d')
    # 由于给的数据是1.17-2.15共计30天的数据，所以需要转换为0~29的数字
    df3['day'] = df3['date'].dt.day.apply(
        lambda x: x - 17 if x >= 17 else x + 14)
    df3['weekday'] = df3['date'].dt.weekday  # 数值为0-6，表示一周中的第几天
    # 增加一个标签，表示迁入地是否为北京
    df3['city'] = df3['arrival_city'].apply(lambda x: 1 if x == '北京市' else 0)
    inbj = df3[df3['city'] == 1]  # 表示迁入数据
    outbj = df3[df3['city'] == 0]  # 表示迁出数据
    index_mean_in = inbj.groupby(['day']).mean()['index']
    index_std_in = inbj.groupby(['day']).std()['index']
    index_mean_in_copy = index_mean_in.copy()
    index_mean_in_copy.index = np.arange(30)
    dict_imi = dict(index_mean_in_copy)
    index_std_in_copy = index_std_in.copy()
    index_std_in_copy.index = np.arange(30)
    dict_isi = dict(index_std_in_copy)
    index_mean_out = outbj.groupby(['day']).mean()['index']
    index_std_out = outbj.groupby(['day']).std()['index']
    index_mean_out_copy = index_mean_out.copy()
    index_mean_out_copy.index = np.arange(30)
    dict_imo = dict(index_mean_out_copy)
    index_std_out_copy = index_std_out.copy()
    index_std_out_copy.index = np.arange(30)
    dict_iso = dict(index_std_out_copy)
    # 预测测试数据所需的输入特征值
    ar_model1 = AR(index_mean_in.values).fit()
    pred_index_mean_in = ar_model1.predict(len(index_mean_in),
                                           len(index_mean_in) + 8, dynamic=True)
    ar_model2 = AR(index_std_in.values).fit()
    pred_index_std_in = ar_model2.predict(len(index_std_in),
                                          len(index_std_in) + 8, dynamic=True)
    ar_model3 = AR(index_mean_out.values).fit()
    pred_index_mean_out = ar_model3.predict(len(index_mean_out),
                                            len(index_mean_out) + 8,
                                            dynamic=True)
    ar_model4 = AR(index_std_out.values).fit()
    pred_index_std_out = ar_model4.predict(len(index_std_out),
                                           len(index_std_out) + 8, dynamic=True)

    pred_index_mean_in = pd.Series(pred_index_mean_in,
                                   index=np.arange(30, 39, 1))
    dict_pimi = dict(pred_index_mean_in)
    pred_index_std_in = pd.Series(pred_index_std_in, index=np.arange(30, 39, 1))
    dict_pisi = dict(pred_index_std_in)
    pred_index_mean_out = pd.Series(pred_index_mean_out,
                                    index=np.arange(30, 39, 1))
    dict_pimo = dict(pred_index_mean_out)
    pred_index_std_out = pd.Series(pred_index_std_out,
                                   index=np.arange(30, 39, 1))
    dict_piso = dict(pred_index_std_out)

    return dict_imi, dict_isi, dict_imo, dict_iso, dict_pimi, dict_pisi, dict_pimo, dict_piso


def processing_train_data(train_data, dict_imi, dict_isi, dict_imo, dict_iso):
    """处理训练数据"""
    dict_at = dict(train_data.groupby(['area_type']).mean()['people_index'])
    train_data['area_type_encoded'] = train_data['area_type'].map(
        dict_at)  # 区域类型平均
    train_data.drop(['area_name', 'area_type'], axis=1, inplace=True)
    train_data = createVariableByInOut(train_data, dict_imi, dict_isi, dict_imo,
                                       dict_iso)

    # 对df4进行处理
    dict_hi = dict(df4.groupby(['hour']).mean()['Index'])
    train_data['hi_encoded'] = train_data['hour'].map(dict_hi)  # 网格间流动小时平均
    # 删除训练数据不需要的特征
    train_data.drop(
        ['ID', 'weekday', 'hour', 'is_weekend', 'Center_x', 'Center_y',
         'Grid_x', 'Grid_y', 'time'],
        axis=1,
        inplace=True)
    return train_data, dict_at, dict_hi


def createVariablesByTime(data):
    """增加时间相关的变量"""
    data['weekday'] = data['time'].dt.weekday
    data['hour'] = data['time'].dt.hour
    data['is_weekend'] = data['weekday'].apply(
        lambda x: 1 if (x == 5 or x == 6) else 0)
    return data


def createVariableByInOut(data, dict_imi, dict_isi, dict_imo, dict_iso):
    """将变量中的值转换为每天携入/携出的index均值、方差，作为增加的新变量"""
    data['index_mean_in'] = data['day'].map(dict_imi)  # 迁入平均
    data['index_std_in'] = data['day'].map(dict_isi)  # 迁入方差
    data['index_mean_out'] = data['day'].map(dict_imo)  # 迁出平均
    data['index_std_out'] = data['day'].map(dict_iso)  # 迁出方差
    return data


def variableToEncoded(data, dict_isweekend, dict_wd, dict_h, dict_id):
    """将变量中的值转换为对应分组下的people index均值，作为增加的新变量"""
    data['ID_encoded'] = data['ID'].map(dict_id)  # 区域平均
    data['hour_encoded'] = data['hour'].map(dict_h)  # 小时平均
    data['weekday_encoded'] = data['weekday'].map(dict_wd)  # 周几平均
    data['is_weekend_encoded'] = data['is_weekend'].map(dict_isweekend)  # 是否是周末
    return data


def processing_test_data(test_df, df2, dict_isweekend, dict_wd, dict_h, dict_id,
                         dict_pimi, dict_pisi, dict_pimo, dict_piso, dict_at,
                         dict_hi):
    """处理测试集数据，增加相关的变量"""
    test_data = test_df.copy()
    test_data.drop(['people_index'], axis=1, inplace=True)
    test_data['time'] = pd.to_datetime(test_data['time'], format='%Y%m%d%H')
    test_data['day'] = test_data['time'].dt.day.apply(lambda x: x + 14)
    # 创建关于时间的衍生变量，如小时、周、是否周末
    test_data = createVariablesByTime(test_data)
    # 按id、小时、周、是否周末计算平均的people index
    test_data = variableToEncoded(test_data, dict_isweekend, dict_wd, dict_h,
                                  dict_id)
    # 计算迁入迁出index的均值和方差
    test_data = createVariableByInOut(test_data, dict_pimi, dict_pisi,
                                      dict_pimo, dict_piso)

    test_data = pd.merge(test_data, df2, on=['ID'])
    test_data['hi_encoded'] = test_data['hour'].map(dict_hi)
    test_data['area_type_encoded'] = test_data['area_type'].map(dict_at)

    test_data.drop(
        ['ID', 'time', 'weekday', 'hour', 'is_weekend',
         'day', 'area_name', 'area_type', 'Center_x', 'Center_y', 'Grid_x',
         'Grid_y'],
        axis=1, inplace=True)
    return test_data


def train_model(train_data, target):
    """训练模型"""
    model_lgb = lgb.LGBMRegressor(num_leaves=60
                                  , max_depth=6
                                  , learning_rate=0.16
                                  , n_estimators=2800  # 最大生成树数量
                                  , n_jobs=-1)  # 所有core进行计算
    x_train, x_test, y_train, y_test = train_test_split(train_data, target,
                                                        test_size=0.2)
    model_lgb.fit(x_train, y_train)
    y_hat = model_lgb.predict(x_test)
    y_hat[y_hat < 0] = 0
    # 计算得分，公式为score=1/(rmse+1)
    score = 1 / (mean_squared_error(y_test, y_hat) ** 0.5 + 1)
    print(score)

    return model_lgb


def get_best_params(train_data, target):
    """参数调试"""
    x_train, x_test, y_train, y_test = train_test_split(train_data, target,
                                                        test_size=0.2)
    scores = []
    l = [i * 0.01 for i in range(1, 21)]  # 学习率参数设置
    for i in l:
        model_lgb = lgb.LGBMRegressor(num_leaves=60
                                      , max_depth=6
                                      , learning_rate=0.16
                                      , n_estimators=2800  # 2500  # 最大生成树数量
                                      , n_jobs=-1)  # 所有core进行计算
        model_lgb.fit(x_train, y_train)
        y_hat = model_lgb.predict(x_test)
        y_hat[y_hat < 0] = 0
        score = 1 / (mean_squared_error(y_test, y_hat) ** 0.5 + 1)
        scores.append(score)
        # # 导出模型
        joblib.dump(model_lgb, "./save_model/model_lgb_v1.0")
    print(scores)
    x = np.arange(1500, 3000, 200)
    plt.plot(l, scores)
    plt.show()

def get_best_param1(train_data, target):
    """
    参数调试
    :param train_data: 训练数据
    :param target: 目标值
    :return:无返回值
    """
    from sklearn.model_selection import GridSearchCV
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=50,
                                  learning_rate=0.16, n_estimators=1000,
                                  max_depth=8,min_child_samples=20,min_child_weight=0.001,
                                  metric='rmse', bagging_fraction=0.6,
                                  feature_fraction=0.88,reg_alpha=0.5,reg_lambda=0.001)

    # 调试参数
    params_test = {
                     'max_depth': range(6, 10, 1),
                    'num_leaves': range(45, 70, 5),
                    'n_estimators': [1000, 1500, 2000, 2500],
                    'min_child_samples': [18, 19, 20, 21, 22],
                    'min_child_weight': [0.001, 0.002],
                    'feature_fraction': [0.85,0.88, 0.9, 0.92,0.94],
                    'bagging_fraction': [0.85,0.88, 0.9, 0.92,0.94],
                    'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
                    'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]

                    }
    # 用网格搜索来调试参数
    gsearch = GridSearchCV(estimator=model_lgb, param_grid=params_test,
                            scoring='neg_mean_squared_error', cv=5, verbose=1,
                            n_jobs=4)
    gsearch.fit(train_data, target)
    print('参数的最佳取值：{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(np.sqrt(-gsearch.best_score_)))

def predictResult(train_data, target, best_model, test_new, test_df):
    """将训练数据采用十折交叉后，给出验证集的得分，同时对测试数据进行预测"""
    predict_results = []
    sk = KFold(n_splits=10, shuffle=True)
    for train, test in sk.split(train_data, target):
        x_train = train_data.iloc[train]
        y_train = target.iloc[train]
        x_test = train_data.iloc[test]
        y_test = target.iloc[test]
        best_model.fit(x_train, y_train)
        y_hat = best_model.predict(x_test)
        y_hat[y_hat < 0] = 0
        print(1 / (mean_squared_error(y_test, y_hat) ** 0.5 + 1))
        predict_results.append(best_model.predict(test_new))
    # 对测试集预测的结果，进行求均值
    predict_results = np.array(predict_results)
    pre_values = predict_results.mean(axis=0)
    pre_values[pre_values < 0] = 0
    # 将预测后的数据赋值到提交样本数据中，并将结果保存
    test_df.iloc[:, 2] = pre_values
    test_df.to_csv('result.csv', encoding='utf-8', header=None,
                   index=None)


if __name__ == '__main__':
    # 导入数据
    df1, df2, df3, df4, test_df = load_data()
    # 对df1进行处理
    df1, dict_wd, dict_h, dict_id, dict_isweekend = processing_df1(df1)
    # 将df1和df2进行合并,合并为train_data
    pd.set_option('display.width', None)
    train_data = df1.copy()
    train_data = pd.merge(df1, df2, on=['ID'])
    # 对df3进行处理
    dict_imi, dict_isi, dict_imo, dict_iso, dict_pimi, dict_pisi, dict_pimo, dict_piso = processing_df3(
        df3)
    train_data, dict_at, dict_hi = processing_train_data(train_data, dict_imi,
                                                         dict_isi, dict_imo,
                                                         dict_iso)
    train_data.head()
    # 对测试数据进行处理，增加衍生变量
    test_data = processing_test_data(test_df, df2, dict_isweekend, dict_wd,
                                     dict_h, dict_id, dict_pimi, dict_pisi,
                                     dict_pimo, dict_piso, dict_at, dict_hi)

    # 只获取训练数据中2月份的数据
    train_data = train_data[train_data['day'] >= 15]
    train_data.index = np.arange(train_data.shape[0])
    # 选择训练集所需的特征，并将目标值单独取出
    target = train_data['people_index']
    train_data.drop(['people_index', 'day'], axis=1, inplace=True)
    print('train_data.head()\n', train_data.head())

    # 将测试集列名与训练集列名保持一致
    test_new = test_data[train_data.columns]
    print('test_new.head()\n', test_new.head())

    # 参数调整
    # get_best_params(train_data, target)

    # # # 建立模型
    best_model = train_model(train_data, target)

    # # 将训练数据进行十折交叉后，输入到训练好的模型中，进行预测，同时对测试数据进行预测
    predictResult(train_data,target,best_model,test_new,test_df)
