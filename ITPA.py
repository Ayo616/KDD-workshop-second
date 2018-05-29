
from sklearn.ensemble import IsolationForest
import pandas as pd
from YUtils.OneHotEncode import OneHotForLael

# parameters
train_file_address = './exp3/less2/train.csv'
test_file_address = "./exp3/less2/test.csv"
feature_columns = ['x','y','z']
label_columns = ['label']
target_label = 'label'
SaveTraindata = './exp3/less2/trainA3040.csv'
SaveTestdata = './exp3/less2/testA5060.csv'


def IsolationForest_calulate(train_data_one,test_data):
    # 使用异常检测方法
    clf = IsolationForest()
    # 训练异常检测模型
    clf.fit(train_data_one)
    # 模型预测
    Pre_result = clf.predict(test_data)
    # 计算多少个概率
    prob = len([x for x in Pre_result if x == 1])/len(Pre_result)
    return prob

def load_data():
    # 读取训练集和测试集
    train_content = pd.read_csv(train_file_address)
    test_content = pd.read_csv(test_file_address)
    # 热编码标签
    train_content = OneHotForLael(train_content,target_label,feature_columns)
    test_content = OneHotForLael(test_content,target_label,feature_columns)
    '''
    # 将训练集中的特征与标签分开
    train_features = pd.DataFrame(train_content,columns=feature_columns)
    trian_label = pd.DataFrame(train_content,columns=label_columns)
    # 将测试集中的特征与标签分开
    test_features = pd.DataFrame(test_content,columns=feature_columns)
    test_label = pd.DataFrame(test_content,columns=label_columns)
    return train_features,trian_label,test_features,test_label
    '''
    return train_content,test_content


def DivedByLabel(data,tdata):
    # get the different number of people dataset
    count = data[target_label].value_counts()
    countMap = {}
    for i in count.keys():
        list = pd.DataFrame(data.loc[data[target_label]==i])
        countMap[i] = list

    Tcount = tdata[target_label].value_counts()
    TcountMap = {}
    for n in Tcount.keys():
        Tlist = pd.DataFrame(tdata.loc[tdata[target_label]==n])
        TcountMap[n] = Tlist
    # 返回训练集实例字典、测试集实例字典
    return countMap,TcountMap

import numpy as np
# 参数
EPOCH = 20
SizeOfSpace = 60
SizeOfSubSpace = 30

def GenerateTrainTable():
    # 加载数据
    train_content,test_content = load_data()
    # 得到中间数据
    countMap,TcountMap = DivedByLabel(train_content,test_content)
    # 从训练集中每种类别抽取若干个
    count = train_content[target_label].value_counts()
    # construct a dataframe to store probility for each train instance for each label
    df = pd.DataFrame(columns=count.keys())
    df = pd.concat([df, pd.DataFrame(columns=['label'])],axis =0)
    for epoch in range(EPOCH):
        for w in count.keys():
            # 构建小空间，用于构建训练表
            ConstructTrainData = countMap[w].sample(n= SizeOfSubSpace)
            list = [] # store temp prob
            for i in count.keys():
                traindata = countMap[i].sample(n = SizeOfSpace)
                # 大空间
                Space = traindata.loc[:,feature_columns]
                # 小空间
                SubSpace = ConstructTrainData.loc[:, feature_columns]
                prob = IsolationForest_calulate(Space,SubSpace)
                list.append(prob)

            # 选取小空间内占大多数的类别，作为标签
            w = ConstructTrainData[target_label].value_counts().index[0]
            list.append(w)
            print('list',list)
            from YUtils.util import Add_list_colum
            df = Add_list_colum(list,df)

    # rebbuild index
    df = df.reset_index(drop=True)
    # print(df.head())
    df.to_csv(SaveTraindata)

def GenerateTestTable():
    # 加载数据
    train_content,test_content = load_data()
    # 得到中间数据
    countMap,TcountMap = DivedByLabel(train_content,test_content)
    # 从训练集中每种类别抽取若干个
    count = train_content[target_label].value_counts()
    # construct a dataframe to store probility for each train instance for each label
    df = pd.DataFrame(columns=count.keys())
    df = pd.concat([df, pd.DataFrame(columns=['label'])],axis =0)
    for epoch in range(EPOCH):
        for w in count.keys():
            # 构建小空间，用于构建测试表
            e = np.random.randint(1,len(test_content)/SizeOfSubSpace-1)
            ConstructTestData = test_content[SizeOfSubSpace*e:SizeOfSubSpace*(e+1)]

            list = [] # store temp prob
            for i in count.keys():
                traindata = countMap[i].sample(n = SizeOfSpace)
                # 大空间
                Space = traindata.loc[:,feature_columns]
                # 小空间
                SubSpace = ConstructTestData.loc[:, feature_columns]
                prob = IsolationForest_calulate(Space,SubSpace)
                list.append(prob)

            # 选取小空间内占大多数的类别，作为标签
            w = ConstructTestData[target_label].value_counts().index[0]
            list.append(w)
            print('list',list)
            from YUtils.util import Add_list_colum
            df = Add_list_colum(list,df)

    # rebbuild index
    df = df.reset_index(drop=True)
    print(df.head())
    df.to_csv(SaveTestdata)

def test(train_feature,train_label,test_feature,test_label):
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    from sklearn.svm import SVC
    # fit a SVM model to the data
    # model = SVC()
    # model = GaussianNB()
    # model = LogisticRegression()
    from sklearn.neighbors import KNeighborsClassifier
    # fit a k-nearest neighbor model to the data
    import time
    currenttime = time.time()
    # model = KNeighborsClassifier()
    model.fit(train_feature, train_label)
    print(model)
    # make predictions
    expected = test_label
    predicted = model.predict(test_feature)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print(metrics.accuracy_score(expected,predicted))

if __name__ == '__main__':
    GenerateTrainTable()
    GenerateTestTable()
    # test()

