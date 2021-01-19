from scipy import stats
from sklearn import tree    # 对比
from sklearn import metrics
from matplotlib import rcParams
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestClassifier  # 对比
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

# '''
#    C4.5
#  & RF
# '''

def entroy(labels):
    pks = np.array([np.where(labels == label)[0].size for label in np.unique(labels)]) / len(labels)
    return - np.dot(pks, np.log2(pks).T)


class Node(object):
    def __init__(self, labels, label=None, attr=None, category=None, division_value=None):
        global convert_attr_to_colname
        self.labels = labels
        self.label = label
        self.attr = convert_attr_to_colname[attr] if attr != None else None
        self.category = category
        self.division_value = division_value
        self.chidnodes = []


# C4.5 决策树
# 考虑到一个问题：建树的过程中需要知道每个属性的各个取值
# 需要加一个最大深度吧
class DecisionTree(object):
    # 可以设置一些属性
    def __init__(self, max_depth=15):
        self.max_depth = max_depth
        self.depth = 0

    # 将连续值进行离散化 (好在连续属性只能划分为大于小于两类)
    def preprocess(self, data):
        # 标志位(离散or连续)
        self.discrete_or_continuous = np.zeros((data.shape[1],))
        self.discrete_serials = []
        for i in range(data.shape[1]):
            if isinstance(data[0, i], float):
                # 将连续值进行离散化
                self.discrete_or_continuous[i] = 1
                diff_values = np.sort(np.unique(data[:, i]))
                self.discrete_serials.append(np.asarray(
                    [(diff_values[j] + diff_values[j + 1]) / 2 for j in range(diff_values.size - 1)]))
            else:
                self.discrete_serials.append(None)
                continue

    def fit(self, data, labels):
        self.preprocess(data)
        self.data = data
        self.labels = labels
        start_time = time.time()
        attrs = np.arange(data.shape[1])

        # 在训练集上训练
        print('\n')
        print("<--------------决策树开始建立-------------->")
        self.node = self._grow(data, labels, attrs, 0)

        # 剪枝
        self.score = self.Score(self.data, self.labels)
        print("Actual depth of decision tree: ",  self.depth)
        print("Traing time: %.2f s" % (time.time() - start_time))
        print("Accuracy on train set: %.2f" % self.score)
        print("<--------------决策树建立完成-------------->")
        print('\n')


    def _grow(self, data, labels, attrs, depth, attr=None, category=None):
        # 记录决策树的最大深度
        if depth > self.depth:
            self.depth = depth
        if len(np.unique(labels)) == 1:
            return Node(labels, labels[0], attr, category)
        elif attrs.size == 0 or len(np.unique(data.astype('<U22')[:, attrs], axis=0)) == 1 or depth >= self.max_depth:
            return Node(labels, stats.mode(labels)[0][0], attr, category)
        else:
            # 最优节点选择策略
            # 信息增益和信息增益率都单独计算
            node = Node(labels, None, attr, category)
            Ent_D = entroy(labels)
            gains = np.zeros((attrs.size, ), dtype=np.float64)
            Ivs = np.zeros_like(gains)
            division_inter_values = np.zeros_like(Ivs)
            for idx, a in enumerate(attrs):
                # 判断是离散属性还是连续属性
                if self.discrete_or_continuous[a]:
                    MEnt_Dv = 0
                    Mratio = 0
                    for _, interplot in enumerate(self.discrete_serials[a]):
                        smaller = np.where(data[:, a] < interplot)[0]
                        if smaller.size == len(data) or smaller.size == 0:
                            continue
                        bigger = np.array(list(filter(lambda x: x not in smaller, np.arange(data.shape[0]))))
                        ratio = smaller.size / len(data)
                        Ent_Dv = Ent_D - ratio * entroy(labels[smaller]) - (1 - ratio) * entroy(labels[bigger])
                        if Ent_Dv > MEnt_Dv:
                            MEnt_Dv = Ent_Dv
                            division_inter_values[idx] = interplot
                            Mratio = ratio
                    gains[idx] = MEnt_Dv
                    Ivs[idx] = - Mratio * np.log2(Mratio) - (1 - Mratio) * np.log2(1 - Mratio) if Mratio else 0
                else:
                    for category in np.unique(data[:, a]):
                        samples = np.where(data[:, a] == category)[0]
                        ratio = len(samples) / len(data)
                        gains[idx] += ratio * entroy(labels[samples])
                        Ivs[idx] -=  ratio * np.log2(ratio)
                    gains[idx] = Ent_D - gains[idx]

            # 启发式：从信息增益大于平均值中选择信息增益率最大
            # 注意浮点数精度问题
            choices = np.where(gains >= gains.mean())[0]
            if choices.size == 0:
                choices = np.arange(attrs.size)
                choice = choices[np.argmax(gains[choices] / Ivs[choices])]
            else:
                choice = choices[np.argmax(gains[choices] / Ivs[choices])]
            best_attr = attrs[choice]

            # 0代表小于，1代表大于
            if self.discrete_or_continuous[best_attr]:
                division_value = division_inter_values[choice]
                node.division_value = division_value
                # 小于划分值是 0
                best_attr_category_samples1 = np.where(data[:, best_attr] < division_value)[0]
                node.chidnodes.append(self._grow(data[best_attr_category_samples1, :], labels[best_attr_category_samples1],
                                   attrs, depth + 1, best_attr, "0"))
                # 大于划分值是 1
                best_attr_category_samples2 = np.where(data[:, best_attr] > division_value)[0]
                node.chidnodes.append(self._grow(data[best_attr_category_samples2, :], labels[best_attr_category_samples2],
                                   attrs, depth + 1, best_attr, "1"))
            else:
                for best_attr_category in np.unique(data[:, best_attr]):
                    best_attr_category_samples = np.where(data[:, best_attr] == best_attr_category)[0]
                    node.chidnodes.append(
                        self._grow(data[best_attr_category_samples, :], labels[best_attr_category_samples],
                                   attrs[attrs != best_attr], depth + 1, best_attr, best_attr_category))
            return node

    def Score(self, val_data, val_labels):
        res = [self.predict(sample) for sample in val_data]
        return (np.where(res == val_labels)[0].size / len(val_labels)) * 100

    # 递归预测
    def predict(self, sample):
        return self.rec_predict(sample, self.node)

    def rec_predict(self, sample, node):
        global convert_colname_to_attr
        if node.label:
            return node.label
        else:
            # 对待连续
            if node.division_value != None:
                if sample[convert_colname_to_attr[node.chidnodes[0].attr]] < node.division_value:
                    return self.rec_predict(sample, [nd for nd in node.chidnodes if nd.category == '0'][0])
                else:
                    return self.rec_predict(sample, [nd for nd in node.chidnodes if nd.category == '1'][0])

            # 对待离散
            same_category_node = [nd for nd in node.chidnodes if
                                  nd.category == sample[convert_colname_to_attr[nd.attr]]]
            if same_category_node:
                return self.rec_predict(sample, same_category_node[0])
            else:
                return stats.mode(node.labels)[0][0]

    def postcut(self, val_data, val_labels):
        self.val_score = self.Score(val_data, val_labels)
        self.cut(val_data, val_labels, self.node)

    # 后剪枝操作
    def cut(self, val_data, val_labels, node=None):
        if not node.label and np.all([nd.label != None for nd in node.chidnodes]):
            node.label = stats.mode(node.labels)[0][0]
            temp_score = self.Score(val_data, val_labels)
            if temp_score > self.val_score:
                self.val_score = temp_score
                node.chidnodes.clear()
            else:
                node.label = None
        else:
            for nd in node.chidnodes:
                self.cut(val_data, val_labels, nd)
        return

class RandomForest(object):
    # 样本扰动 + 属性扰动
    def __init__(self, tree_nums, val_data, val_labels):
        self.tree_nums = tree_nums
        self.val_data = val_data
        self.val_labels = val_labels
        self.Dtrees = []
        self.sub_attrs = []

    def fit(self, data, labels):
        attrs = np.arange(data.shape[1])
        for i in range(self.tree_nums):
            # 样本扰动
            train_index = np.random.choice(np.arange(data.shape[0]), int(0.5 * data.shape[0]), replace=False)
            train_data = data[train_index, :]
            train_labels = labels[train_index]

            # 属性扰动 (随机大小的属性子集)
            sub_set = np.sort(np.random.choice(attrs, np.random.randint(3, data.shape[1]), replace=False))
            self.sub_attrs.append(sub_set)

            # 训练决策树
            new_tree = DecisionTree(3)
            new_tree.fit(train_data[:, sub_set], train_labels)
            new_tree.postcut(self.val_data[:, sub_set], self.val_labels)
            self.Dtrees.append(new_tree)

    def Score(self, test_data, test_labels):
        res = [self.predict(sample) for sample in test_data]
        return (np.where(res == test_labels)[0].size / len(test_labels)) * 100

    # 采用 "简单投票法" 进行预测
    def predict(self, sample):
        res = []
        for idx, Dtree in enumerate(self.Dtrees):
            res.append(Dtree.predict(sample[self.sub_attrs[idx]]))
        return stats.mode(res)[0][0]


# 数据预处理
def Abalone():
    abalone = pd.read_csv("Abalone.csv")

    abalone['Age'] = abalone['Rings'] + 1.5
    abalone.drop(columns=['Rings'], inplace=True)

    var = 'Viscera weight'
    abalone.drop(abalone[(abalone[var] > 0.5) & (abalone['Age'] < 25)].index, inplace=True)
    abalone.drop(abalone[(abalone[var] < 0.5) & (abalone['Age'] > 25)].index, inplace=True)

    var = 'Shell weight'
    abalone.drop(abalone[(abalone[var] > 0.6) & (abalone['Age'] < 25)].index, inplace=True)

    var = 'Shucked weight'
    abalone.drop(abalone[(abalone[var] >= 1) & (abalone['Age'] < 20)].index, inplace=True)
    abalone.drop(abalone[(abalone[var] < 1) & (abalone['Age'] > 20)].index, inplace=True)

    var = 'Whole weight'
    abalone.drop(abalone[(abalone[var] >= 2) & (abalone['Age'] < 25)].index, inplace=True)

    var = 'Diameter'
    abalone.drop(abalone[(abalone[var] < 0.1) & (abalone['Age'] < 5)].index, inplace=True)
    abalone.drop(abalone[(abalone[var] < 0.6) & (abalone['Age'] > 25)].index, inplace=True)
    abalone.drop(abalone[(abalone[var] >= 0.6) & (abalone['Age'] < 25)].index, inplace=True)

    var = 'Height'
    abalone.drop(abalone[(abalone[var] > 0.4) & (abalone['Age'] < 15)].index, inplace=True)
    abalone.drop(abalone[(abalone[var] < 0.02) & (abalone['Age'] > 7)].index, inplace=True)

    var = 'Length'
    abalone.drop(abalone[(abalone[var] < 0.1) & (abalone['Age'] < 5)].index, inplace=True)
    abalone.drop(abalone[(abalone[var] < 0.8) & (abalone['Age'] > 25)].index, inplace=True)
    abalone.drop(abalone[(abalone[var] >= 0.8) & (abalone['Age'] < 25)].index, inplace=True)

    return abalone.drop(columns=['Height', 'Length'],  axis=1)

def Convert(data):
    cdata = data.to_numpy(dtype=object)

    # 对年龄预测
    labels = cdata[:, -1]
    data = cdata[:, :-1]

    # 对性别预测 
    labels = cdata[:, 0]
    data = cdata[:, 1:]

    return data, labels

def Breast_cancer():
    data = pd.read_csv("data.csv")
    del_cols = ['id', 'Unnamed: 32', 'diagnosis']

    # 提取属性
    x = data.drop(columns=del_cols, axis=1)
    x = (x - x.mean()) / (x.std())   

    # 提取标签
    y = data['diagnosis']

    # 去除异常值
    K = 1.5
    for col in x.columns:
        Q1 = x[col].quantile(0.25)
        Q3 = x[col].quantile(0.75)
        IQR = Q3 - Q1 

        filter = (x[col] >= Q1 - K * IQR) & (x[col] <= Q3 + K *IQR)
        x = x.loc[filter]
        y = y.loc[filter]

    # 特征选择
    # 0-1转化 
    y_ = y.copy()
    y_[y_ == 'B'] = 0
    y_[y_ == 'M'] = 1
    data = pd.concat([x, y_], axis=1)
    data['diagnosis'] = pd.to_numeric(data['diagnosis']) 
    correlation = data.corr().to_numpy()
    cmps = np.abs(correlation[-1, :-1])

    # 构建剩余特征集合
    res_features = np.arange(30)     
    threshold = 0.8

    for i in range(30):
        relevant_features = res_features[np.where([correlation[i, res_features] > threshold])[0]]
        relevant_features = np.delete(relevant_features, np.argmax(cmps[relevant_features]))
        res_features = np.setdiff1d(res_features, relevant_features)

    return pd.concat([x[x.columns[res_features]], y], axis=1)

# 绘制多分类ROC曲线
def plot_multiclass_roc(y_score, y_test, n_classes):
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 编码
    cols = np.unique(y_test).size
    labels = np.sort(np.unique(y_test))
    y_score_dummies = np.zeros((len(y_score), cols), dtype=np.uint8)
    y_test_dummies = np.zeros((len(y_test), cols), dtype=np.uint8)

    for i in range(len(y_score)):
        y_score_dummies[i, np.where(labels == y_score[i])[0]] = 1

    for i in range(len(y_test)):
        y_test_dummies[i, np.where(labels == y_test[i])[0]] = 1

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score_dummies[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()

    fig.savefig("roc.svg")
    plt.show()

if __name__ == '__main__':
    # data = Breast_cancer()

    data = Abalone()
    data = data.sample(frac=0.3, random_state=200)

    print("总样本数: ", len(data))
    Attr_names = list(data.columns)[:-1]

    # # 划分训练集、验证集和测试集
    temp = data.sample(frac=0.8, random_state=200)
    test = data.drop(temp.index)

    train = temp.sample(frac=0.8, random_state=200)
    val = temp.drop(train.index)

    # 数据格式转换
    train_data, train_labels = Convert(train)
    val_data, val_labels = Convert(val)
    test_data, test_labels = Convert(test)

    print("训练集样本数：", len(train_data))
    print("验证集样本数：", len(val_data))
    print("测试集样本数：", len(test_data))

    # 属性名
    convert_attr_to_colname = {}
    convert_colname_to_attr = {}
    for idx, name in enumerate(Attr_names):
        convert_attr_to_colname[idx] = name
        convert_colname_to_attr[name] = idx

    rcParams['font.family'] = 'Microsoft YaHei'

    # 对比sklearn
    le1 = preprocessing.LabelEncoder()
    le1.fit(["M", "F", 'I'])
    train_labels1 = le1.transform(train_labels)
    test_labels1 = le1.transform(test_labels)

    # clf = tree.DecisionTreeClassifier(max_depth=13)
    # clf.fit(train_data, train_labels1)
    # y_pred = clf.predict(test_data)
    # print("准确度:", metrics.accuracy_score(test_labels1, y_pred))
    # print("macro精度:", metrics.precision_score(test_labels1, y_pred, average="macro"))
    # print("macro召回率:", metrics.recall_score(test_labels1, y_pred, average="macro"))

    # 演示一
    Dtree = DecisionTree(13)
    Dtree.fit(train_data, train_labels)
    Dtree.postcut(val_data, val_labels)
    res = np.array([Dtree.predict(sample) for sample in test_data])
    # 绘制ROC曲线并计算AUC值
    plot_multiclass_roc(res, test_labels, 3)

    res = le1.transform(res)
    print("准确度:", metrics.accuracy_score(test_labels1, res))
    print("macro精度:", metrics.precision_score(test_labels1, res, average="macro"))
    print("macro召回率:", metrics.recall_score(test_labels1, res, average="macro"))

    ax = plt.subplot()
    cm = confusion_matrix(test_labels1, res)
    sns.heatmap(cm, annot=True, ax=ax)
    ax.xaxis.set_ticklabels(['M', 'F', 'I'])
    ax.yaxis.set_ticklabels(['M', 'F', 'I'])
    plt.savefig("cm.svg")
    plt.show()

    # 演示二: 控制数的深度 -> 控制训练误差
    # train_scores = []
    # test_scores = []
    # cut_test_scores = []
    # for dep in range(2, 20):
    #     Dtree = DecisionTree(dep)
    #     Dtree.fit(train_data, train_labels)
    
    #     # 训练集准确率
    #     train_scores.append(Dtree.score)
    
    #     # 剪枝前测试
    #     test_scores.append(Dtree.Score(test_data, test_labels))
    
    #     # 剪枝
    #     Dtree.postcut(val_data, val_labels)
    
    #     # 剪枝后测试
    #     cut_test_scores.append(Dtree.Score(test_data, test_labels))

    # # 绘制训练误差随决策树深度的变化过程 (希望满足预期)
    # fig, axs = plt.subplots(nrows=1, ncols=2)
    
    # axs[0].plot(np.arange(len(train_scores)) + 3, train_scores, label="训练集准确率")
    # axs[1].plot(np.arange(len(test_scores)) + 3, test_scores, label="剪枝前测试集准确率")
    # axs[1].plot(np.arange(len(cut_test_scores)) + 3, cut_test_scores, label="剪枝后测试集准确率")
    
    # for i in range(2):
    #     axs[i].legend()
    
    # plt.savefig("depth.svg")
    # plt.show()

    # 演示三：随机森林
    # Rf = RandomForest(10, val_data, val_labels)
    # Rf.fit(train_data, train_labels)
    # res = np.array([Rf.predict(sample) for sample in test_data])
    # res = le1.transform(res)
    # print("准确度:", metrics.accuracy_score(test_labels1, res))
    # print("macro精度:", metrics.precision_score(test_labels1, res, average="macro"))
    # print("macro召回率:", metrics.recall_score(test_labels1, res, average="macro"))