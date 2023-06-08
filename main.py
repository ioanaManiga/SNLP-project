import pandas as pd
from collections import Counter
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

df1 = pd.read_csv('music_genre.csv', encoding='utf-8')
df1 = df1.dropna()

df1.drop('instance_id', inplace=True, axis=1)
df1.drop('artist_name', inplace=True, axis=1)
df1.drop('track_name', inplace=True, axis=1)
df1.drop('obtained_date', inplace=True, axis=1)

columns_names = df1.columns.values

categ1 = ['key', 'mode', 'music_genre']

(sample, vnum) = df1.shape
vnum = vnum - 1

values = df1.values
y1 = values[:, vnum]

column_transformer = make_column_transformer(
    (OneHotEncoder(), categ1),
    remainder='passthrough')

transformed = column_transformer.fit_transform(df1, y1)

transformed_X_music = pd.DataFrame(
    transformed
)

################################################################################################################

df2 = pd.read_csv('Job_Placement_Data.csv', encoding='utf-8')
df2 = df2.dropna()

categ2 = ['gender', 'ssc_board', 'hsc_board', 'hsc_subject', 'undergrad_degree', 'work_experience', 'specialisation',
          'status']

(sample, vnum) = df2.shape
vnum = vnum - 1

values = df2.values
y2 = values[:, vnum]

# print("", sorted(Counter(y).items()))

column_transformer = make_column_transformer(
    (OneHotEncoder(), categ2)
)

transformed = column_transformer.fit_transform(df2, y2)

transformed_X_flights = pd.DataFrame(
    transformed
)


def decision_tree(data, classes, crit, nodes):
    max_acc = 0
    max_acc_description = ''
    for c in crit:
        for n in nodes:
            clf = tree.DecisionTreeClassifier(criterion=c, max_leaf_nodes=n)
            pred = cross_val_predict(clf, data, classes, cv=5)
            acc = accuracy_score(classes, pred, normalize=False)
            if acc > max_acc:
                max_acc = acc
                max_acc_description = 'criterion = ' + c + ', max_leaf_nodes = ' + str(n)

    return 'Decision tree', max_acc / len(data.index), max_acc_description


def naive_bayes(data, classes, smoothing):
    max_acc = 0
    max_acc_description = ''
    for s in smoothing:
        clf = GaussianNB(var_smoothing=s)
        pred = cross_val_predict(clf, data, classes, cv=5)
        acc = accuracy_score(classes, pred, normalize=False)
        if acc > max_acc:
            max_acc = acc
            max_acc_description = 'var_smoothing = ' + str(s)

    return 'Naive bayes', max_acc / len(data.index), max_acc_description


def knn(data, classes, neighbours, weights, power):
    max_acc = 0
    max_acc_description = ''
    for n in neighbours:
        for w in weights:
            for p in power:
                print(n, w, p, "knn")
                clf = KNeighborsClassifier(n_neighbors=n, weights=w, p=p)
                pred = cross_val_predict(clf, data, classes, cv=5)
                acc = accuracy_score(classes, pred, normalize=False)
                if acc > max_acc:
                    max_acc = acc
                    max_acc_description = 'n_neighbours = ' + str(n) + ', weights = ' + w + ', p = ' + str(p)

    return 'k-NN', max_acc / len(data.index), max_acc_description


def logistic_regression(data, classes, tol, reg):
    max_acc = 0
    max_acc_description = ''
    for t in tol:
        for c in reg:
            print(t, c, "lr")
            clf = LogisticRegression(tol=t, C=c, max_iter=500)
            pred = cross_val_predict(clf, data, classes, cv=5)
            acc = accuracy_score(classes, pred, normalize=False)
            if acc > max_acc:
                max_acc = acc
                max_acc_description = 'tol = ' + str(t) + ', C = ' + str(c)

    return 'Logistic regression', max_acc / len(data.index), max_acc_description


def neural_network(data, classes, layers_sizes, activation, max_iter):
    max_acc = 0
    max_acc_description = ''
    for l in layers_sizes:
        for a in activation:
            print(l, a)
            clf = MLPClassifier(hidden_layer_sizes=l, activation=a, max_iter=max_iter)
            pred = cross_val_predict(clf, data, classes, cv=5)
            acc = accuracy_score(classes, pred, normalize=False)
            if acc > max_acc:
                max_acc = acc
                max_acc_description = 'activation = ' + a + ', hidden_layer_sizes = ' + str(l)

    return 'Neural networks', max_acc / len(data.index), max_acc_description


def find_best_configuration(x, y, set_name):
    results = []
    results.append(decision_tree(x, y, ['gini', 'entropy', 'log_loss'], [None, 10, 12, 14, 16, 18, 20]))

    results.append(naive_bayes(x, y, [1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]))

    results.append(knn(x, y, [1, 3, 5, 7, 9, 11], ["uniform", "distance"], [1, 2, 3]))

    results.append(logistic_regression(x, y, [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12],
                                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

    results.append(
        neural_network(x, y, [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,), (110,), (50, 50)],
                       ["identity", "logistic", "relu", "tanh"], 2000))

    file = open(set_name + ".txt", "w")
    file.write('RESULTS FOR ' + set_name + '\n\n')
    for elem in results:
        file.write(elem[0] + ' - best accuracy: ' + str(elem[1]) + ' for: ' + elem[2] + '\n')
    results = sorted(results, key=lambda result: result[1], reverse=True)
    file.write('The configurations that obtained best accuracy are:\n')
    index = 0
    while results[index][1] == results[0][1]:
        file.write('-> ' + results[index][0] + ': ' + results[index][2] + '\n')
        index += 1
    file.write('\n\n')
    file.close()


find_best_configuration(transformed_X_music, y1, 'Music genre')
find_best_configuration(transformed_X_flights, y2, 'Job placement')
