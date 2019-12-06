import collections
from subprocess import call

import numpy as np
import pandas as pd
import pydotplus
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

x_train = pd.read_csv('./train.csv')
x_test = pd.read_csv('./test.csv')
y_test = pd.read_csv('./gender_submission.csv')

x_test = pd.concat([y_test, x_test], axis=1)
# зависимая переменная
y_test = y_test['Survived']
y_train = x_train['Survived']
# убираем ненужные значения
x_train = x_train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Survived'], axis=1)
x_test = x_test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Survived'], axis=1)
# test[train.isnull()].info()
# дополняем значение возраста
x_train['Age'] = x_train.Age.fillna(x_train['Age'].median())
x_test['Age'] = x_test.Age.fillna(x_test['Age'].median())
# дополняем незаполненый порт отправления
x_train.Embarked = x_train.Embarked.fillna('S')
x_test.Embarked = x_test.Embarked.fillna('S')
# train[train.isnull()].info()

x_train['Sex'] = x_train['Sex'].map({"male": 0, "female": 1})
x_train['Embarked'] = x_train['Embarked'].map({"S": 0, "C": 1, "Q": 2})
# отсутствующим полям суммы отплаты за плавание присваивается медианное значение
x_train["Fare"] = x_train["Fare"].fillna(x_train["Fare"].median())
x_train['Age'] = x_train['Age'].astype(int)

x_test['Sex'] = x_test['Sex'].map({"male": 0, "female": 1})
x_test['Embarked'] = x_test['Embarked'].map({"S": 0, "C": 1, "Q": 2})
# отсутствующим полям суммы отплаты за плавание присваивается медианное значение
x_test["Fare"] = x_test["Fare"].fillna(x_test["Fare"].median())
x_test['Age'] = x_test['Age'].astype(int)

param_grid = {'max_depth': np.arange(1, 21),
              'min_samples_leaf': [1, 5, 10, 20, 50, 100]}

grid = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv = 5)
grid.fit(x_train, y_train)

print("Правильность на тренеровачном наборе: {:.2f}".format(grid.score(x_train, y_train)))
print("Наилучшие значения параметров: {}".format(grid.best_params_))
print("Наилучшее значение кросс-валидац. правильности:{:.2f}".format(grid.best_score_))
print("Правильность на тестовом наборе: {:.2f}".format(grid.score(x_test, y_test)))
'''
decisiontree = DecisionTreeClassifier(max_depth=10, random_state = 10, criterion = 'entropy')
cross_val_score(decisiontree, x_train, y_train, cv=5).mean()
print(cross_val_score(decisiontree, x_train, y_train, cv=5).mean())
print(cross_val_score(decisiontree, x_train, y_train, cv=5).mean())
#x_train.info()
#x_test.info()

model = tree.DecisionTreeClassifier(max_depth=7, random_state = 13, criterion = 'entropy')
model.fit(x_train, y_train)

#точность предсказания
print("Правильность на тренеровачном наборе: {:.2f}".format(model.score (x_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(model.score(x_test, y_test)))
'''
dot_data = export_graphviz(grid.best_estimator_,
                           feature_names=x_train.columns,
                           out_file=None,
                           filled=True,
                           rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')
