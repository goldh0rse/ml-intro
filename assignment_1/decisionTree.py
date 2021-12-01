from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=10)

for i in range(2, 8):
    tree = DecisionTreeClassifier(max_depth=i)
    tree.fit(X_train,y_train)
    tree.predict(X_test)
    print('Depth: %d' % i)
    print('acc on train dataset: {:.3f}'.format(tree.score(X_train, y_train)))
    print('acc on test dataset: {:.3f}'.format(tree.score(X_test, y_test)))
