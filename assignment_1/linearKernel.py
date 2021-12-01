from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import numpy as np


# kernel=linear
def my_kernel_linear(xi,xj):

    return np.dot(xi, np.transpose(xj))


X, y = load_breast_cancer(return_X_y=True)
X_scaled = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

svm_linear_kernel = SVC(kernel=my_kernel_linear)
svm_linear_kernel.fit(X_train, y_train)

prediction = svm_linear_kernel.predict(X_test)
print('acc on train data: {:.3f}'.format(svm_linear_kernel.score(X_train, y_train)))
print('acc on test data: {:.3f}'.format(svm_linear_kernel.score(X_test, y_test)))
