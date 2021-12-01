from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.svm import SVC
import math
import numpy as np

    
# kernel=gaussian
def my_kernel_gaussian1(xi, xj):
    sigma = 0.485
    x = np.dot(xi, np.transpose(xj))
    res = -(x**2)/(2*sigma**2)
    res = np.array(np.exp(res))
    return res


X, y = load_breast_cancer(return_X_y=True)
X_scaled = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

svm_gaussian_kernel = SVC(kernel=my_kernel_gaussian)
svm_gaussian_kernel.fit(X_train, y_train)

prediction = svm_gaussian_kernel.predict(X_test)
print('acc on train data: {:.3f}'.format(svm_gaussian_kernel.score(X_train, y_train)))
print('acc on test data: {:.3f}'.format(svm_gaussian_kernel.score(X_test, y_test)))
