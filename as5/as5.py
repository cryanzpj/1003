import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

file_train = open('data/banana_train.csv')
file_test = open('data/banana_test.csv')

train = np.array(map(lambda x: x[:2] + [x[-1].strip()], [i.split(',') for i in file_train]), dtype='float')
test = np.array(map(lambda x: x[:2] + [x[-1].strip()], [i.split(',') for i in file_test]), dtype='float')

y_train = np.array([0 if i == -1 else 1 for i in train[:, 0]])
y_test = np.array([0 if i == -1 else 1 for i in test[:, 0]])
X_train = train[:, 1:]
X_test = test[:, 1:]

n_classes = 2
plot_colors = "bry"
plot_step = 0.02

error = np.zeros((2, 10))

for i in xrange(1, 11):
    idx = np.arange(X_train.shape[0])
    np.random.seed(1)
    np.random.shuffle(idx)
    X = X_train[idx]
    y = y_train[idx]

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    clf = DecisionTreeClassifier(max_depth=i).fit(X, y)
    plt.subplot(3, 4, i)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    training_error = np.sum(np.equal(clf.predict(X_train), 1 - y_train)) / float(y_train.shape[0])
    testing_error = np.sum(np.equal(clf.predict(X_test), 1 - y_test)) / float(y_test.shape[0])
    error[:, i - 1] = np.array([training_error, testing_error])

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.axis("tight")

    for j, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == j)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=str(j),
                    cmap=plt.cm.Paired)

    plt.axis("tight")
    plt.legend(fontsize = 10)
    plt.title('Max Depth = '+ str(i))
plt.suptitle('Decision surface for different depth')
plt.show()

plt.plot(list(xrange(1, 11)), error[0, :], label='training error')
plt.plot(list(xrange(1, 11)), error[1, :], label='testing error')
plt.xlabel("max depths")
plt.ylabel('error')
plt.legend()
plt.suptitle('Error plot')
plt.show()

min_error = 1
ite = 0
for a in xrange(1, 21):
    for b in xrange(1, 21):
        for c in xrange(1, 21):
            idx = np.arange(X_train.shape[0])
            np.random.seed(1)
            np.random.shuffle(idx)
            X = X_train[idx]
            y = y_train[idx]

            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std
            #normalize testing data
            X_test_temp = (X_test -mean)/std

            clf = DecisionTreeClassifier(max_depth=a, min_samples_leaf=b, min_samples_split=c).fit(X, y)

            #training_error = np.sum(np.equal(clf.predict(X_train), 1 - y_train)) / float(y_train.shape[0])
            testing_error = np.sum(np.equal(clf.predict(X_test_temp), 1 - y_test)) / float(y_test.shape[0])
            if testing_error < min_error:
                min_error = testing_error
                par = [a, b, c]
            ite += 1



y_train = np.array([-1 if i == 0 else 1 for i in y_train])
y_test = np.array([-1 if i == 0 else 1 for i in y_test])


def AdaBoost(X, y,n_round=5,test_x = None,test_y= None,visual = False):

    n_instance = X.shape[0]
    w = np.ones(n_instance) / n_instance
    models = np.zeros(n_round,dtype = object)
    error = np.zeros(n_round)
    alphas = np.zeros(n_round)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    

    for n in xrange(1, n_round + 1):
        W = np.sum(w)
        clf = DecisionTreeClassifier(max_depth=3).fit(X,y,sample_weight=w)
        models[n-1] = clf
        error = np.sum((1-np.equal(clf.predict(X), y)) * w)/W
        alphas[n-1] = np.log((1-error)/error)
        w = np.exp((1- np.equal(clf.predict(X),y))*alphas[n-1])*w

        #visualizer
        if visual:
            plot_colors = 'br'
            plt.subplot(2, 5, n)
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

            Z = np.sum(map(lambda i:alphas[i] * models[i].predict(np.c_[xx.ravel(), yy.ravel()]),list(xrange(n))),0)
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
            plt.axis("tight")

            for i, color in zip([-1,1], plot_colors):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1],s=10000*(w/W) , c=color, label="class" + str(i),
                            cmap=plt.cm.Paired)
                plt.legend(fontsize=10,title = 'Round' + str(n))
    plt.suptitle('Decision surface for different rounds')
    plt.show()

    # record rounds errors
    G_n = np.array(map(lambda i: alphas[i] * models[i].predict(X), list(xrange(n))))
    train_error = 1 - np.sum(np.equal(np.sign(np.sum(G_n,0)),y))/float(n_instance)

    if  (test_x != None) and  (test_y != None) :
        G_n_test = np.array(map(lambda i: alphas[i] * models[i].predict(test_x), list(xrange(n))))
        test_error = 1 - np.sum(np.equal(np.sign(np.sum(G_n_test,0)),test_y))/float(test_y.shape[0])
        return [train_error,test_error]

    else:
        return train_error

AdaBoost(X_train,y_train,10,visual = True)

q_3_3 = np.zeros((2,10))
for i in range(10):
    q_3_3[:,i] = AdaBoost(X_train,y_train,i+1,X_test,y_test)

plt.plot(list(xrange(1, 11)), q_3_3[0, :], label='training error')
plt.plot(list(xrange(1, 11)), q_3_3[1, :], label='testing error')
plt.xlabel("number of rounds")
plt.ylabel('error')
plt.legend()
plt.suptitle('Error plot')
plt.show()
