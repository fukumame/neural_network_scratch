import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# 境界線を引くための関数定義
def plot_decision_boundary(pred_func,X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid

    # こちらが参考になる
    # http://kaisk.hatenadiary.com/entry/2014/11/05/041011
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# 損失関数の計算
def calculate_loss(model, X, y, weight_decay_rate):
    num_examples = len(X)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 予測を算出するためのForward propagation
    z1 = X.dot(W1) + b1
    # 活性化関数
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    # soft-max関数による出力層の定義
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # probsは予測値であり、例えば以下のような形となる。 1番目が0である確率、2番めが1である確率
    # [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]

    # 予測値における、正解ラベルの確率を配列で出している
    # 例えば、正解ラベル y = [1, 0, 1]の場合、[0.9, 0.8, 0.7] となる
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Lossに荷重減衰を与える (optional)
    # 荷重減衰は、P.193を参照のこと
    data_loss += weight_decay_rate/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# 予測関数
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    # soft-max関数による出力層の定義
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # argmaxは、引数で与えられた配列のうち、大きな値をもつ配列のINDEXを返す
    # 例えば、[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]] の場合、
    # [1,0,1]を返す。 (つまり正解とみなしたラベルの値)
    return np.argmax(probs, axis=1)

# 予測モデルの構築
def build_model(X, y, nn_input_dim, nn_hdim, nn_output_dim, weight_decay_rate, learning_rate, num_passes=20000, print_loss=True):
    np.random.seed(0)
    num_examples = len(X)
    # Xavier の初期値を用いて初期化する。 正規分布に従うランダムな数を出力し、それを前層のノードの個数の平方根で割る
    # 前層のノード数が多いほど、重みのスケール(値の幅)は小さくなる。
    # 入力層のノード数が2, 隠れ層のノード数が3の場合、np.random.randn(2, 3)は、以下の通りになる
    #array([[ 1.33867319, -0.05720098,  1.47962537],
    #    [ 1.51797757, -1.85726716, -0.00898161]])
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)

    # 1x3のゼロ埋め行列を作る。 具体的には、array([[ 0.,  0.,  0.]])
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}
    for i in range(0, num_passes):
        # 現在のモデルのパラメータを用いて予測を実施
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # バックプロパゲーションの実施
        delta3 = probs
        # 以下は出力された確率(0-1の間の値を取る)から-1を引くため、1に近いほどデルタ(差分)はゼロに近づく
        # 逆に1から遠いほど、差分は大きくなる
        # 例えば [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]の場合、[-0.1, -0.2, -0.3]となる。
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        # powerは累乗 a1の二条のこと
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Weight decayによる過学習の防止
        # http://olanleed.hatenablog.com/entry/2013/12/03/010945
        dW2 += weight_decay_rate * W2
        dW1 += weight_decay_rate * W1

        # 学習率に基づき、パラメータを更新
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2

        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 1000回の学習に1回、損失関数を出力する
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model, X, y, weight_decay_rate)))
    return model

np.random.seed(0)
# 学習のための、moonのデータを生成
X, y = sklearn.datasets.make_moons(200, noise=0.20)

## X = [[1,2], [3,4], [5,6]]となっており、x[:,0]は、:で全ての行を表し、その全ての行の0番目の項目を取得しArrayで返す
## つまり、X[:,0]は、[1,3,5]となり、X[:,1]は[2,4,6]となる
## 散布図の生成
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
## グラフの表示
#plt.show()

# ロジスティック回帰モデルを学習させる
#clf = sklearn.linear_model.LogisticRegressionCV()
#clf.fit(X, y)
#
#plot_decision_boundary(lambda x: clf.predict(x),X,y)
#plt.title("Logistic Regression")
#plt.show()

nn_input_dim = 2 # インプット層の次元数
nn_hidden_dim = 3 # 隠れ層の次元数
nn_output_dim = 2 # アウトプット層の次元数

learning_rate = 0.01 # 学習率
weight_decay_rate = 0.01 # 過学習を避けるための正則化の強さ

model = build_model(X, y, nn_input_dim, nn_hidden_dim, nn_output_dim,  weight_decay_rate, learning_rate)
plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()
