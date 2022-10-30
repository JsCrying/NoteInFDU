## K-NN

k-nearest neighbor

一般k取奇数（否则可能出现平手情况）

Image classification

步骤：

-   Given a test doc $d$ and the training  data

    -   identify the set $S_k$ of the $k$ nearest neighbors of $d$,

        $i.e.$, the $k$ training docs most to d.

    -   for each class $c_j \in C$

        -   compute $N(S_k, c_j)$ the number of $S_k$ members that belong to class $c_j$
        -   estimate $Pr[c_j|d]$ as $N(S_k, c_j) / k$

    -   classify $d$ to the majority class of $S_k$ members.

    $c(d)=\underset{c_j \in C}{argmax}Pr[c_j|d]=\underset{c_j\in C}{argmax}N(S_k, c_j)$

-   Pseudocode

    Training corpus or textural dataset: $(C, D)$

    Label set: $C$

    Input raw feature set: $D$
    $$
    \begin{align}
    &Train-kNN(C, D) \\
    &1\space D'\leftarrow Preprocess(D) \\
    &2\space k\leftarrow Select-k(C, D') \\
    &3\space return\space D', k \\
    \\
    &Apply-kNN(C,D',k,d) \\
    &1\space S_k\leftarrow ComputeNeareastNeighbors(D',k,d) \\
    &2\space for\space each\space c_j\in C \\
    &3\space do\space p_j\leftarrow |S_k\cap c_j|/k \\
    &4\space return\space \underset{j}{argmax}p_j
    \end{align}
    $$

## Basic concept for ML

-   Mapping function $f(x, theta)$, input feature vector: $x$
-   -   Linear model
    -   Nonlinear model
-   Object/error/loss functions
-   Optimization techniques

### 梯度下降算法

-   1.  通过迭代寻找最小值
    2.  将数据标记在坐标系，数据拟合（回归）存在着线性关系
    3.  直线方程$h_{\theta}(x)=\theta_0 +\theta_1x$
    4.  找出最好的$\theta_0与\theta_1$
    5.  使用均方误差（Mean Square Error）$\frac{1}{m}\sum_{i=1}^{m}[h_\theta (x_{i})-y_{i}]^2$，m=样本个数，均方差越小越合适
    6.  定义代价函数$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}[h_\theta (x_{i})-y_{i}]^2$，用以求出$\theta_0与\theta_1$
-   数学原理
    -   多元函数$f(x,y,z),g(x,y)$，输入往往是$vector$
    -   偏导数
    -   梯度$\nabla f(x,y,z)=(\frac{\partial f(x,y,z)}{\partial x},\frac{\partial f(x,y,z)}{\partial y},\frac{\partial f(x,y,z)}{\partial z})$，一个函数全部偏导数构成的向量，梯度向量的方向是函数值变化率最大的方向

### Classification model

$w$的状态决定线性或非线性

-   线性

-   Mapping function $y(x)=w^{T}x + w_0$

    $y(x)=f(w^{T}x + w_0),\space f: activation\space function，单调递增$ 很重要！

    $y(x)=f(w^{T}\phi(x)),\phi:Basis\space function$

    object function

    -   perception: $E_P(w)=-\underset{\phi_i \in M}{\sum}w^{T}\phi_i t_i$

    -   Least squared:$E_D(w)=\frac{1}{2}(xw-T)^T(xw-T)$

        $\Rightarrow w=(x^T x)^{-1}x^T T=x^{+}T$

        $\Rightarrow y(x)=w^T x=T^T(x^+)^T x$

    -   Logistic Regression: cross entropy

        $E(w)=-lnP(t|w)=-\overset{N}{\underset{n=1}{\sum}}=\{t_n \ln y_n + (1-t_n)\ln(1-y_n)\}$

        比如用在抛硬币

        

-   非线性

-   1.  $y_k=f(\sum_{j=0}^{n_H}f(\sum_{i=0}^{d}x_i w_{ji})w_{kj})$

    2.  $E(w)=\frac{1}{2} \overset{N}{\underset{n=1}{\sum}}(y(x_n.w)-t_n)^2$

        $P(t|x.w)=y(x.w)^t \{1-y(x.w)\}^{1-t}$

        $\Rightarrow E(w)=-lnP(t|x.w)=-\overset{N}{\underset{n=1}{\sum}}=\{t_n \ln y_n + (1-t_n)\ln(1-y_n)\}$

-   Gradient Descent

    -   $w^*=\underset{w}{argmin}L(w)$
    -   consider loss function $L(w)$ with one parameter w:
        -   (Randomly) Pick an initial value $w^0$
        -   Compute $\frac{dL}{dw}|_{w=w^0},w^1\leftarrow w^0- \eta \frac{dL}{dw}|_{w=w^0}$
        -   Compute $\frac{dL}{dw}|_{w=w^1},w^2\leftarrow w^1- \eta \frac{dL}{dw}|_{w=w^1}$
        -   Many iterations
    -   If two parameters 分别计算

### STEPS

Training

1.  define mapping function
2.  define loss from training data
3.  optimization

### Evaluation

FN: false negatives

TN: true negatives

TP: true positives

FP: false positives

-   How many positives predicted are true positives?

     $Precision=\frac{TP}{TP+FP}$

-   How many positives come back?

    $Recall=\frac{TP}{TP+FN}$

-   How many positives predicted are true positives and how many negatives predicted are true negatives?

    $Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$

## Linear Regression

### supervised Learning

### Polynomial Curve Fitting

-   components for learning in common

    -   a set of variables: inputs x, which are measured or preset
    -   one or more outputs (responses) y
    -   the goal is to use the inputs to predict the values of the outputs: $x\rightarrow y$

-   supervised learning

    -   given a set of data $D=(x_i,y_i)_{i=1}^n, where\space x\in R^d,y\in R$
    -   the prediction of a new sample $x\space by\space D, i.e.,y(x|D)\space or\space P(x|D)$

-   gt: 真实值，标签（label）

-   A simple regression problem

    -   $y(x,w)=\overset{M}{\underset{j=0}{\sum}}w_j x^j$，典型线性模型（关于w的）

        -   The sum of the squares of the errors(SSE) between predictions

            for each data point and the corresponding target value

        -   $MinE(w)=\frac{1}{2}\overset{N}{\underset{n=1}{\sum}}\{y(x_n,w)-t_n\}^2$

        -   The resulting polynomial is given by the function $y(x,  w^*)$

        -   Over-fitting: $E(w^*)=0$, but very poor representation of the function, bad generalization

            -   Regularization control: $\overset{-}{E}(w)=\frac{1}{2}\overset{N}{\underset{n=1}{\sum}}\{y(x_n,w)-t_n\}^2+\frac{\lambda}{2}||w||^2$

        -   root-mean-square(RMS) error: $E_{RMS}=\sqrt{2E(w^*)/N}$

            N for comparing different sizes of databases in the same footing

            the square root for measuring on the same scale as the target variable

        -   $E_{RMS}$计算结果接近的情况下，我们选择更简单的模型（剃刀准则）


### Probability Perspective for Regression

-   maximizing the posterior distribution is equivalent to minimizing the regularized sum-of-squares error function, with a regularization 

    parameter given by $\lambda=\frac{\alpha}{\beta}$

### Loss Function for Regression

-   One simple generalization of the squared loss, called the Minkowski loss, whose expectation is given by

    $\epsilon[\tau_q]=\int\int|y(x)-t|^qp(x,t)dxdt$

    q=2: the expected squared loss

### Linear Basis Function Models

-   $y(x,w)=w_0+w_1x_1+\ldots +w_Dx_D$，where $x=(x_1,\ldots,x_D)^T$

    This is often simply known as linear regression

    parameters: $w_0,\ldots, w_D$

    input variables: $x_i$

-   An extension by considering linear combinations of fixed nonlinear functions of the input variables:

    $y(x,w)=w_0+\overset{M-1}{\underset{j=1}{\sum}}w_j\phi_j(x)$，where $\phi_j(x)$ are know as basis function

    e.g.: in polynomial curve fitting, $\phi_j(x)=x^j$

    $w_0$ is called a bias parameter, $y(x,w)=\overset{M-1}{\underset{j=0}{\sum}}w_j\phi_j(x)=w^T\phi(x)$

-   Polynomial curve fitting, $\phi_j(x)=x^j$

-   Gaussian basis functions, $\phi_j(x)=exp\{-\frac{(x-\mu_j)^2}{2s^2}\}$

-   Sigmodal basis functions, $\phi_j(x)=\sigma(\frac{x-\mu_j}{s})$，where $\sigma(a)$ is the logistic sigmoid function defined by $\sigma(a)=\frac{1}{1+exp(-a)}$

-   $M<N,\space S=span(\phi_1,\ldots,\phi_{M-1})$

    y can live anywhere in the M-dimensional subspace

    $E_D(w)=||y-t||^2$ （真实值和估计值差距）

    the least-squares solution for w corresponds to that choice of that lies in subspace S and that is closet to t

    the solution corresponds to the orthogonal projection of t onto the subspace S

-   To control over-fitting, total error function takes the form

    $\overset{-}{E}(w)=E_D(w)+\lambda E_w(w)$

    one of the simplest forms of regularizer is given by

    $E_w(w)=\frac{1}{2}w^Tw$

    if the sum-of squares error function is taken, then total error functions

    $\frac{1}{2}\overset{N}{\underset{n=1}{\sum}}\{t_n-w^T\phi(x_n)\}^2+\frac{1}{2}w^Tw$

    the close-formed solution for w is

    $w=(\lambda I+\phi^T\phi)^{-1}\phi^Tt$

### Model Complexity Issue

Bias-Variance Decomposition

## Linear Models for Classification

-   Regression: $x\rightarrow y\space or\space \overset{\rightarrow}{y}$

-   Classification: $x\rightarrow y=C_k,\space k=1,\ldots,K$

    for probabilistic models: 

    -   if K=2 the binary case, $y=1\rightarrow C_1$ and $y=0\rightarrow C_2$

    -   if K>2 the multiple cases, we can use 1-of-K coding scheme, $\overset{\rightarrow}{y}\in R^k,\space y_j=1$, if the class is $C_j$; otherwise 0;

        e.g.: $\overset{\rightarrow}{y}=(0,1,0,0,0)^T$

-   Discriminant function: directly assigns each vector x to a specific class

-   Probability inference: Modeling the conditional probability distribution $P(C_k|x)$ in an inference stage, then uses this to make optimal decision

### Generalized Linear Models

-   Linear classification models: $y(x)=w^Tx+w_0$

-   Generalized linear models: generally we want to predict posterior probabilities (0, 1)

    $y(x)=f(w^Tx+w_0)$

    These models are called generalized linear models

    -   f(.) is known as activation function
    -   the decision surfaces correspond to $y(x)=constant\rightarrow w^Tx_0+w_0=constant$
    -   the decision surfaces are linear functions of x, even if f(.) is nonlinear

    The input vector x is assigned according to 
    $$
    x\in  
    \begin{cases}
    C_1,\space if\space y(x)\le \space 0 \\
    C_2,\space otherwise
    \end{cases}
    $$

-   Problem setting

    -   Each class $C_k$ is described by its own linear model:

        $y_k(x)=w_k^T+w_0$

    -   Using vector notation: $y(x)=W^Tx$ by omitting the bias $w_0$, $W=[w_1,\ldots,w_k]$

    -   Considering a training dataset $\{x_n,t_n\}_{n=1}^N$ and $X=[x_1,\ldots,x_N]$,$T=[t_1,\ldots,t_N]$

    -   The sum-of squares error function can be written as $E_D(W)=\frac{1}{2}(XW-T)^T(XW-T)$

        $\Rightarrow W=(X^TX)^{-1}X^TT=X^+T$

        $\Rightarrow y(x)=W^Tx=T^T(X^+)^Tx$

-   Consider a two-class problem in which there are $N_1$ points of class $C_1$ and $N_2$ points of class $C_2$

-   The mean vectors in the original space:

    $m_k=\frac{1}{N_k}\underset{i\in C_k}{\sum}x_i$

    The means vectors in the projection spae with some projected direction w:

    $\mu_k=\frac{1}{N_k}\underset{i\in C_k}{\sum}w^Tx_i$

-   maximizing the between class variance and minimizing the within class variance is given by

    $J(w)=\frac{(\mu_2-\mu1)^2}{\sigma_1+\sigma_2}=\frac{w^TS_Bw}{w^TS_Ww}$

    $\ldots\Rightarrow w=S_W^{-1}(m_2-m_1)$

-   Proessing elements can be regarded as the basis functions of a generalized linear discriminant:

    $y(x)=f(w^T\phi(x))$

    The nonlinear activation function f(.) is given by a step function of the form
    $$
    f(a)=
    \begin{cases}
    +1,\space a\ge0 \\
    -1,\space a\lt0
    \end{cases}
    $$

    $$
    \begin{align}
    &x_i\Rightarrow \Phi(x_i)=\Phi_i,x_i\Rightarrow t_i \\
    &C_1:\space w^T\Phi_i\gt0;C2:\space w^T\Phi_i\lt0 \\
    &C_1:\space t_i=+1;C_2:\space t_i=-1
    \end{align}
    $$

    

    The error function of the perceptrons

    $E_P(w)=-\underset{\Phi_i\in M}{\sum}w^T\Phi_it_i$, where $M$ is the set of vectors $\Phi_i$ which are misclassified by the current weight vector $w$

-   If we apply the pattern-by-pattern gradient descent rule to the perceptron cirterion we obtain

    $w^{(\tau+1)}=w^{(\tau)}-\mu\nabla E_P(w)=w^{(\tau)}+\mu\phi(x_i)t_i$

    if correctly classified:

    $w^{(\tau+1)}=w^{(\tau)}$

    