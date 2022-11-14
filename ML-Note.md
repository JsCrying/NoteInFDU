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

    针对预测结果而言的，表示预测为正的样例中有多少是真正的正样例

     查准率$Precision=\frac{TP}{TP+FP}$

-   How many positives come back?

    针对原来的样本而言的，表示的是样本中的正例有多少被预测正确

    查全率$Recall=\frac{TP}{TP+FN}$

-   How many positives predicted are true positives and how many negatives predicted are true negatives?

    $Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$
    
-   F1值是查准率与查全率的调和平均

    $F1=\frac{2PR}{P+R}=\frac{2TP}{样例总数+TP-TN}$

-   BEP平衡点，是查准率=查全率时的取值。可以以查准率为纵轴，以查全率为横作图的到曲线，简称P-R曲线来求BEP

-   真正例率$TPR=\frac{TP}{TP+FN}$

-   假正例率$FPR=\frac{FP}{TN+FP}$

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

-   概率模型

    $\sigma(a)=\frac{1}{1+exp(-a)}$

    多个类：

    $P(C_k|x)=\frac{p(x|C_k)P(C_k)}{\sum_j p(x|C_j)P(C_j)}=\frac{exp(a_k)}{\sum_j exp(a_j)}$, where $a_k=\ln p(x|C_k)P(C_k)$

### Summary for Generative Models

1.  choice of class conditional densities $p(x|C_k)$
2.  using ML for the parameter estimation
3.  together with prior probability
4.  using Bayes' theorem, the posterior probabilities $P(C_k|x)$ are generalized linear function of $x\Rightarrow $ implicitly finding the parameters of a generalized linear model

## SVM

-   Suppose exists the optimal solution $(w^*,b^*)$, which defines a decision boundary corrctly classifying all the training samples, and every training sample is at least distance $\rho>0$ from the decision boundary, i.e.,

    $|f(x_i)|=|(w^*)^Tx_i+b^*|=\rho$

    $\rho$ plays a ceucial role for the percptron as it determines 

    (1)how well the two classes can be separated

    (2)how fast the perceptron learning algorithm convergess

    (3)$\rho$ call a $margin$

-   Definition: The hyperplane is in canonical form w.r.t. $X$ if $min_{x_i\in X}|w^Tx_i+b|=1$, $\rho=\frac{1}{2}(f(x^+)-f(x^-))=\frac{1}{||w||}$

-   $\underset{W}{min}\{||w||^2\},s.t.\forall_{i=1}^{n}:y_i(w^Tx_i+b)\geq1$

    using Lagrange theory:

    $L(w,b,\alpha)=\frac{1}{2}||w||^2-\overset{n}{\underset{i=1}{\sum}}\alpha_i(y_i(w^Tx_i+b)-1)$, $\alpha_i\geq0$

    $\Rightarrow w=\sum_{i=1}^{n}\alpha_iy_ix_i$

-   The decision function $f(x)=w^Tx+b=sgn[\overset{n}{\underset{i=1}{\sum}}\alpha_{i}^*y_ix_{i}^*+b^*]$

### 支持向量机基本型

-   超平面方程: $w^Tx+b=0$

-   最大间隔: 寻找参数$w,b$使得$\gamma$最大

    $\underset{w,b}{argmax}\frac{2}{||w||},\space s.t.\space y_i(w^Tx_i+b)\geq1,i=1,2,\ldots,m$

    $\underset{w,b}{argmin}\frac{1}{2}||w||^2,\space s.t.\space y_i(w^Tx_i+b)\geq1,i=1,2,\ldots,m$

-   最终模型: $f(x)=w^Tx+b=\sum_{i=1}^{m}\alpha_iy_ix_{i}^Tx+b$

-   KKT条件: 
    $$
    \begin{cases}
    \alpha_i\geq0 \\
    y_if(x_i)\geq1 \\
    \alpha_i(y_if(x_i)-1)=0 \\
    \end{cases}
    $$
    $y_i(f(x_i))\gt1\Rightarrow \alpha_i=0$

### 非线性SVM

-   预处理数据$\phi: \space X\rightarrow H,\space x\rightarrow \phi(x)$,  通常$dim(x)\ll dim(H)$

-   $L(\alpha)=\overset{n}{\underset{i=1}{\sum}}\alpha_i-\frac{1}{2}\overset{n}{\underset{i=1}{\sum}}\overset{n}{\underset{l=1}{\sum}}y_iy_l\alpha_i\alpha_lk(x_i,x_l)$
    $$
    s.t.
    \begin{cases}
    \alpha_i\geq0,\space 1\leq i\leq n \\
    \sum_{i=1}^ny_i\alpha_i=0
    \end{cases}
    $$
    solution:

    $f(x)=\overset{n}{\underset{i=1}{\sum}}\alpha_{i}^*y_ix^Tx_i+b^*$

    can be formulated as:

    $f(x)=\overset{n}{\underset{i=1}{\sum}}\alpha_{i}^*y_ik(x,x_i)+b^*$

    examples of common kernels used:

    Gaussian kernels: $k(x,x')=exp(-\frac{||x-x'||^2}{2\sigma^2})$

    Polynomial kernels: $k(x,x')=(x^Tx'+c)^d$

### Multi-Class Classification

-   $\underset{w^{lj},\xi{lj}}{min}\{\frac{1}{2}||w^{lj}||^2+C\overset{n}{\underset{i=1}{\sum}}\xi_i^{lj}\}$

    s.t.
    $$
    \begin{cases}
    <w^{lj},x_i>+b\geq 1-\xi_{i}^{lj},y_i=l \\
    <w^{lj},x_i>+b\leq -1+\xi_{i}^{lj},y_i=l,y_i=j \\
    \forall_{i=1}^n:\xi_{i}^{lj}\gt0,l,j=1,\ldots,C
    \end{cases}
    $$

### Primal SVM

-   primal objective function for the soft-margin SVMs with constraints is:

    $\frac{1}{2}||w||^2+C\overset{n}{\underset{i=1}{\sum}}\xi_i,s.t.\space \forall_{i=1}^n:y_i(w^Tx_i+b)\geq 1-\xi_i,\xi_i\geq0$

-   By the algebra operation, the constraints can be integrated in objective function such that the objective function without constraints is:

    $\frac{1}{2}||w||^2+C\overset{n}{\underset{i=1}{\sum}}V(y_i,w^Tx_i+b)$

    $V(y_i,w^Tx_i+b)$ is the loss for the training patterns $x_i\in X_l$ defined by $V(y,t)=max(0,1-yt)$ (called $hinge \space loss$)

-   Empirical risk functional

    $R_{emp}[f]=\frac{1}{n}\overset{n}{\underset{i=1}{\sum}}V(y_i,f(x_i))$

    for the classical regularization Networks:

    $V(y_i,f(x_i))=(y_i-f(x_i))^2$

    for the support vector classification:

    $V(y_i,f(x_i))=|1-y_if(x_i)|_+,where\space |t|_+=\begin{cases}t,\space t\geq0 \\0,\space t\lt0\end{cases}$

-   if the kernel exists, we can define the mapping 

    $\phi(x)=k(\cdot,x)$

    by the Representer theorem, we have

    $f(\cdot)=\overset{n}{\underset{i=1}{\sum}}\beta_ik(\cdot,x_i)$

    and with the reproducing property, we have

    $f(x)=<f(\cdot),k(\cdot,x)>=\overset{n}{\underset{i=1}{\sum}}\beta_ik(x,x_i)$

### Summary: Steps for Classification

1.  prepare the pattern matrix

2.  select the kernel function to use

3.  select the parameter of the kernel function and the value of C

    use the value suggested by the SVM software, or you can set apart a validation set to determine the value of the parameter

4.  execute the training algorithm and obtain the $a_i$

5.  unseen data can be classified using the $a_i$ and the support vectors

### SVR

-   Linear Support Vector Regression

    $y=w_1x+b$

    $|y_i-f(x_i)|_\epsilon\equiv max\{0,|y_i-f(x_i)|-\epsilon\}$

-   Non-linear Support Vector Regression
    $$
    x\rightarrow \Phi(x)=(\sqrt x,\sqrt 2x^2) \\
    Age\rightarrow \Phi(Age) \\
    Age\rightarrow (\sqrt {Age},\sqrt 2Age^{2})
    $$
    $y=w_1x+b$

    $y=w_1\sqrt x+w_2\sqrt 2x^2+b$

-   Given training data $\{x_i,y_i\}_{i=1}^{n}$

    Find: $w_1, b$ such that $y=w_1x+b$ optimally describes the data

    $\underset{Complexity}{|w_1|} vs.\space \underset{Sum \space of\space errors}{\sum_i(\xi_i+\xi_{i}^*)}$

    $\underset{w_1,b,\xi_i,\xi_{i}^*}{min}\frac{1}{2}w_1^2+C\underset{i}{\sum}(\xi_i+\xi_{i}^*)$

    subect to:

    $y_i-(w_1x_{i1})-b\leq\epsilon+\xi_i$

    $w_1x_{i1}+b-y_i\leq\epsilon+\xi_{i}^*$

    $\xi_i,\xi_{i}^*\geq0\space i=1,2,\ldots,n$

    $L:=\frac{1}{2}||w||^2+C\underset{i}{\sum}(\xi_i+\xi_{i}^*)-\underset{i}{\sum}(\eta_i\xi_i+\eta_{i}^*\xi_{i}^*)\\-\underset{i}{\sum}\alpha_i(\epsilon+\xi_i-y_i+w'\phi(x_i)+b)\\ -\underset{i}{\sum}\alpha_{i}^{*}(\epsilon+\xi_{i}^*+y_i-w'\phi(x_i)-b)$
    $$
    \begin{align}
    &\Rightarrow \\
    &\frac{\partial L}{\partial w}=w-\sum_i(\alpha_i\alpha_{i}^*\phi(x_i))=0 \\
    &\ldots \\
    &f(x)=w'\phi(x)+b \\
    &f(x)=\sum_i(\alpha_i-\alpha_{i}^*)(\phi(x_i)'\phi(x))+b \\
    &f(x)=\sum_i(\alpha_i-\alpha_{i}^*)k(x_i,x)+b
    \end{align}
    $$
    
    
    

## 习题

### 模型评估与选择

1.  数据集包含 1000 个样本，其中 500 个正例、 500 个反例，将其划分为包含 70% 样本的训练集和 30% 样本的测试集用于留出法评估，估算有多少种划分方式

    解答：训练集/测试集的划分要尽可能保持数据分布一致

    故训练集中应该包含350个正例、350个反例，剩余的作测试集，那么划分方式有$(C_{500}^{350})^2$

2.  若学习器A的F1值比学习器B高，试分析A的BEP值是否也比B高

    解答：$F_1=\frac{2PR}{P+R}$，$P=\frac{TP}{TP+FP}$，$R=\frac{TP}{TP+FN}$

    BEP是 $P=R$ 时取到的值

    $F_{1A}\gt F_{1B}\Leftrightarrow \frac{1}{P_{1A}}+\frac{1}{R_{1A}}\lt \frac{1}{P_{1B}}+\frac{1}{R_{1B}}$

    $BEP_A=P_{1A}=R_{1A}$，$BEP_B=P_{1B}=R_{1B}$

    $\Rightarrow BEP_A\gt BEP_B$

3.  试阐述真正例率、假正例率与查准率、查全率之间的联系

    解答：

    查全率：$R=\frac{TP}{TP+FN}$

    查准率：$P=\frac{TP}{TP+FP}$

    真正例率：$TPR=\frac{TP}{TP+FN}$，同查全率

    假正例率：$FPR=\frac{FP}{FP+TN}$，即所有反例中被预测为正例的比率

### 线性模型

1.  试着分析在什么情况下，在以下式子$f(x)=w^Tx+b$中不考虑偏置项b

    解答：在样本$x$中有某一个属性为$x_i$为固定值时。此时$w_ix_i+b$等价于偏置项

2.  证明：对于参数$w$，对率回归的目标函数$y=\frac{1}{e^{-(w^Tx+b)}}$是非凸的，但其对数似然函数$l(\beta)=\sum_{i=1}^m(-y_i\beta ^T\overline{x}_i+\ln(1+e^{\beta ^T\overline{x}_i}))$是凸的

    解答：

    引理：对于多元函数，其Hessian Matrix为半正定即为凸函数

    辅助：

    梯度下降法、牛顿法

    有矩阵$A$和向量$x$

    $\frac{\partial Ax}{\partial x}=A^{T}$

    $\frac{\partial Ax}{\partial x^T}=A$

    $\frac{\partial (x^TA)}{\partial x}=A$

    目标函数：$\frac{\partial y}{\partial w}=x(y-y^2)$

    $\frac{\partial ^2y}{\partial w\partial w^T}=xx^Ty(1-y)(1-2y)$，即为Hessian Matrix

    $r(xx^T)=1\Rightarrow 非零特征值只有1个$，标准化以后，二次型的值由$y$决定。当$y$在$(0,1)$之间变化时$y(1-y)(1-2y)$正负号变化，故矩阵并非半正定，因此非凸

    对数似然函数：$\frac{\partial ^2l(\beta)}{\partial \beta \partial \beta^{T}}=\sum_{i=1}^m\overline{x}_i\overline{x}_{i}^Tp_1(\overline{x}_i;\beta)(1-p1(\overline{x}_i;\beta))$

    $\Rightarrow P_{ii}=p_1(\overline{x}_i;\beta)(1-p1(\overline{x}_i;\beta))\geq 0$，$\underset{i\neq j}{P_{ij}=0}$

    二次型恒>=0，因此矩阵为半正定，为凸

### SVM

1.  试证明样本空间中任意点$x$到超平面$(w,b)$的距离为式$r=\frac{|w^Tx+b|}{||w||}$

    解答：

    对于任意点$A$，作$A$到超平面的投影$B$，将距离记为$r$。

    $\overline{BA}=r*\frac{w}{||w||}$

    $w^T\overline{B}+b=0$

    $\overline{B}=\overline{A}-\overline{BA}$

    $\Rightarrow w^T(\overline{A}-r*\frac{w}{||w||})+b=0$

    $\Rightarrow r=\frac{|w^T\overline{A}+b|}{||w||}$

2.  试给出式$\begin{cases}\alpha_i(f(x_i)-y_i-\epsilon-\xi_i)=0 \\ \overline{\alpha_i}(-f(x_i)+y_i-\epsilon-\overline{\xi_i})=0 \\ \alpha_i\overline{\alpha_i}=0，\xi_i\overline{\xi_i}=0 \\ (C-\alpha_i)\xi_i=0，(C-\overline{\alpha_i})\overline{\xi_i}=0\end{cases}$的完整KKT条件

    解答：

### 贝叶斯分类

1.  
