第二类：Kernel Methods
数学基础：RKHS。包括 Kernel Regression、Kernel Ridge Regression、Support Vector Regression、Gaussian Process。

模板：
# 方法名称
---
# 1. Historical Motivation（历史背景）
## 1.1 为什么提出 Kernel Method
## 1.2 Linear Basis 的局限
- Basis需要人工设计
- 高维维数爆炸
- 局部泛化能力差
## 1.3 Kernel 的核心思想
为什么不用设计 Basis，而直接利用样本作为 Basis。
---
# 2. Mathematical Motivation（数学动机）
这一章回答：
为什么会想到 Kernel？
## 2.1 从线性模型开始
推导
f(x)=w^Tφ(x)
为什么需要 Feature Mapping。
---
## 2.2 Feature Space
定义
φ:X→H
为什么映射以后变得线性。
几何解释。
---
## 2.3 Infinite-dimensional Feature Space
为什么可以无限维。
举 Gaussian Kernel 例子。
---
## 2.4 Kernel Trick
完整推导
φ(x)^Tφ(z)
↓
K(x,z)
为什么可以不用知道φ。
证明 Kernel Trick。
---
# 3. Mathematical Foundations（数学基础）
这一章要求完整推导。
---
## 3.1 Hilbert Space
定义
Inner Product
Norm
Projection
---
## 3.2 Reproducing Kernel Hilbert Space（RKHS）
为什么叫 RKHS。
完整定义。
Reproducing Property
逐步证明。
---
## 3.3 Mercer Theorem
为什么 Kernel 一定存在 Feature Space。
Mercer展开。
逐步推导。
---
## 3.4 Representer Theorem
这是 Kernel Method 最重要定理。
要求完整证明思路。
为什么最优解一定表示成
f(x)=ΣαiK(x,xi)
---
# 4. Mathematical Definition（数学定义）
统一数学形式
定义
Kernel Function
Kernel Matrix
Gram Matrix
全部符号解释。
---
# 5. Kernel Function（核函数）
为什么叫 Kernel。
核函数需要满足什么性质。
Positive Semi-definite
Symmetric
证明。
---
## 常见 Kernel
Linear
Polynomial
Gaussian
Laplacian
Sigmoid
Matérn
每一个给：
定义
几何意义
适用场景
---
# 6. Approximation Formula（逼近公式）
统一写成
f(x)=ΣαiK(x,xi)
逐步推导。
为什么样本就是 Basis。
为什么属于 Non-parametric。
---
# 7. Optimization（优化）
如何学习 α。
Least Squares
Regularization
Kernel Ridge
Dual Problem
Lagrange
完整推导。
---
# 8. Approximation Theory（逼近理论）
为什么 Kernel 可以逼近任意连续函数。
Universal Kernel
Consistency
Approximation Error
Generalization Error
Bias-Variance
全部推导。
---
# 9. Computational Complexity（计算复杂度）
Kernel Matrix
训练复杂度
预测复杂度
内存复杂度
为什么 O(N²)
为什么 GP 是 O(N³)
如何降低复杂度。
Nyström
Random Features
Sparse GP
---
# 10. RL中的应用
Kernel TD
Kernel SARSA
Kernel LSPI
Gaussian Process RL
Kernel Bellman Equation
数学推导。
---
# 11. Convergence Theory（收敛理论）
Kernel TD 是否收敛。
Kernel Bellman Projection。
Projected Fixed Point。
误差界。
---
# 12. Advantages
---
# 13. Disadvantages
---
# 14. Comparison
与
Linear Basis
Tree
Neural Network
比较。
---
# 15. Applications
机器人
控制
Bayesian RL
GP-RL
---
# 16. Classic Papers
按时间排序。
---
# 17. References
