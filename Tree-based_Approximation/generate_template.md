第三类：Tree-based Approximation
数学基础：递归空间划分（Recursive Space Partition）。包括 Decision Tree、Extra Trees、Random Forest、Gradient Boosting。
模板：
# 方法名称
---
# 1. Historical Motivation（历史背景）
为什么提出 Tree。
为什么不用线性模型。
为什么不用 Kernel。
Tree 能解决什么问题。
---
# 2. Mathematical Motivation（数学动机）
这一章重点解释：
为什么可以把函数逼近转化成空间划分。
从
f(x)
↓
Partition
↓
Piecewise Constant
逐步推导。
---
# 3. Mathematical Foundations（数学基础）
这是整个 Tree 最重要部分。
---
## 3.1 Recursive Space Partition
为什么不断切空间。
数学定义。
超矩形。
Region。
---
## 3.2 Piecewise Constant Approximation
证明 Tree 本质就是：
f(x)=ΣciI(x∈Ri)
为什么成立。
---
## 3.3 Piecewise Linear Approximation
为什么有些 Tree 可以线性。
什么时候变成 Piecewise Linear。
---
## 3.4 Empirical Risk Minimization
Tree 为什么选择最优划分。
ERM 推导。
---
# 4. Mathematical Definition（数学定义）
定义：
Node
Leaf
Split
Depth
Region
Impurity
全部数学符号说明。
---
# 5. Split Criterion（划分准则）
这是最重要章节。
每一种全部推导。
Regression
Variance Reduction
Classification
Entropy
Information Gain
Gini
全部公式推导。
---
# 6. Optimization（训练）
Tree 如何长出来。
Greedy
为什么 NP-Hard。
为什么只能 Greedy。
剪枝。
Cost Complexity。
全部推导。
---
# 7. Approximation Theory（逼近能力）
为什么 Piecewise Constant 可以逼近任意函数。
一致收敛。
逼近误差。
Bias。
Variance。
VC Dimension。
---
# 8. Ensemble Theory（集成理论）
Extra Tree
Random Forest
Gradient Boosting
分别推导。
为什么 Bagging 降低方差。
为什么 Boosting 降低 Bias。
完整数学解释。
---
# 9. Computational Complexity
建树复杂度。
预测复杂度。
内存复杂度。
并行复杂度。
---
# 10. RL中的应用
Fitted Q Iteration
Extra Tree RL
Tree Backup
Decision Tree Policy
Model Tree
全部数学公式。
---
# 11. Convergence Theory
Tree FQI。
Error Propagation。
Bellman Backup。
Approximation Error。
Estimation Error。
---
# 12. Advantages
---
# 13. Disadvantages
---
# 14. Comparison
Linear
Kernel
MLP
Boosting
全部比较。
---
# 15. Applications
Control
Offline RL
Robot
Medical
Finance
---
# 16. Classic Papers
按年代整理。
---
# 17. References
