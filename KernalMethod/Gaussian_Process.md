# Gaussian Process（高斯过程回归）
---
# 1. Historical Motivation（历史背景）

## 1.1 为什么提出 Kernel Method

参见前述核方法文档的一般动机。Gaussian Process（GP）由地质统计学中的 Kriging 方法（Krige, 1951）发展而来，1990年代被 Rasmussen、Williams 等人系统引入机器学习领域。与 KRR/SVR 的"点估计"视角不同，GP 的核心动机是把函数逼近问题彻底**贝叶斯化**——不是求一个最优的 f̂，而是对所有可能的函数 f 维护一个完整的概率分布，逼近结果自带不确定性量化。

## 1.2 Linear Basis 的局限

同前述文档：人工设计基函数、维数灾难、局部泛化差。GP 额外解决的问题是：确定性的函数逼近方法（包括KRR/SVR）在训练数据稀疏的区域给出一个点估计，但**不知道这个估计有多可信**；在RL的探索问题中，"知道自己不知道什么"（认知不确定性）恰恰是指导探索的关键信息，这是GP相比其他核方法在RL中格外重要的原因。

## 1.3 Kernel 的核心思想

同前述文档，样本作为基函数。GP 的特殊之处：核函数 K(x,z) 不再只是"相似度度量"，而被赋予了明确的概率含义——**协方差函数**，K(x,z)=Cov[f(x),f(z)]，直接定义了函数值之间的相关性结构。

---
# 2. Mathematical Motivation（数学动机）

## 2.1 从线性模型开始

贝叶斯线性回归：f(x)=wᵀφ(x)，对 w 赋予高斯先验 w~N(0,Σ_p)。**为什么需要 Feature Mapping**：与其他核方法相同的理由——线性模型表达能力不足以捕捉非线性关系。

## 2.2 Feature Space

给定先验 w~N(0,Σ_p)，f(x)=wᵀφ(x) 在任意有限点集 {x₁,...,xₙ} 上的取值联合服从多元高斯分布（高斯分布的线性变换仍是高斯分布）：

```
[f(x₁),...,f(xₙ)] ~ N(0, ΦΣ_pΦᵀ)
```

**几何解释**：这意味着"函数"本身可以看作一个无穷维随机变量，任意有限维"切片"都服从多元高斯——这就是"高斯过程"的定义性质。

## 2.3 Infinite-dimensional Feature Space

**为什么可以无限维**：协方差矩阵 ΦΣ_pΦᵀ 的每个元素 (i,j) 等于 φ(xᵢ)ᵀΣ_pφ(xⱼ)，同样只涉及内积形式，可以直接用核函数 K(xᵢ,xⱼ) 替代，φ 可以是无穷维（如高斯核）。

**Gaussian Kernel 例子**：用高斯核作为协方差函数 K(x,z)=σ_f²exp(−||x−z||²/2ℓ²)，等价于假设先验函数 f 是从一个"处处无穷阶可微、光滑度由ℓ控制"的无穷维高斯分布中采样得到的。

## 2.4 Kernel Trick

**完整推导**：高斯过程定义为——对任意有限点集 {x₁,...,xₙ}，f(x₁),...,f(xₙ) 的联合分布是均值为 m(x)（通常设为0）、协方差为 K(xᵢ,xⱼ) 的多元高斯分布：

```
f(X) ~ N(m(X), K(X,X))
```

**为什么可以不用知道φ**：GP 的定义直接建立在核函数（协方差函数）之上，完全不需要显式构造 φ，这是"函数空间视角"（function-space view）与"权重空间视角"（weight-space view，即2.1-2.3节的推导路径）在数学上等价的体现——Kernel Trick 在GP中不是"技巧"，而是定义本身。

---
# 3. Mathematical Foundations（数学基础）

## 3.1 Hilbert Space
同前述文档。GP 的样本路径（sample path）几乎必然落在对应核函数的 RKHS 的某个稠密子集之外（这是一个精细的技术性结论——GP先验样本路径通常比RKHS中的函数"更粗糙"），但GP后验均值恰好落在RKHS中，这是GP与KRR联系的关键。

## 3.2 Reproducing Kernel Hilbert Space（RKHS）
同前述文档定义与再生性质证明。GP 的协方差函数 K 与其对应的 RKHS 共享同一个核函数。

## 3.3 Mercer Theorem
同前述文档。GP 的协方差函数通过 Mercer 展开 K(x,z)=Σᵢλᵢψᵢ(x)ψᵢ(z) 可以等价地表示为无穷个独立高斯随机变量 wᵢ~N(0,λᵢ) 加权基函数 ψᵢ(x) 的叠加：f(x)=Σᵢ√λᵢwᵢψᵢ(x)，这是"函数空间视角"与"权重空间视角"等价性的Mercer定理证明。

## 3.4 Representer Theorem

**为什么最优解一定表示成 f(x)=Σᵢ αᵢ K(x,xᵢ)**：GP 回归的后验均值函数可以通过贝叶斯公式直接推导（见下），其结果恰好与 Representer Theorem 给出的形式一致——这并非巧合，GP 后验均值等价于在RKHS中最小化"负对数似然+RKHS范数正则化"的MAP估计，满足 Representer Theorem 的一般条件。

**GP后验均值的完整推导**：设观测 y=f(X)+ε，ε~N(0,σ_n²I)，先验 f(X)~N(0,K(X,X))。由高斯分布的联合-条件公式，测试点 x* 处的后验分布为：

```
f(x*)|y ~ N(μ*, σ*²)
μ* = k(x*)ᵀ(K+σ_n²I)⁻¹y
σ*² = K(x*,x*) − k(x*)ᵀ(K+σ_n²I)⁻¹k(x*)
```

其中 μ* = k(x*)ᵀα，α=(K+σ_n²I)⁻¹y——与 KRR 的解（λ=σ_n²时）**形式完全相同**，验证了 Representer Theorem 的结论，也说明了 GP 后验均值与 KRR 点估计在数学上的等价关系。

---
# 4. Mathematical Definition（数学定义）

统一数学形式：

```
f(x) ~ GP(m(x), K(x,x'))
```

后验预测分布：

```
μ*(x) = k(x)ᵀ(K+σ_n²I)⁻¹y
σ*²(x) = K(x,x) − k(x)ᵀ(K+σ_n²I)⁻¹k(x)
```

**全部符号解释**：
- m(x)：均值函数（先验均值，通常设为0）
- K(x,x')：协方差函数（核函数），Cov[f(x),f(x')]
- σ_n²：观测噪声方差
- K：训练点间的Gram/协方差矩阵
- k(x)：测试点与训练点间的协方差向量
- μ*(x)：后验预测均值（点估计）
- σ*²(x)：**后验预测方差——GP相比其他核方法独有的不确定性量化**

---
# 5. Kernel Function（核函数）

同前述文档：核函数（此处称"协方差函数"）需满足对称性、半正定性；常见核函数（Linear/Polynomial/Gaussian/Laplacian/Sigmoid/Matérn）定义、几何意义与适用场景同前述文档，不重复展开。**GP领域尤其偏好Matérn核**，因为其光滑度参数ν有明确的物理解释（对应函数的可微阶数），比高斯核（对应无穷阶可微、可能过度光滑）更适合真实物理过程建模。

---
# 6. Approximation Formula（逼近公式）

统一写成：

```
f̂(x) = μ*(x) = Σᵢ αᵢ K(x,xᵢ)，α=(K+σ_n²I)⁻¹y
```

**逐步推导**：见第3.4节贝叶斯后验推导。

**为什么样本就是 Basis**：同前述文档，但GP额外提供了对"基函数组合是否可信"的定量描述（后验方差）。

**为什么属于 Non-parametric**：GP 是"非参数贝叶斯方法"的代表——先验直接定义在无穷维函数空间上，没有固定数量的参数，模型复杂度随数据自适应确定。

---
# 7. Optimization（优化）

**Least Squares**：GP后验均值公式在数学上与最小二乘/KRR解一致（见第3.4节）。

**Regularization / Kernel Ridge**：GP中的"正则化"体现为观测噪声方差σ_n²（等价于KRR中的λ），是贝叶斯框架下自然出现的量，而非人为添加的惩罚项。

**Dual Problem / Lagrange**：GP的推导路径与SVR不同，不涉及不等式约束和KKT条件，而是通过条件高斯分布公式直接得到闭式解，不需要对偶化技巧。

**超参数学习（GP特有）**：核函数的超参数（如带宽ℓ、信号方差σ_f²、噪声方差σ_n²）通过最大化**边际似然（marginal likelihood）**学习：

```
log p(y|X) = −(1/2)yᵀ(K+σ_n²I)⁻¹y − (1/2)log|K+σ_n²I| − (N/2)log(2π)
```

该目标函数自动在"数据拟合程度"（第一项）和"模型复杂度惩罚"（第二项，Occam剃刀效应）之间权衡，通常用梯度上升求解——这是GP相比KRR/SVR最独特的优势：**超参数选择内嵌在概率框架中，不需要单独的交叉验证**。

---
# 8. Approximation Theory（逼近理论）

**Universal Kernel**：同前述文档，高斯核等Universal Kernel保证GP先验支撑集在紧集上稠密于连续函数空间。

**Consistency**：在核函数选择合适、噪声方差正确指定的条件下，GP后验均值随样本量N→∞依概率收敛到真实函数，后验方差同时收缩到0（贝叶斯一致性）。

**Approximation Error / Generalization Error / Bias-Variance**：与KRR共享同样的Bias-Variance分析（因为均值公式相同），但GP额外提供了**校准的（calibrated）不确定性**——如果模型假设（核函数、噪声水平）设定合理，后验方差可以直接解释为预测误差的置信区间宽度，这是GP相较于纯频率学派方法（KRR/SVR）在"泛化误差的可解释性"上的独特优势。

---
# 9. Computational Complexity（计算复杂度）

- **Kernel Matrix**：O(N²) 存储
- **训练复杂度**：需要计算 (K+σ_n²I)⁻¹（用于均值和边际似然）以及行列式 |K+σ_n²I|（用于边际似然），标准做法是 Cholesky 分解，复杂度 **O(N³)**
- **预测复杂度**：均值 O(N)，方差 O(N²)（需要计算 k(x)ᵀ(K+σ_n²I)⁻¹k(x)）
- **内存复杂度**：O(N²)

**为什么GP是O(N³)**：Cholesky分解求解 (K+σ_n²I)⁻¹ 以及计算对数行列式（边际似然优化每一步都需要）的标准复杂度就是 O(N³)——这是GP区别于KRR（虽然公式相同但若只需均值预测、不需要边际似然优化，可以用迭代方法避免显式O(N³)分解）在实践中计算负担更重的原因，因为超参数学习需要反复计算行列式。

**如何降低复杂度**：
- **Nyström方法**：低秩近似Gram矩阵，降至O(NM²)
- **Random Features**：用有限维随机特征近似GP先验，转化为贝叶斯线性回归
- **Sparse GP（稀疏高斯过程）**：引入 M≪N 个诱导点（inducing points）u=f(Z)，用变分推断（Titsias, 2009）或FITC近似（Snelson & Ghahramani, 2006）逼近完整GP后验，将复杂度降至 **O(NM²)**，是当前处理大规模GP的标准方法

---
# 10. RL中的应用

**Gaussian Process RL（GP-RL）**：用GP对价值函数、Q函数或环境动力学模型 P(s'|s,a) 做贝叶斯建模，后验方差直接用于指导探索（如置信上界策略）。

**Kernel Bellman Equation（GP版本，数学推导）**：GTD/GP结合的方法中，把贝尔曼方程 V=R+γPV 中的 V 建模为GP先验，通过贝叶斯滤波/最小二乘的方式递推更新后验（GP-SARSA, Engel et al. 2005）：

```
GP-SARSA: Q(s,a) ~ GP(0,K)，后验通过在线Bayesian filtering递推更新均值与协方差
```

**PILCO（Deisenroth & Rasmussen, 2011）**：用GP学习环境动力学模型（状态转移函数），并通过对GP预测分布做解析的矩匹配（moment matching），将模型不确定性显式传播到多步预测和策略梯度计算中，是样本效率极高的经典 model-based RL 方法，在真实机器人任务上表现突出（因为GP在小样本区域给出准确的不确定性，避免了模型误差累积导致的策略退化）。

**GP-UCB探索策略**：利用GP后验均值和方差构造置信上界 UCB(x)=μ(x)+βσ(x)，是贝叶斯优化和GP-RL中最经典的探索机制，有理论后悔界（regret bound）保证。

---
# 11. Convergence Theory（收敛理论）

**Kernel TD是否收敛**：GP版本的TD/SARSA（如GP-SARSA）在理论上继承核化TD的一般收敛性分析框架（第9篇Kernel Regression文档第11节），额外需要考虑贝叶斯递推更新（而非梯度下降）带来的不同误差传播路径。

**Kernel Bellman Projection / Projected Fixed Point**：GP-RL中，后验均值的更新同样可以看作向 span{K(·,sᵢ)} 子空间的投影，与线性/核化TD的投影贝尔曼方程理论一致。

**误差界**：GP-RL 独特之处在于可以给出**贝叶斯遗憾界（Bayesian regret bound）**，直接利用后验方差的收缩速度界定累积遗憾，这是GP-UCB等算法理论分析的标准工具，是其他核方法（KRR/SVR）在RL中不具备的理论优势。

---
# 12. Advantages

- 天然提供校准的不确定性估计，直接用于指导探索
- 超参数学习内嵌在边际似然最大化框架中，无需单独交叉验证
- 理论完备（贝叶斯非参数统计、RKHS理论、Bayesian regret理论三者统一）
- 小样本下表现出色（这是PILCO等方法在真实机器人任务中样本效率极高的关键原因）

---
# 13. Disadvantages

- O(N³)训练复杂度，是所有核方法中扩展性最差的一类，大规模数据下必须依赖稀疏近似
- 核函数（协方差函数）及其超参数假设错误时，后验不确定性可能失准（过度自信或过度保守）
- 高斯假设本身可能不适合有明显非高斯噪声或多峰分布的场景

---
# 14. Comparison

| 方法 | 是否贝叶斯 | 是否提供不确定性 | 训练复杂度 | 超参数选择方式 |
|---|---|---|---|---|
| Gaussian Process | 是 | 是（校准的） | O(N³) | 边际似然最大化 |
| KRR | 否（点估计，虽公式同GP均值） | 否 | O(N³) | 交叉验证 |
| SVR | 否 | 否 | O(N²)~O(N³)（QP） | 交叉验证 |
| Bayesian Neural Network | 是 | 是（近似，通常非精确） | 训练高，近似推断 | 变分/采样方法 |

---
# 15. Applications

- **机器人**：PILCO 及其变种，用于低样本量下的机器人运动控制学习
- **控制**：GP-based MPC（模型预测控制），利用GP不确定性做鲁棒控制
- **Bayesian RL / GP-RL**：GP-SARSA、GP-TD，探索-利用权衡的经典理论测试平台
- 贝叶斯优化（超参数调优）——虽非RL本身，但与GP-RL共享核心数学工具

---
# 16. Classic Papers（按时间排序）

- Krige, D. G. (1951). *A Statistical Approach to Some Basic Mine Valuation Problems*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*.
- Engel, Y., Mannor, S., & Meir, R. (2005). *Reinforcement Learning with Gaussian Processes*. ICML.
- Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NeurIPS.
- Deisenroth, M. P., & Rasmussen, C. E. (2011). *PILCO: A Model-Based and Data-Efficient Approach to Policy Search*. ICML.
- Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. AISTATS.

---
# 17. References

- Rasmussen, C. E., & Williams, C. K. I. *Gaussian Processes for Machine Learning*（GP领域权威教材，全书免费在线可读）
- Deisenroth, M. P. *Efficient Reinforcement Learning using Gaussian Processes*（PhD论文，PILCO完整推导）
