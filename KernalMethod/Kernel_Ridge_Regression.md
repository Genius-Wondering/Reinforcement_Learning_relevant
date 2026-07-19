# Kernel Ridge Regression（核岭回归）
---
# 1. Historical Motivation（历史背景）

## 1.1 为什么提出 Kernel Method

参见《Kernel Regression》文档第1.1节——核方法的共同出发点是让训练样本本身充当基函数，避免人工设计特征。Kernel Ridge Regression（KRR）在此基础上，进一步解决了纯核回归/核插值在训练数据有噪声时容易**过拟合**的问题：Nadaraya-Watson 核回归是无正则化的局部加权平均，噪声敏感；KRR 把岭回归（Ridge Regression）的 L2 正则化思想与核方法结合，是"正则化 + 核化"的标准范例。

## 1.2 Linear Basis 的局限

同 Kernel Regression 文档第1.2节：人工设计基函数、高维维数爆炸、局部泛化能力差。KRR 额外要解决的问题是：普通线性岭回归在特征维度很高（甚至无穷维）时，权重向量 w 的维度也很高，直接优化不可行——需要通过核化把优化问题的维度从"特征维度"降到"样本数量"。

## 1.3 Kernel 的核心思想

同 Kernel Regression 文档第1.3节。KRR 的特殊之处在于，它是**岭回归的对偶形式**——通过对偶变换，w 的显式表示被替换为训练样本核函数的线性组合，即便原始特征空间是无穷维，对偶问题的变量数也只有 N 个。

---
# 2. Mathematical Motivation（数学动机）

## 2.1 从线性模型开始

标准岭回归：

```
min_w ||y − Φw||² + λ||w||²
```

其中 Φ 是设计矩阵（第i行为 φ(xᵢ)ᵀ）。闭式解为 w = (ΦᵀΦ+λI)⁻¹Φᵀy。当 φ(x) 维度 D 很高（甚至无穷维）时，ΦᵀΦ 是 D×D 矩阵，求逆不可行——这是需要核化的直接动机。

## 2.2 Feature Space

同 Kernel Regression 文档 2.2 节。KRR 的关键洞察是：可以用**对偶形式**改写岭回归的解，使其只涉及 Φ Φᵀ（一个 N×N 矩阵，即Gram矩阵），而非 ΦᵀΦ（D×D矩阵）。

## 2.3 Infinite-dimensional Feature Space

同 Kernel Regression 文档 2.3 节，高斯核对应无穷维特征空间；KRR 通过对偶化，使得无论 φ 维度是有限还是无穷，优化问题的规模都只取决于样本数 N。

## 2.4 Kernel Trick

**从原始（primal）到对偶（dual）的完整推导**：

由矩阵恒等式（Woodbury恒等式的特例），岭回归的解可以等价地写成：

```
w = Φᵀ(ΦΦᵀ+λI)⁻¹y
```

（证明：w=(ΦᵀΦ+λI)⁻¹Φᵀy，两边左乘(ΦᵀΦ+λI)，右乘(ΦΦᵀ+λI)⁻¹展开可验证恒等式成立）

预测值：

```
f̂(x) = φ(x)ᵀw = φ(x)ᵀΦᵀ(ΦΦᵀ+λI)⁻¹y = k(x)ᵀ(K+λI)⁻¹y
```

其中 k(x)=[K(x,x₁),...,K(x,xₙ)]ᵀ，K=ΦΦᵀ 即Gram矩阵（K_{ij}=φ(xᵢ)ᵀφ(xⱼ)=K(xᵢ,xⱼ)）。**全程只出现内积形式 φ(xᵢ)ᵀφ(xⱼ)，可直接替换为核函数 K(xᵢ,xⱼ)，无需显式知道 φ**——这就是 KRR 中 Kernel Trick 的完整应用。

---
# 3. Mathematical Foundations（数学基础）

## 3.1 Hilbert Space
同 Kernel Regression 文档 3.1 节，KRR 的假设空间同样是 RKHS。

## 3.2 Reproducing Kernel Hilbert Space（RKHS）
同 Kernel Regression 文档 3.2 节完整定义与证明。

## 3.3 Mercer Theorem
同 Kernel Regression 文档 3.3 节完整推导。

## 3.4 Representer Theorem

KRR 是 Representer Theorem 最直接的应用案例：损失函数 ||y−f(X)||² 只依赖训练点上的取值，正则化项 λ||f||²_H 是范数的单调函数，完全满足第3.4节证明的条件形式，因此最优解必然形如 f(x)=Σᵢαᵢ K(x,xᵢ)。**KRR 可以视为 Representer Theorem 应用于"平方损失+L2正则"这一特定组合下的显式解**（这也是为什么 KRR 有闭式解，而更一般的核方法如 SVR 需要求解二次规划）。

---
# 4. Mathematical Definition（数学定义）

统一数学形式：

```
f̂(x) = Σᵢ αᵢ K(x,xᵢ) = k(x)ᵀα，α = (K+λI)⁻¹y
```

**全部符号解释**：
- K：N×N Gram矩阵，K_{ij}=K(xᵢ,xⱼ)
- k(x)：N维向量，k(x)ᵢ=K(x,xᵢ)，测试点与所有训练点的核函数值
- λ：正则化系数（岭参数），控制模型复杂度
- α：对偶系数向量，α=(K+λI)⁻¹y

---
# 5. Kernel Function（核函数）

同 Kernel Regression 文档第5节：核函数需满足对称性与半正定性，常见核函数（Linear/Polynomial/Gaussian/Laplacian/Sigmoid/Matérn）的定义、几何意义、适用场景完全通用，不再重复展开。KRR 中最常用高斯核，因其对应无穷维特征空间且数值行为稳定（正定性严格保证 K+λI 可逆）。

---
# 6. Approximation Formula（逼近公式）

统一写成：

```
f(x) = Σᵢ αᵢ K(x,xᵢ)，α = (K+λI)⁻¹y
```

**逐步推导**：见第2.4节的原始-对偶转换推导。

**为什么样本就是 Basis**：同 Kernel Regression 文档第6节。

**为什么属于 Non-parametric**：KRR 的"参数"α 的数量等于训练样本数 N，随数据增长而增长；但注意 KRR 相比纯核回归多了一个**正则化系数 λ**——这是唯一固定的（不随N增长的）超参数，用于控制有效模型复杂度，防止插值过拟合。

---
# 7. Optimization（优化）

**Least Squares**：KRR 的目标函数本身就是（正则化的）最小二乘问题。

**Regularization / Kernel Ridge**：目标函数

```
J(α) = ||y−Kα||² + λαᵀKα
```

对 α 求梯度并令其为0：

```
∂J/∂α = −2K(y−Kα) + 2λKα = 0
⟹ K(y−Kα−λα) = 0
⟹ Ky = K(K+λI)α （在K的值域内成立）
⟹ α = (K+λI)⁻¹y
```

**Dual Problem / Lagrange**：KRR 本身已经是对偶形式的显式解，不需要像 SVR 那样引入不等式约束和 Lagrange 乘子；但可以从"原始"岭回归问题出发，通过拉格朗日对偶的一般框架重新导出上述解（对偶变量α与原始问题中残差 y−Φw 的关系为 α=(y−Φw)/λ），这一视角有助于理解 α 的物理含义——**每个 αᵢ 正比于该样本的残差**，残差越大，该样本对最终预测函数的"贡献权重"越大。

---
# 8. Approximation Theory（逼近理论）

**Universal Kernel**：同 Kernel Regression 文档，若核函数（如高斯核）是Universal Kernel，则 RKHS 在紧集上稠密于连续函数空间，KRR 理论上可逼近任意连续函数（当N→∞、λ→0且以适当速率收敛时）。

**Consistency**：在 λ_N→0 且 λ_N·N→∞（正则化衰减速率适中）的条件下，KRR 估计量依概率收敛到贝叶斯最优预测函数。

**Approximation Error / Generalization Error / Bias-Variance**：λ 越大，αᵀKα 惩罚越强，模型趋于平滑，Bias 增大、Variance 减小；λ→0 时 KRR 退化为无正则化的核插值（若K可逆，训练误差趋于0，但方差增大、易过拟合）。最优 λ 通常通过交叉验证选择，等价于在 Bias-Variance 曲线上寻找最小泛化误差点。

---
# 9. Computational Complexity（计算复杂度）

- **Kernel Matrix**：O(N²) 存储
- **训练复杂度**：求解 (K+λI)⁻¹y 需要 O(N³)（矩阵求逆/Cholesky分解）
- **预测复杂度**：单点预测 O(N)（计算k(x)并做内积）
- **内存复杂度**：O(N²)

**为什么 O(N²)**：Gram矩阵大小为N×N。**为什么 GP 是 O(N³)**：见 Gaussian Process 文档第9节，KRR 与 GP 的后验均值公式在数学形式上完全一致（KRR = GP回归的均值预测的频率学派对应版本），因此训练复杂度同样是 O(N³)。

**如何降低复杂度**：
- **Nyström 方法**：用 M≪N 个地标点低秩近似 K≈K_{NM}K_{MM}⁻¹K_{MN}，将求逆复杂度降到 O(NM²)
- **Random Features**：用随机傅里叶特征（Rahimi & Recht, 2007）显式近似核函数，把 KRR 转化为普通（低维）岭回归，训练复杂度降为 O(ND²)
- **Sparse GP / 诱导点方法**：同样适用于 KRR 的大规模近似

---
# 10. RL中的应用

**Kernel TD / Kernel LSPI**：同 Kernel Regression 文档第10节，KRR 的正则化闭式解形式常被直接用于 Kernel LSTD 的实现——在投影贝尔曼方程的最小二乘求解中加入 L2 正则项，得到形如 α=(ΦᵀΦ+λI)⁻¹Φᵀy 的核化解（此处 Φ、y 由贝尔曼残差构造）。

**正则化的必要性（RL特有考量）**：在RL中，价值函数的"标签" y（如TD目标 r+γV̂(s')）本身是自举（bootstrapped）产生的、含噪声且非平稳，纯核插值（无正则化）极易在小样本区域剧烈震荡，KRR 的正则化机制对稳定 RL 中的核方法训练尤为重要。

---
# 11. Convergence Theory（收敛理论）

同 Kernel Regression 文档第11节的一般框架适用；KRR 因为有显式闭式解（不依赖迭代优化），其"优化误差"这一项天然为0，总误差只需考虑逼近误差+估计误差+（RL中的）投影误差，理论分析相对纯核TD更简洁。

---
# 12. Advantages

- 有唯一闭式解（相比SVR等需要求解二次规划的方法计算更直接）
- 正则化机制天然防止过拟合，比纯核回归/核插值更鲁棒
- 继承了核方法的全部理论完备性（RKHS、Representer Theorem）

---
# 13. Disadvantages

- 仍需 O(N³) 训练复杂度，大规模数据集不可行
- 不产生稀疏解（所有训练样本的αᵢ通常都非零，不像SVR那样只有"支持向量"非零）
- 核函数与λ的联合选择仍需交叉验证，调参成本不低

---
# 14. Comparison

| 方法 | 是否有闭式解 | 解是否稀疏 | 是否提供不确定性 | 计算复杂度 |
|---|---|---|---|---|
| KRR | 是 | 否（稠密解） | 否 | O(N³) |
| Kernel Regression（无正则化） | 是（简单形式） | 否 | 否 | O(N²) |
| SVR | 否（需QP求解） | 是（支持向量稀疏） | 否 | O(N²)~O(N³) |
| Gaussian Process | 是（均值同KRR形式） | 否 | 是（天然贝叶斯） | O(N³) |

---
# 15. Applications

- 中小规模回归任务的标准基线方法（statistics/ML教学与工业界均常用）
- RL 中作为 Kernel LSTD/LSPI 的核心求解模块
- 时间序列预测、化学信息学中分子性质预测等中等规模结构化数据任务

---
# 16. Classic Papers（按时间排序）

- Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*.
- Saunders, C., Gammerman, A., & Vovk, V. (1998). *Ridge Regression Learning Algorithm in Dual Variables*.
- Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NeurIPS.

---
# 17. References

- Murphy, K. P. *Machine Learning: A Probabilistic Perspective*，核岭回归章节
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*
