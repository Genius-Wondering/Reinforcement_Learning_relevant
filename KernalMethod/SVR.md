# Support Vector Regression（支持向量回归，SVR）
---
# 1. Historical Motivation（历史背景）

## 1.1 为什么提出 Kernel Method

参见《Kernel Regression》文档第1.1节的一般动机。SVR 由 Vapnik 及其合作者在1990年代（从SVM分类扩展而来，Drucker et al. 1997）提出，动机是在核方法的非参数灵活性之上，进一步引入**统计学习理论（VC理论）**的间隔最大化思想，获得比KRR更好的泛化理论保证，同时通过 ε-不敏感损失获得**稀疏解**。

## 1.2 Linear Basis 的局限

同前述文档：人工设计基函数、维数灾难、局部泛化差。SVR 额外要解决 KRR 的一个问题：KRR 的解是稠密的（所有样本都参与预测），预测阶段计算代价高；SVR 希望只用一小部分"关键样本"（支持向量）就能完整刻画决策函数。

## 1.3 Kernel 的核心思想

同前述文档。SVR 的特殊之处：通过 **ε-不敏感损失函数**（对预测误差在 ε 范围内的样本不惩罚），使得大部分训练样本的对偶变量恰好为0，只有落在"间隔边界"外或边界上的样本（支持向量）才有非零系数，从而在保留核方法灵活性的同时获得**稀疏**表示。

---
# 2. Mathematical Motivation（数学动机）

## 2.1 从线性模型开始

线性 SVR 的原始问题：

```
min_{w,b} (1/2)||w||² + C Σᵢ(ξᵢ+ξᵢ*)
s.t.  yᵢ − wᵀxᵢ − b ≤ ε + ξᵢ
      wᵀxᵢ + b − yᵢ ≤ ε + ξᵢ*
      ξᵢ, ξᵢ* ≥ 0
```

**为什么需要 Feature Mapping**：当数据的真实关系非线性时，线性 SVR 表达能力不足，需要先做 φ(x) 映射，把 wᵀxᵢ 替换为 wᵀφ(xᵢ)。

## 2.2 Feature Space
同前述文档。

## 2.3 Infinite-dimensional Feature Space
同前述文档，SVR 常用高斯核对应无穷维特征空间。

## 2.4 Kernel Trick

**完整推导（对偶问题）**：引入 Lagrange 乘子 αᵢ, αᵢ* ≥0（分别对应上下两个不等式约束），构造 Lagrangian 并对 w, b, ξ 求偏导为零，代回原问题，可以证明 w 必然满足：

```
w = Σᵢ(αᵢ−αᵢ*)φ(xᵢ)
```

代入预测函数 f(x)=wᵀφ(x)+b：

```
f(x) = Σᵢ(αᵢ−αᵢ*)φ(xᵢ)ᵀφ(x) + b = Σᵢ(αᵢ−αᵢ*)K(xᵢ,x) + b
```

**为什么可以不用知道φ**：全过程 φ 只以内积 φ(xᵢ)ᵀφ(x) 形式出现，直接替换为核函数 K(xᵢ,x)——这是 Kernel Trick 在 SVR 中的标准应用，与 SVM 分类完全同源。

---
# 3. Mathematical Foundations（数学基础）

## 3.1 Hilbert Space
同前述文档。

## 3.2 Reproducing Kernel Hilbert Space（RKHS）
同前述文档，SVR 的假设空间同样是 RKHS，正则化项 (1/2)||w||² 等价于 (1/2)||f||²_H。

## 3.3 Mercer Theorem
同前述文档。

## 3.4 Representer Theorem

**为什么最优解一定表示成 f(x)=Σᵢ αᵢ K(x,xᵢ)**：SVR 的原始问题可以重写为 min_f (1/2)||f||²_H + C Σᵢ L_ε(yᵢ,f(xᵢ))，其中 L_ε 是 ε-不敏感损失，这完全符合第3.4节 Representer Theorem 的一般形式（损失只依赖训练点取值 + 正则化项是范数的单调函数），因此最优解必然形如 f(x)=Σᵢ(αᵢ−αᵢ*)K(x,xᵢ)+b（额外的偏置项b来自SVR原始问题中显式保留的截距项，不影响 Representer Theorem 对系数部分的结论）。

**SVR 与 KRR 的解的形式差异根源**：两者都满足 Representer Theorem 因而形式相同（都是核函数的线性组合），但 SVR 因为损失函数是分段线性的 ε-不敏感损失（而非KRR的平方损失），对偶问题带有 KKT 互补松弛条件，导致大量 αᵢ 恰好为0——这是稀疏性的数学根源，而非 Representer Theorem 本身带来的。

---
# 4. Mathematical Definition（数学定义）

统一数学形式：

```
f̂(x) = Σᵢ (αᵢ−αᵢ*) K(x,xᵢ) + b
```

**全部符号解释**：
- ε：不敏感带宽，预测误差在 [−ε,ε] 内不计入损失
- C：正则化系数，权衡"间隔最大化（模型简单）"与"训练误差最小化"
- αᵢ, αᵢ*：对偶变量，分别对应"预测值高于/低于真实值超出ε"两种情况，满足 0≤αᵢ,αᵢ*≤C
- **支持向量（Support Vectors）**：使 αᵢ−αᵢ* ≠ 0 的训练样本，即落在 ε-管道边界上或外部的点
- **Kernel Matrix / Gram Matrix**：K_{ij}=K(xᵢ,xⱼ)，与其他核方法定义一致

---
# 5. Kernel Function（核函数）

同前述文档第5节：对称性、半正定性要求及证明通用；常见核函数（Linear/Polynomial/Gaussian/Laplacian/Sigmoid/Matérn）定义、几何意义、适用场景不再重复。SVR 中高斯核和多项式核最常用，Sigmoid核因不严格满足Mercer条件，实践中效果不稳定，使用较少。

---
# 6. Approximation Formula（逼近公式）

统一写成：

```
f(x) = Σᵢ (αᵢ−αᵢ*) K(x,xᵢ) + b
```

**逐步推导**：见第2.4节对偶推导。

**为什么样本就是 Basis**：同前述文档，但 SVR 的特殊之处是——真正参与预测公式求和的只有支持向量（αᵢ−αᵢ*≠0的样本），非支持向量的贡献恒为0，是所有核方法中"有效基函数最精简"的一种。

**为什么属于 Non-parametric**：模型复杂度（支持向量数量）随数据分布和ε、C的设置自适应变化，不是预先固定的。

---
# 7. Optimization（优化）

**Least Squares**：SVR 不使用平方损失，而是 ε-不敏感损失，因此不是标准最小二乘问题。

**Regularization**：目标函数 (1/2)||w||²+C·Σ(ξᵢ+ξᵢ*) 中，(1/2)||w||² 项本身就是一种正则化（间隔最大化等价于L2正则化，这是SVM/SVR理论的经典结论）。

**Dual Problem / Lagrange 完整推导**：

Lagrangian：

```
L = (1/2)||w||² + CΣ(ξᵢ+ξᵢ*) − Σαᵢ(ε+ξᵢ−yᵢ+wᵀφ(xᵢ)+b) − Σαᵢ*(ε+ξᵢ*+yᵢ−wᵀφ(xᵢ)−b) − Σ(μᵢξᵢ+μᵢ*ξᵢ*)
```

对 w, b, ξᵢ, ξᵢ* 分别求偏导并令为0：

```
∂L/∂w = 0  ⟹  w = Σᵢ(αᵢ−αᵢ*)φ(xᵢ)
∂L/∂b = 0  ⟹  Σᵢ(αᵢ−αᵢ*) = 0
∂L/∂ξᵢ = 0  ⟹  C−αᵢ−μᵢ = 0  ⟹  0≤αᵢ≤C
∂L/∂ξᵢ* = 0  ⟹  0≤αᵢ*≤C
```

代回得到只含 α, α* 的对偶问题（二次规划，QP）：

```
max_{α,α*} −(1/2)Σᵢⱼ(αᵢ−αᵢ*)(αⱼ−αⱼ*)K(xᵢ,xⱼ) − εΣᵢ(αᵢ+αᵢ*) + Σᵢyᵢ(αᵢ−αᵢ*)
s.t. Σᵢ(αᵢ−αᵢ*)=0，0≤αᵢ,αᵢ*≤C
```

**KKT互补松弛条件**给出支持向量的判定：αᵢ(ε+ξᵢ−yᵢ+f(xᵢ))=0，只有当样本恰好位于ε-管道边界或之外时，对应αᵢ才可能非零——这正是稀疏性的来源。

---
# 8. Approximation Theory（逼近理论）

**Universal Kernel**：同前述文档，Universal Kernel（如高斯核）保证SVR假设空间在紧集上稠密于连续函数空间。

**Consistency**：随样本量增加、C和ε按适当速率调整，SVR的解依概率收敛到贝叶斯最优回归函数（Steinwart & Christmann, 2008 有系统的一致性理论）。

**Approximation Error / Generalization Error**：SVR 独特之处在于其泛化误差界直接来自**统计学习理论（VC理论/间隔理论）**——间隔越大（||w||越小），VC维越低，泛化误差界越紧，这是SVR相比KRR在小样本下往往泛化更好的理论依据。

**Bias-Variance**：C 越大，对训练误差惩罚越重，模型越贴合数据（Bias减小，Variance增大）；ε 越大，容忍的误差带越宽，有效支持向量越少，模型越简单（Bias增大，Variance减小）。

---
# 9. Computational Complexity（计算复杂度）

- **Kernel Matrix**：O(N²) 存储
- **训练复杂度**：标准QP求解器复杂度约 O(N²)~O(N³)（取决于具体算法，如SMO算法可优化到接近O(N²)）
- **预测复杂度**：O(N_sv)，N_sv 为支持向量数量，通常远小于N——这是SVR相比KRR在预测阶段的核心优势
- **内存复杂度**：训练时O(N²)，训练完成后只需存储支持向量，内存降为O(N_sv)

**为什么 O(N²)**：Gram矩阵计算与存储。**为什么GP是O(N³)**：见Gaussian Process文档；SVR因为是QP问题而非直接矩阵求逆，实践中用SMO等分解算法可以避免显式O(N³)求逆，工程上比GP更容易扩展到较大数据集。

**如何降低复杂度**：
- **Nyström方法**：同样适用于近似SVR训练中的Gram矩阵
- **Random Features**：将SVR转化为线性SVR（如线性SVM求解器可处理百万级样本）
- **Sparse GP方法**：GP领域的稀疏化思想，可类比迁移到大规模SVR近似（如使用工作集选择、分块优化等专用于SVR的稀疏化训练策略）

---
# 10. RL中的应用

**Kernel TD / Kernel SARSA / Kernel LSPI**：文献中偶有用SVR的稀疏解特性替代KRR，以降低核化RL算法在长期在线学习中价值函数表示的内存/计算开销。

**Gaussian Process RL**：SVR与GP在数学形式上相近（都是核方法+间隔/协方差正则化），但GP-RL因提供不确定性估计，在探索导向的RL文献中比SVR更常见；SVR-RL的应用相对更集中在**离线、批量**场景（无需在线更新支持向量集合），如 Fitted Value Iteration 中用SVR拟合价值函数。

**Kernel Bellman Equation 数学推导**：与Kernel Regression文档一致，若用SVR的ε-不敏感损失替代平方损失来求解投影贝尔曼方程，可以得到更鲁棒（对TD目标中的噪声容忍度更高）但求解更复杂（需要QP而非闭式解）的价值函数逼近。

---
# 11. Convergence Theory（收敛理论）

同Kernel Regression文档第11节的一般框架适用于SVR的核化TD/LSPI变体；SVR因优化问题是QP而非闭式解，实践中的"优化误差"需要额外考虑QP求解器的收敛精度，这是SVR相比KRR在RL在线更新场景下计算开销更大的原因之一，也是SVR在RL文献中不如KRR/GP常见的实践原因。

---
# 12. Advantages

- 稀疏解（支持向量），预测阶段计算和内存效率高于KRR
- 理论基础扎实（结合了RKHS理论与VC理论/间隔最大化）
- ε-不敏感损失对小噪声天然鲁棒（在ε管道内的误差完全不惩罚）

---
# 13. Disadvantages

- 训练需要求解QP问题，比KRR的闭式解计算更复杂、调参更多（需要同时选择C、ε和核参数）
- 不提供不确定性估计（相比GP）
- 大规模数据下训练仍然昂贵，需要专用求解器（如SMO）才能实际可扩展

---
# 14. Comparison

| 方法 | 损失函数 | 解的稀疏性 | 是否提供不确定性 | 训练方式 |
|---|---|---|---|---|
| SVR | ε-不敏感损失 | 稀疏（支持向量） | 否 | QP |
| KRR | 平方损失+L2正则 | 稠密 | 否 | 闭式解 |
| Gaussian Process | 概率模型（似然+先验） | 稠密 | 是 | 闭式解（均值同KRR） |
| Kernel Regression | 无正则化局部加权 | 稠密 | 否 | 闭式（简单加权平均） |

---
# 15. Applications

- 中小规模非线性回归任务中稀疏性要求较高的场景（如嵌入式设备上部署的预测模型，需要低预测延迟）
- 金融时间序列预测、生物信息学等经典SVR应用领域
- 离线/批量强化学习中的价值函数拟合（Fitted Value/Q Iteration 的函数逼近器选项之一）

---
# 16. Classic Papers（按时间排序）

- Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*.
- Drucker, H., Burges, C. J., Kaufman, L., Smola, A., & Vapnik, V. (1997). *Support Vector Regression Machines*. NeurIPS.
- Smola, A. J., & Schölkopf, B. (2004). *A Tutorial on Support Vector Regression*. Statistics and Computing.

---
# 17. References

- Vapnik, V. N. *The Nature of Statistical Learning Theory*
- Schölkopf, B., & Smola, A. J. *Learning with Kernels*
