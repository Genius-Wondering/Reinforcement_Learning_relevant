# Fourier Basis（傅里叶基函数逼近）
---
# 1. Motivation（为什么提出）

**历史背景**：Fourier 在研究热传导方程时（1807年）提出任意周期函数可以分解为正弦余弦波的叠加，这是频域分析的起点。RL 中的 Fourier Basis 由 Konidaris et al. (2011) 系统引入，作为比多项式更稳定、比 RBF 更少超参数的线性逼近器。

**解决什么问题**：多项式基在高阶时数值不稳定（Vandermonde 矩阵病态），RBF 需要手动选择中心和带宽；Fourier Basis 希望提供一组**正交、参数少、数值稳定**的基函数。

**为什么以前的方法不好**：Polynomial 全局基振荡严重；RBF 中心数量随维度指数增长且需要调参；Fourier Basis 用固定频率组合、无需调参中心位置，构造更系统化。

---
# 2. Mathematical Idea（数学思想）

**一句话解释**：把函数看成不同频率正弦/余弦波的叠加，用有限个频率分量去逼近。

**几何解释**：{cos(kπx), sin(kπx)} 在 L²([0,1]) 空间中构成一组**正交基**，逼近即为函数在这组正交基下的投影（截断傅里叶级数）。

**数学解释**：正交性意味着系数可以独立计算（不需要联立求解方程组），wᵢ = ⟨f, φᵢ⟩/⟨φᵢ, φᵢ⟩，这是 Fourier Basis 相比一般线性基展开的独特优势。

---
# 3. Mathematical Definition（数学定义）

一维情形（x ∈ [0,1]）：

```
f̂(x) = Σₖ₌₀ⁿ wₖ cos(kπx)
```

多维情形（x ∈ [0,1]ᵈ，Konidaris 版本）：

```
φᵢ(x) = cos(π cᵢ · x)，cᵢ ∈ {0,1,...,n}ᵈ
```

**符号说明**：
- k / cᵢ：频率（阶数）向量，控制该基函数振荡快慢
- n：最高阶数，决定基函数总数（多维时为 (n+1)ᵈ）
- x：归一化到 [0,1]ᵈ 的状态特征

**变量解释**：cᵢ 的每个分量控制对应输入维度上的振荡频率；cᵢ全零对应常数基（直流分量）。

---
# 4. Basis Function（基函数）

**为什么这样设计**：余弦基满足 Neumann 边界条件（导数在边界为零），比正弦更适合近似有边界的价值函数（避免边界处的强制过零）。

**Basis长什么样**：φ₀(x)=1（直流分量），φ₁(x)=cos(πx)（半个周期），φ₂(x)=cos(2πx)（一个完整周期），阶数越高振荡越快。

**举例**：n=3 时，一维基为 {1, cos(πx), cos(2πx), cos(3πx)}

**画图（文字描述）**：φ₀ 是水平直线；φ₁ 从1单调降到−1；φ₂ 是一个完整余弦波动；阶数越高波峰波谷越多，类似"频率越来越高的振荡曲线"。

---
# 5. Approximation Formula（逼近公式）

统一写成：

```
f(x) = Σᵢ wᵢφᵢ(x)，φᵢ(x) = cos(π cᵢ·x)
```

**如何得到**：由 Fourier 级数理论，任意 L² 函数可以展开为该正交基的无穷级数，截断到有限阶即得逼近。

**为什么是线性的**：与 Polynomial Basis 相同，wᵢ 线性加权，模型关于参数是线性的；额外优势是基函数正交，使得最小二乘问题的设计矩阵 ΦᵀΦ 接近对角阵，数值条件数远优于 Polynomial。

---
# 6. Mathematical Foundation（数学基础）

- **Fourier Analysis**：Fourier 级数、Fourier 变换、Parseval 定理（能量在时域/频域守恒）
- **正交多项式/正交函数理论**：{cos(kπx)} 关于内积 ⟨f,g⟩=∫f(x)g(x)dx 正交
- **Approximation Theory**：解析函数的 Fourier 系数按指数衰减（谱收敛的来源）

---
# 7. Approximation Ability（逼近能力）

**为什么能逼近任意函数**：L² 空间中正交基张成稠密子空间，任意平方可积函数的 Fourier 级数在 L² 范数下收敛到原函数。

**误差如何分析**：若 f 是解析函数（无穷次可微且满足特定增长条件），Fourier 系数呈指数衰减，截断误差为 O(e⁻ᶜⁿ)；若 f 只是有限次可微，误差为多项式阶衰减。

**是否谱收敛**：是，对解析函数成立——这是 Fourier Basis 相比 Polynomial Basis 最大的理论优势。

**是否局部收敛**：否，同样是全局基，Gibbs 现象会在函数不连续点附近产生振荡伪影。

**是否一致收敛**：连续可微函数逐点一致收敛；若函数本身不连续（如价值函数在障碍处有跳变），会出现 Gibbs 振荡，不一致收敛。

---
# 8. Optimization（参数学习）

- **Least Squares**：由正交性，wᵢ = ⟨f,φᵢ⟩/||φᵢ||² 可独立计算，无需矩阵求逆（对比 Polynomial 需要求逆病态矩阵）
- **Gradient Descent / SGD**：在线场景下常用，逐样本更新
- **TD Learning**：Konidaris 论文中给出针对不同频率分量的**自适应学习率**方案（高频分量用更小学习率，因为其对应更精细/更易过拟合的成分）
- **LSTD**：同样适用，且由于基正交，设计矩阵条件数更好，数值稳定性优于多项式基

---
# 9. RL中的应用

- **表示V**：V̂(s;w) = Σᵢ wᵢcos(πcᵢ·s)，s 需要先归一化到 [0,1]ᵈ
- **表示Q**：对每个离散动作维护独立的一组权重 wₐ，Q̂(s,a) = Σᵢ wₐ,ᵢφᵢ(s)
- **表示Policy**：通常不直接用，因为策略需要非负/归一化，Fourier Basis 更常用于价值逼近侧
- **对应算法**：Linear SARSA(λ)、Q-learning，Konidaris et al. 在 Mountain Car、Pinball、Acrobot 上做了系统对比实验

---
# 10. Theoretical Properties（理论性质）

- **是否线性**：是（关于参数）
- **是否凸优化**：是
- **是否保证收敛**：on-policy 线性 TD 收敛性同样适用（属于线性函数逼近的一般理论）
- **是否稳定**：数值上比 Polynomial 更稳定（正交基，条件数低）
- **复杂度**：多维情形基函数数量为 (n+1)ᵈ，同样存在维度灾难，但常数因子和数值条件优于其他方法

---
# 11. Advantages（优点）

- 正交基，最小二乘问题数值稳定，甚至可解耦独立求解
- 超参数少（只需指定最高阶 n），无需像 RBF 一样调中心和带宽
- 对光滑函数有谱收敛的理论保证

---
# 12. Disadvantages（缺点）

- 高维时基函数数量指数增长（维度灾难依然存在）
- 全局基，函数若有局部尖锐特征（不连续点），逼近效果差，出现 Gibbs 振荡
- 需要将输入归一化到有界区间 [0,1]ᵈ，对无界状态空间不友好

---
# 13. Comparison（与其他方法比较）

| 方法 | 参数调节难度 | 数值稳定性 | 光滑函数逼近速度 | 局部特征捕捉 |
|---|---|---|---|---|
| Fourier | 低（仅调阶数） | 高（正交） | 快（谱收敛） | 差 |
| Polynomial | 低 | 低（高阶病态） | 中 | 差 |
| Wavelet | 中 | 高 | 中-快 | 好 |
| RBF | 高（需调中心/带宽） | 中 | 依赖带宽 | 好 |
| Tile Coding | 中（需调分辨率） | 高 | 慢 | 好 |

---
# 14. Applications（实际应用）

- **Mountain Car**、**Pinball**、**Acrobot**：Konidaris 原论文的标准测试环境，Fourier Basis 显著优于同参数量的 RBF/Polynomial
- 低维连续控制任务中作为线性 SARSA/Q-learning 的标准特征表示
- 因维度灾难限制，较少直接用于高维图像输入任务（此类任务转向 CNN）

---
# 15. Classic Papers（经典论文）

- Konidaris, G., Osentoski, S., & Thomas, P. (2011). *Value Function Approximation in Reinforcement Learning using the Fourier Basis*. AAAI.

---
# 16. References（教材）

- Sutton & Barto, *Reinforcement Learning: An Introduction*，第9.5节 专门介绍 Fourier Basis
- Stein, E. M., & Shakarchi, R. *Fourier Analysis: An Introduction*
