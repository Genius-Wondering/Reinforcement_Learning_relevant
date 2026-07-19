# Orthogonal Polynomial（正交多项式：Legendre / Chebyshev / Hermite / Laguerre）
---
# 1. Motivation（为什么提出）

**历史背景**：普通多项式基 {1,x,x²,...} 虽然张成同一个函数空间，但作为一组基"不正交"，导致最小二乘拟合时设计矩阵病态（尤其阶数升高时接近奇异）。18-19世纪的数学家（Legendre、Chebyshev、Hermite、Laguerre）针对不同的定义域和权重函数，构造出各自正交的多项式族，从根本上解决了这个数值问题。

**解决什么问题**：需要一组既保持"多项式"的简单解析形式，又像 Fourier 基一样正交、系数可独立求解、数值稳定的基函数。

**为什么以前的方法不好**：普通幂基 xⁱ 在区间内高度线性相关（尤其高阶项彼此"看起来很像"），导致最小二乘的 Gram 矩阵 ΦᵀΦ 条件数随阶数指数恶化；正交多项式通过在特定内积下互相正交，从根本上消除了这个问题。

---
# 2. Mathematical Idea（数学思想）

**一句话解释**：针对不同的定义域和"权重函数"（对函数不同区域的重视程度），构造一组彼此正交的多项式基，兼具多项式的简单性和正交基的数值稳定性。

**几何解释**：与 Fourier Basis 相同的几何图景——逼近即为函数在正交基下的投影；区别在于 Fourier 用三角函数做正交基，这里用多项式做正交基，且允许非均匀的权重（对应不同物理/概率背景）。

**数学解释**：正交多项式族由特定的**权重函数 ω(x)** 和内积 ⟨f,g⟩=∫f(x)g(x)ω(x)dx 决定；不同的权重函数和定义域对应不同名字的正交多项式族。

---
# 3. Mathematical Definition（数学定义）

一般正交多项式满足：

```
∫ Pᵢ(x)Pⱼ(x)ω(x)dx = 0，当 i≠j
```

四类经典正交多项式及其定义域/权重：

| 名称 | 定义域 | 权重函数 ω(x) | 典型应用背景 |
|---|---|---|---|
| Legendre | [−1,1] | ω(x)=1（均匀权重） | 无先验偏好的一般区间逼近 |
| Chebyshev | [−1,1] | ω(x)=1/√(1−x²) | 极小化最大误差（minimax逼近） |
| Hermite | (−∞,∞) | ω(x)=e^(−x²) | 高斯分布相关问题、量子谐振子 |
| Laguerre | [0,∞) | ω(x)=e^(−x) | 指数分布相关、半无限区间问题 |

逼近表示为：

```
f̂(x) = Σᵢ wᵢPᵢ(x)
```

**符号说明**：
- Pᵢ(x)：第 i 阶正交多项式（依具体族有不同递归定义）
- ω(x)：权重函数，反映内积对定义域不同位置的"重视程度"
- wᵢ：投影系数，wᵢ = ⟨f,Pᵢ⟩/⟨Pᵢ,Pᵢ⟩

**变量解释**：选择哪一族正交多项式，本质上是在为你的问题选择一个"合适的定义域 + 合适的误差度量方式（权重）"。

---
# 4. Basis Function（基函数）

**为什么这样设计**：每族多项式的权重函数都对应特定的应用场景——Chebyshev 权重在区间两端加大权重，目的是压制 Runge 现象（均匀化最大误差）；Hermite 权重对应高斯分布，天然适合处理正态噪声主导的问题；Laguerre 权重对应指数衰减过程。

**Basis长什么样**：
- Legendre：P₀=1, P₁=x, P₂=(3x²−1)/2，形状与普通多项式接近但在[-1,1]内振荡幅度均匀
- Chebyshev：T₀=1, T₁=x, T₂=2x²−1，等价于 Tₙ(cosθ)=cos(nθ)，振荡幅度在区间内完全均匀（这是其minimax性质的来源）
- Hermite：H₀=1, H₁=2x, H₂=4x²−2，随|x|增大快速增长但被权重e^(−x²)压制
- Laguerre：L₀=1, L₁=1−x, L₂=(x²−4x+2)/2，用于半无限区间

**举例**：在[−1,1]上用3阶 Chebyshev 展开逼近一个光滑函数，比用同阶数普通多项式在区间端点处的最大误差小得多。

**画图（文字描述）**：Legendre 多项式看起来像"温和摆动"的曲线，振幅在区间中部略大；Chebyshev 多项式的振幅在整个区间内完全一致（像正弦波一样均匀波动）；Hermite/Laguerre 多项式随自变量增大会剧烈增长，但因为定义中隐含权重函数的压制，实际"有效贡献"仍然是局部化的。

---
# 5. Approximation Formula（逼近公式）

统一写成：

```
f(x) = Σᵢ wᵢφᵢ(x)，φᵢ(x) = Pᵢ(x)（某一族正交多项式）
```

**如何得到**：由该族多项式相对于其权重函数 ω(x) 的正交性，系数可直接通过内积积分计算得到（Galerkin投影），无需求解线性方程组。

**为什么是线性的**：与所有线性基展开方法相同，wᵢ 线性加权；额外优势与 Fourier Basis 相同——正交性使 Gram 矩阵接近对角，数值求解远比普通多项式基稳定。

---
# 6. Mathematical Foundation（数学基础）

- **Orthogonal Polynomial 理论**：Sturm-Liouville 理论统一给出了这四类正交多项式的生成方式（都是某类二阶微分方程的特征函数）
- **Approximation Theory**：Chebyshev 多项式在 minimax（切比雪夫）逼近意义下是最优的，即在给定阶数下最小化最大误差
- **概率论**：Hermite 多项式与高斯分布的矩、Wiener 混沌展开（用于随机过程分析）紧密相关；Laguerre 多项式与指数/Gamma分布相关
- **数值分析**：Chebyshev 节点是多项式插值中避免 Runge 现象的标准选择

---
# 7. Approximation Ability（逼近能力）

**为什么能逼近任意函数**：本质仍是 Stone-Weierstrass 定理的推论——正交多项式族张成与普通多项式相同的函数空间，只是基的选取方式不同，因此同样在各自定义域上稠密。

**误差如何分析**：
- Chebyshev 逼近对解析函数有指数（谱）收敛，且是所有同阶多项式逼近中**最大误差最小**的（minimax最优）
- Legendre 逼近在 L² 意义下最优（最小化均方误差而非最大误差）
- Hermite/Laguerre 的收敛速度依赖函数在对应权重下的可积性和光滑性

**是否谱收敛**：Chebyshev 和 Legendre 对解析函数均可达到谱收敛（指数收敛），是正交多项式方法相比普通 Polynomial Basis 的核心优势。

**是否局部收敛**：否，仍是全局基，不具备局部性。

**是否一致收敛**：Chebyshev 展开对连续函数在整个区间上一致收敛性质最好（因为其最大误差最小化的性质直接对应一致范数）。

---
# 8. Optimization（参数学习）

- **Least Squares**：由正交性，系数可直接通过数值积分（如 Gauss-Chebyshev 求积、Gauss-Hermite 求积）高效计算
- **Gradient Descent**：在线学习/RL场景下的标准做法
- **TD Learning**：作为线性RL特征，半梯度TD更新，收敛性分析与其他线性基展开方法一致
- **LSTD**：适用，且由于正交性，设计矩阵条件数优于普通多项式基

---
# 9. RL中的应用

- **表示V**：V̂(s;w) = Σᵢ wᵢPᵢ(s)，需先将状态归一化到对应族的标准定义域（如Legendre/Chebyshev归一化到[−1,1]）
- **表示Q**：类似地构造状态-动作联合空间上的多元正交多项式（张量积形式）
- **表示Policy**：较少直接使用，多用于价值函数一侧
- **对应算法**：Linear SARSA/Q-learning，在需要比 Fourier Basis 更契合特定分布假设（如状态噪声近似高斯）的场景中，Hermite 多项式偶有使用

---
# 10. Theoretical Properties（理论性质）

- **是否线性**：是（关于参数）
- **是否凸优化**：是
- **是否保证收敛**：线性 TD 收敛理论适用
- **是否稳定**：数值稳定性显著优于普通 Polynomial Basis（这是引入正交多项式的根本目的）
- **复杂度**：与 Polynomial Basis 相同，多维时基函数数量随维度增长（O(nᵈ)），维度灾难依然存在

---
# 11. Advantages（优点）

- 数值稳定性远优于普通多项式基，高阶展开依然可靠
- Chebyshev 族具有 minimax 最优性，理论保证最强
- 与经典数值分析、概率论（Hermite-高斯、Laguerre-指数）有深厚的理论联系，便于借用成熟工具

---
# 12. Disadvantages（缺点）

- 仍是全局基，局部适应性差
- 需要根据问题选择合适的族（定义域、权重），选错会导致收敛变差
- 高维推广同样面临维度灾难

---
# 13. Comparison（与其他方法比较）

| 方法 | 数值稳定性 | 收敛性质 | 与其他理论的联系 | 局部性 |
|---|---|---|---|---|
| Orthogonal Polynomial | 高 | 谱收敛（Chebyshev/Legendre对解析函数） | 数值分析、概率论 | 差 |
| 普通 Polynomial | 低（高阶病态） | 代数阶收敛 | 无 | 差 |
| Fourier | 高 | 谱收敛（解析函数） | 信号处理 | 差 |
| RBF | 中 | 依带宽 | 核方法 | 好 |

---
# 14. Applications（实际应用）

- 数值求解偏微分方程中的谱方法（Chebyshev/Legendre 谱方法），间接影响基于模型的RL中环境动力学的数值求解
- 不确定性量化中的多项式混沌展开（Polynomial Chaos Expansion，基于 Hermite/Legendre 多项式），用于随机动力系统建模
- 教学与理论分析中作为"改进版"的 Polynomial Basis 基线

---
# 15. Classic Papers（经典论文）

- Chebyshev, P. L. (1854). *Théorie des mécanismes connus sous le nom de parallélogrammes*（切比雪夫逼近思想的早期来源）
- Wiener, N. (1938). *The Homogeneous Chaos*（Hermite 多项式在随机过程展开中的奠基性工作）

---
# 16. References（教材）

- Trefethen, L. N. *Approximation Theory and Approximation Practice*（Chebyshev 方法的现代权威教材）
- Abramowitz, M., & Stegun, I. A. *Handbook of Mathematical Functions*，正交多项式性质速查
- Xiu, D. *Numerical Methods for Stochastic Computations*（Hermite/Legendre 多项式混沌展开）
