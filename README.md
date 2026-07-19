# Reinforcement Learning —— Function Approximation Theory
> 强化学习中的函数逼近：从数学基础到现代深度RL的完整知识框架

---

## 说明：这份框架相比原版做了什么调整

1. **新增第零章**，把"逼近误差"这个贯穿全书的核心概念提前讲，否则后面每一章都在用却没定义。
2. **数学基础章节**补了统计学习理论（PAC、VC维、Rademacher复杂度），这是"为什么深度网络能泛化"的理论根基，原版缺失。
3. **逼近方法**里补了 NTK（Neural Tangent Kernel）——这是连接"神经网络"和"核方法"两大类的桥梁，现代理论必备。
4. **RL中的函数逼近**补了 GTD 家族、LSTD、Eligibility Trace 与函数逼近的结合，这些是"Deadly Triad"一章能立住的前提。
5. **理论分析**补了 Borkar-Meyn 的 ODE 方法（这是证明随机逼近算法收敛性的标准工具），以及 Approximate Dynamic Programming（Bertsekas 体系）。
6. **深度RL**补了 Double/Dueling DQN、Distributional RL、DDPG/TD3 连续控制这条线，原版从 DQN 直接跳到 PPO，中间断档。
7. 保留你提出的**比较分析**和**发展历史**两章，并入正式结构（第十、十一章），因为这两章确实是串联全书的关键。

---

# 目录（Table of Contents）

- 零、核心概念先行：什么是"逼近误差"
- 一、Function Approximation 基础
- 二、数学理论基础
- 三、函数逼近方法
- 四、优化方法
- 五、强化学习中的函数逼近
- 六、理论分析（Deadly Triad 与收敛性）
- 七、现代深度强化学习
- 八、比较分析
- 九、发展历史
- 十、经典论文
- 十一、推荐教材

---

# 零、核心概念先行：逼近误差的分解

后面所有章节本质上都在回答同一个问题的不同侧面：**用有限参数的 f̂(x;θ) 去逼近真实的 f(x)，误差从哪来？**

总误差可以分解为三部分：

```
总误差 = 逼近误差(Approximation Error) + 估计误差(Estimation Error) + 优化误差(Optimization Error)
```

- **逼近误差**：即使拿到无穷数据、θ 也能完美优化，函数类本身能不能表示 f(x)？（→ 对应第二章"逼近理论"）
- **估计误差**：函数类能表示，但你只有有限样本，学出来的 θ̂ 离最优 θ* 有多远？（→ 对应"统计学习理论"）
- **优化误差**：即使函数类够、数据够，你的优化算法（SGD等）实际有没有收敛到最优点？（→ 对应第四章"优化方法"）

**在RL里还要再加一层**：即使这三项都控制住了，你逼近的目标本身（Bellman 算子的不动点）会不会因为函数逼近而不再存在、不再唯一、甚至不收敛？这就是第六章"Deadly Triad"要讲的东西。

记住这个分解，后面每一章都是在处理其中某一项。

---

# 一、Function Approximation 基础

## 1. 为什么需要函数逼近

- Curse of Dimensionality（状态空间随维度指数增长，表格法不可行）
- Generalization（相似状态应共享经验，而不是逐个记忆）
- Continuous State Space
- Continuous Action Space
- Memory / Sample Efficiency（表格法需要访问每个状态才能学到东西）

## 2. RL中需要逼近什么？

| 对象 | 符号 | 说明 |
|---|---|---|
| State Value Function | V(s) | 逼近一个标量场 |
| Action Value Function | Q(s,a) | 逼近一个定义在 状态×动作 上的标量场 |
| Policy | π(a\|s) | 逼近一个（条件）概率分布 |
| Model | P(s'\|s,a), R(s,a) | 逼近转移分布与奖励函数 |
| Advantage | A(s,a) | Q(s,a) − V(s)，Actor-Critic 中常单独逼近 |

## 3. Function Approximation 统一数学形式

所有方法最终都表示成：

```
f(x)  →  f̂(x; θ)
```

不同方法的唯一区别在于：**如何构造 f̂，以及 θ 是线性还是非线性地进入 f̂**。这个区分（线性 vs 非线性参数化）是整本书最重要的一条分界线——线性方法理论完备、非线性方法（神经网络）理论目前仍不完整，这也是为什么第六章的收敛性结论大多只对线性情形成立。

---

# 二、数学理论基础

这一章回答两个问题：**为什么能逼近任意函数？** 以及 **逼近得"好不好"该怎么衡量？**

## 1. 逼近理论（Approximation Theory）
- Stone-Weierstrass 定理（多项式在连续函数空间中稠密）
- Universal Approximation Theorem（单隐层网络的稠密性，Cybenko/Hornik）
- Depth 与 Width 的表达能力权衡（深度网络的指数级表达优势）
- Spectral Approximation / 误差的频域分析
- Error Analysis（逼近阶数与光滑性的关系，如 Jackson 定理）

## 2. 线性代数
- Vector Space / Basis / Linear Combination
- Orthogonal Basis / Gram-Schmidt
- Projection（最小二乘本质上是到子空间的正交投影，这是 TD 中"投影贝尔曼方程"的代数基础）

## 3. 泛函分析
- Hilbert Space / Banach Space
- Inner Product / Norm
- Complete Space
- **Contraction Mapping & Banach 不动点定理**（贝尔曼算子是压缩映射的证明，是整个RL收敛性理论的地基）

## 4. Fourier 分析
- Fourier Series / Fourier Transform
- Spectral Method
- 用于分析函数的"频率成分"，Fourier Basis 逼近器的理论来源

## 5. 小波分析（Wavelet Analysis）
- Multi-resolution Analysis
- Scaling Function / Wavelet Basis
- 局部 + 多尺度，适合非平稳函数

## 6. 核方法理论（Kernel Theory）
- RKHS（再生核希尔伯特空间）
- Mercer 定理
- Kernel Trick
- **Neural Tangent Kernel (NTK)**：把无穷宽神经网络训练动力学等价为核回归，是理解"神经网络为何能收敛/泛化"的现代理论工具，连接了第三章的"神经逼近"与"核方法"两类

## 7. 概率论
- Bayesian Estimation
- Gaussian Process
- Concentration Inequalities（Hoeffding, Bernstein —— 有限样本误差界的基础工具）

## 8. 统计学习理论（原版缺失，新增）
- Bias-Variance Tradeoff
- PAC Learning（Probably Approximately Correct）
- VC Dimension
- Rademacher Complexity
- Generalization Bound（模型复杂度和样本量如何共同决定估计误差）

## 9. 优化基础（与第四章呼应）
- Gradient Descent / Stochastic Gradient
- Convexity / Newton Method
- 凸优化 vs 非凸优化（线性逼近是凸问题，神经网络逼近是非凸问题——这也是理论难度断层的分界线）

---

# 三、函数逼近方法（Approximation Models）

## 第一类：Linear Basis Expansion

数学形式：`f(x) = Σ wᵢφᵢ(x)`，参数 θ=w 线性进入模型（凸优化、理论完备）。

| 方法 | 数学基础 |
|---|---|
| Polynomial | Taylor 展开 |
| Fourier Basis | 正交三角基 |
| Wavelet | 多尺度正交基 |
| Tile Coding | 分段常数逼近 |
| CMAC | 多重分段叠加 |
| Radial Basis Function (RBF) | 高斯基函数 |
| Spline | 分段多项式 |
| Orthogonal Polynomial | Legendre / Chebyshev / Hermite / Laguerre |

## 第二类：Kernel Methods
数学基础：RKHS。包括 Kernel Regression、Kernel Ridge Regression、Support Vector Regression、Gaussian Process。

## 第三类：Tree-based Approximation
数学基础：递归空间划分（Recursive Space Partition）。包括 Decision Tree、Extra Trees、Random Forest、Gradient Boosting。

## 第四类：Local Approximation
包括 KNN、LOESS、Local Polynomial、Moving Least Squares。

## 第五类：Neural Approximation
数学基础：复合函数（Composition Function），参数**非线性**进入模型。
包括 MLP / CNN / RNN / LSTM / GRU / Transformer / Graph Neural Network。

## 第六类：Bayesian Approximation
包括 Bayesian Linear Regression、Bayesian Neural Network、Gaussian Process（同时给出不确定性估计，RL中用于探索）。

## 第七类：Low-rank Approximation
包括 Matrix Factorization、Tensor Train、Tucker、CP 分解（大规模状态-动作表的压缩表示）。

---

# 四、优化方法

回答：参数怎么学？

- Least Squares（线性情形闭式解）
- Gradient Descent / SGD
- Momentum / RMSProp / Adam
- **Natural Gradient**（在参数空间的黎曼度量下做最速下降，Policy Gradient 理论中的核心工具）
- Trust Region（TRPO/PPO 的理论基础）
- Second-order methods（Newton / Gauss-Newton，收敛更快但代价高）

---

# 五、强化学习中的函数逼近

## Prediction（预测）
逼近 Vπ、Qπ。对应算法：MC、TD、TD(λ)。

## Control（控制）
逼近 Q*。对应算法：SARSA、Q-learning、Expected SARSA。

## Policy Approximation
逼近 π(a|s)，对应 Policy Gradient 家族。

## Actor-Critic
同时逼近 Actor（策略）与 Critic（价值函数），两者的逼近误差会相互耦合影响收敛性。

## Model Learning
逼近 P(s'|s,a)、R(s,a)，用于 Model-based RL。

## 原版缺失、补充的一节：批量与最小二乘方法
- **LSTD (Least-Squares TD)**：直接求解投影贝尔曼方程的闭式解，样本效率高于普通TD
- **LSPI (Least-Squares Policy Iteration)**：LSTD + 策略迭代
- **GTD / GTD2 / TDC (Gradient-TD family)**：解决 off-policy + 线性函数逼近下 TD 不收敛的问题，是应对 Deadly Triad 的关键算法
- **Eligibility Trace 与函数逼近的结合**：λ-return 如何在线性/非线性逼近器下实现

---

# 六、理论分析（这是全书理论核心）

## 基础算子理论
- Bellman Operator（压缩映射，唯一不动点）
- Projected Bellman Equation（函数逼近下，投影到函数子空间后的"近似"不动点）
- Projected Fixed Point
- Semi-gradient（TD更新并非真梯度，是"半梯度"，这也是许多不收敛问题的根源）
- Residual Gradient（真梯度版本，收敛但有偏）

## 误差度量
- MSBE（Mean Squared Bellman Error）
- MSPBE（Mean Squared Projected Bellman Error，GTD家族真正优化的目标）

## Deadly Triad（死亡三角）
三者同时出现时，收敛性不再有保证：
- Bootstrapping（自举）
- Off-policy learning
- Function Approximation

## 收敛性分析
- Linear TD / Linear SARSA / Linear Q-learning 的收敛条件
- Gradient TD、Emphatic TD 如何修复 off-policy 不收敛问题
- Residual Algorithm
- **随机逼近的 ODE 方法（Borkar-Meyn 定理）**：把离散随机迭代与对应的连续时间常微分方程联系起来证明收敛，是这一整套收敛性证明背后的通用工具，原版未提及但非常关键
- **Approximate Dynamic Programming（Bertsekas 体系）**：从最优控制角度看函数逼近下的价值迭代/策略迭代误差传播

## 稳定性技巧（工程手段，但有理论动机）
- Target Network（打破自举中的相关性）
- Experience Replay（打破样本相关性，逼近 i.i.d. 假设）
- Normalization / Gradient Clipping
- Double Q-learning（消除 max 算子的过估计偏差）

---

# 七、现代深度强化学习

## 价值逼近这条线
- **DQN**：为什么用 Target Network + Replay 才能稳定训练？
- **Double DQN**：解决 Q-learning 的过估计
- **Dueling DQN**：V(s) 与 A(s,a) 分离逼近
- **Distributional RL (C51 / QR-DQN)**：不逼近 Q 值的期望，而逼近整个回报分布
- **Rainbow**：以上所有改进的集成，每个模块都对应理论中的某个问题

## 策略逼近这条线
- **PPO**：Trust Region 思想的一阶近似实现
- **DDPG / TD3**：连续动作空间下的确定性策略梯度 + 函数逼近
- **SAC**：最大熵框架下的 Actor-Critic

## Model-based 这条线
- **Dreamer**：在隐空间中学习 World Model
- **MuZero**：不显式学习环境模型，只学习"对规划有用"的隐式模型

## 序列建模视角
- **Decision Transformer**：把 RL 转化为条件序列建模问题，用 Transformer 做函数逼近器

---

# 八、比较分析

| 方法 | 数学原理 | 优点 | 缺点 | 理论收敛性 | 典型RL应用 |
|---|---|---|---|---|---|
| Fourier Basis | 正交展开 | 谱收敛快，理论完备 | 全局基函数，维度灾难 | √ | Mountain Car |
| Wavelet | 多尺度局部基 | 局部性好 | 边界效应复杂 | √ | 机器人控制 |
| Tile Coding | 分段常数 | 计算快，稀疏 | 精度受分辨率限制 | √ | 经典控制任务 |
| RBF | 高斯核 | 光滑、可解释 | 中心/带宽难选 | √ | 连续状态空间 |
| LSTD/LSPI | 闭式最小二乘 | 样本效率高 | 计算复杂度O(d²)~O(d³) | √ | 中小规模线性问题 |
| MLP/CNN | Universal Approximation | 表达能力强 | 理论保证弱，易过拟合 | 部分/依赖NTK等假设 | Deep RL 通用 |
| Gaussian Process | 贝叶斯非参数 | 自带不确定性估计 | O(N³) 计算代价 | √ | 贝叶斯RL、探索 |
| Distributional (C51) | 分布逼近 | 捕捉回报的不确定性 | 需要离散化支撑集 | 实验验证为主 | Atari等高方差环境 |

---

# 九、发展历史

- **1983** CMAC 函数逼近器提出
- **1988** Sutton 提出 TD(λ) + 线性函数逼近
- **1993–1996** RBF Networks 用于RL、LSPI 提出
- **~2000** Kernel-based RL 兴起（Gaussian Process RL 等）
- **2009–2010** Gradient TD 家族（GTD, GTD2, TDC）解决 off-policy 线性收敛问题
- **2013–2015** DQN（首次证明深度网络可稳定用于价值逼近）
- **2015–2016** Double DQN, Dueling DQN, Prioritized Replay
- **2016** A3C（异步并行 Actor-Critic）
- **2017** PPO、Distributional RL (C51)、Rainbow
- **2018** SAC、TD3
- **2019–2020** Dreamer、MuZero（Model-based 复兴）
- **2021** Decision Transformer（序列建模范式）
- **2022–2024** World Models、Foundation RL、大规模离线RL与预训练策略模型的结合

这条时间线的主线是：**表格法 → 线性逼近（理论完备期）→ 核方法/贝叶斯方法（不确定性量化）→ 深度网络（表达力优先，理论滞后）→ 现代方法尝试把理论"补回来"（NTK、Distributional、Offline RL 理论）**。

---

# 十、经典论文（按主题归类，建议按顺序精读）

- Linear Function Approximation：Tsitsiklis & Van Roy (1997) — TD 线性收敛性证明
- Fourier Basis：Konidaris et al. (2011)
- Tile Coding / CMAC：Albus (1975)
- RBF in RL：经典综述类文献
- Gradient TD：Sutton, Maei, Precup et al. (2009)
- DQN：Mnih et al. (2015), Nature
- Double DQN：van Hasselt et al. (2016)
- Distributional RL：Bellemare et al. (2017), C51
- PPO：Schulman et al. (2017)
- SAC：Haarnoja et al. (2018)
- MuZero：Schrittwieser et al. (2020)
- Decision Transformer：Chen et al. (2021)

---

# 十一、推荐教材

- **Sutton & Barto** — *Reinforcement Learning: An Introduction*（入门与主干，第9-13章直接对应函数逼近）
- **Bertsekas** — *Neuro-Dynamic Programming* / *Reinforcement Learning and Optimal Control*（理论最严谨，Approximate DP 视角）
- **Csaba Szepesvári** — *Algorithms for Reinforcement Learning*（简洁的理论小册子，适合过一遍收敛性证明框架）
- **David Silver** — UCL RL 课程讲义（直观、适合搭配教材理解）
- **Hastie, Tibshirani, Friedman** — *The Elements of Statistical Learning*（第二章数学理论的统计学习部分，建议补充阅读）
