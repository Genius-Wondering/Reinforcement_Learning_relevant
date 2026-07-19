# CMAC（Cerebellar Model Articulation Controller）
---
# 1. Motivation（为什么提出）

**历史背景**：CMAC 由 James Albus 于 1975 年提出，灵感来自对小脑（cerebellum）神经回路结构的建模——小脑中的浦肯野细胞（Purkinje cells）通过大量重叠的感受野对输入做局部响应，Albus 将这一生物学结构抽象为一种可计算的函数逼近器，最初用于机器人手臂的实时控制。

**解决什么问题**：机器人控制需要在毫秒级时间内完成非线性函数逼近与更新，当时的神经网络（感知机）训练慢、难以做到实时；CMAC 希望提供一种**受生物启发、计算极快、支持在线学习**的局部函数逼近方案。

**为什么以前的方法不好**：全局基函数方法（多项式等）无法做到局部快速更新；早期神经网络缺乏高效的在线训练算法（反向传播尚未普及）；CMAC 用查表+局部叠加的方式绕开了这些问题。

---
# 2. Mathematical Idea（数学思想）

**一句话解释**：用多张相互重叠、量化的感受野网格对输入做局部编码，输出为激活的感受野权重之和——这正是 Tile Coding 的原始生物学版本。

**几何解释**：与 Tile Coding 相同，多套偏移的量化网格叠加，产生比单套网格精细得多的等效分辨率，同时保持局部更新特性。

**数学解释**：CMAC 可以看作一种**哈希 + 局部核函数**的组合：输入先被映射到高维稀疏二值向量（感受野激活模式），再通过线性权重求和得到输出。

---
# 3. Mathematical Definition（数学定义）

CMAC 由输入空间到概念存储器的映射 S: X → {0,1}^N（N为感受野/存储单元总数）定义，输出为：

```
f̂(x) = Σᵢ₌₁ᴺ wᵢ Sᵢ(x)，Sᵢ(x) ∈ {0,1}
```

**符号说明**：
- Sᵢ(x)：第 i 个感受野（存储单元）是否被输入 x 激活（1为激活，0为不激活）
- wᵢ：第 i 个存储单元对应的权重
- N：物理存储表大小（常通过哈希压缩，远小于理论上的感受野组合数）

**变量解释**：对任意输入 x，只有固定数量 ρ（泛化参数，对应 Tile Coding 中的 tiling 套数 m）个 Sᵢ(x)=1，其余为0，因此计算和更新都是 O(ρ)。

---
# 4. Basis Function（基函数）

**为什么这样设计**：模拟生物神经元的局部感受野重叠机制，用简单的二值激活代替连续的核函数计算，兼顾生物合理性与计算效率。

**Basis长什么样**：与 Tile Coding 完全一致——每个基函数是输入空间中某个矩形/超立方体区域上的指示函数。

**举例**：一个二维输入通过 ρ=8 套重叠量化层，每层对输入的每一维独立量化并加偏移，8层的激活组合唯一确定输出的插值权重和。

**画图（文字描述）**：与 Tile Coding 完全相同的"多层错位方格纸叠加"图像，CMAC 是这一图像最早的生物学与工程学起源。

---
# 5. Approximation Formula（逼近公式）

统一写成：

```
f(x) = Σᵢ wᵢφᵢ(x)，φᵢ(x) = Sᵢ(x) ∈ {0,1}
```

**如何得到**：通过设计量化分辨率、重叠层数 ρ 以及哈希函数（当理论感受野组合数远大于实际存储表大小 N 时，用哈希做压缩存储，代价是引入哈希碰撞误差）。

**为什么是线性的**：与 Tile Coding 完全一致，wᵢ 线性加权求和，模型关于参数线性。

---
# 6. Mathematical Foundation（数学基础）

- **Approximation Theory**：分段常数逼近，与 Tile Coding 共享同一数学基础
- **神经科学建模**：小脑浦肯野细胞感受野的抽象化，是"生物启发算法"的早期代表
- **哈希理论**：CMAC 实现中大量依赖哈希函数将高维稀疏索引压缩到有限存储表，涉及哈希碰撞概率分析

---
# 7. Approximation Ability（逼近能力）

**为什么能逼近任意函数**：与 Tile Coding 相同，分段常数函数集合在重叠层数、分辨率趋于无穷时可任意精度逼近有界函数。

**误差如何分析**：误差主要由量化分辨率（每层网格宽度）决定，为一阶（线性）收敛；哈希压缩会额外引入碰撞噪声误差。

**是否谱收敛**：否。

**是否局部收敛**：是，局部性是 CMAC 的核心设计目标。

**是否一致收敛**：分辨率趋于无穷时逐点一致收敛，但存在哈希碰撞时会有额外的、不随分辨率消失的噪声下限。

---
# 8. Optimization（参数学习）

- **Least Squares**：理论可行，但 CMAC 设计初衷就是在线实时学习，极少用批量最小二乘
- **Gradient Descent / SGD**：标准做法，Albus 原始论文中用 LMS（最小均方）规则更新权重
- **TD Learning**：CMAC 是最早与 TD 学习结合用于控制任务的函数逼近器之一
- **LSTD**：由于存储表可能极大（依赖哈希压缩），批量最小二乘方法在 CMAC 场景下不常见

---
# 9. RL中的应用

- **表示V**：V̂(s;w) = Σᵢ wᵢSᵢ(s)
- **表示Q**：每个动作独立维护一套 CMAC 权重表
- **表示Policy**：较少直接使用
- **对应算法**：早期机器人控制中的 Sarsa/Q-learning + CMAC，是 Tile Coding 在 RL 社区被广泛采用之前的原始形式

---
# 10. Theoretical Properties（理论性质）

- **是否线性**：是（关于参数）
- **是否凸优化**：是
- **是否保证收敛**：与 Tile Coding 共享同样的线性 TD 收敛理论
- **是否稳定**：数值稳定（0/1特征），但哈希碰撞会引入额外、不可忽略的系统性噪声
- **复杂度**：每次更新/预测 O(ρ)，与 Tile Coding 相同

---
# 11. Advantages（优点）

- 计算极快，适合实时控制（这是 CMAC 最初的设计目标）
- 生物学上有一定合理性，局部更新、局部泛化
- 是 Tile Coding 等后续方法的理论源头，历史地位重要

---
# 12. Disadvantages（缺点）

- 哈希压缩带来的碰撞噪声难以完全消除
- 分辨率设计（量化粒度、重叠层数）需要经验调参
- 相比现代方法（Tile Coding 的规范化实现、RBF等），CMAC 在学术界已较少被直接使用，更多作为历史/教学参照

---
# 13. Comparison（与其他方法比较）

| 方法 | 与CMAC关系 | 计算速度 | 存储方式 | 现代使用率 |
|---|---|---|---|---|
| CMAC | —— | 快 | 哈希压缩表 | 低（历史方法） |
| Tile Coding | CMAC的规范化简化版 | 最快 | 通常不压缩或简单哈希 | 中（教学与基线） |
| RBF | 连续版感受野 | 中 | 显式存储中心 | 中 |
| MLP | 数据驱动的感受野 | 依赖网络规模 | 权重矩阵 | 高（现代主流） |

---
# 14. Applications（实际应用）

- Albus 最初应用于**机械臂实时控制**
- 早期自适应控制、机器人步态学习任务
- 现代教材中通常作为 Tile Coding 的历史铺垫或对比案例出现

---
# 15. Classic Papers（经典论文）

- Albus, J. S. (1975). *A New Approach to Manipulator Control: The Cerebellar Model Articulation Controller (CMAC)*. Journal of Dynamic Systems, Measurement, and Control.
- Albus, J. S. (1975). *Data Storage in the Cerebellar Model Articulation Controller (CMAC)*.

---
# 16. References（教材）

- Sutton & Barto, *Reinforcement Learning: An Introduction*，第9章相关脚注与历史介绍
- Miller, W. T., Glanz, F. H., & Kraft, L. G. (1990). *CMAC: An associative neural network alternative to backpropagation*. Proceedings of the IEEE.
