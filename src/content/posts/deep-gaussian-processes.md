---
title: 深度高斯过程（Deep Gaussian Processes）——从浅层到深层的概率建模
published: 2026-05-08
description: 深度高斯过程（DGP）将高斯过程的非参数灵活性与深度学习的层级抽象能力结合，形成一种兼具不确定性与表达力的概率模型。本文从 GP 基础出发，逐步拆解 DGP 的动机、数学模型、训练挑战与应用场景。
tags: [高斯过程, 深度高斯过程, DGP, 机器学习, 概率建模, 贝叶斯方法]
category: 机器学习
draft: false
image: ''
lang: ''
---

## 1. 从高斯过程说起

高斯过程（Gaussian Process, GP）是机器学习中最优雅的概率模型之一。一个高斯过程完全由均值函数 $m(\mathbf{x})$ 和协方差函数（核函数）$k(\mathbf{x}, \mathbf{x}')$ 定义：

$$
f(\mathbf{x}) \sim \mathcal{GP}\big(m(\mathbf{x}),\ k(\mathbf{x}, \mathbf{x}')\big)
$$

给定训练数据 $(\mathbf{X}, \mathbf{y})$，GP 对任意测试点 $\mathbf{X}_*$ 的后验预测分布为：

$$
\begin{aligned}
\mu_* &= \mathbf{K}_{*f} (\mathbf{K}_{ff} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y} \\[4pt]
\Sigma_* &= \mathbf{K}_{**} - \mathbf{K}_{*f} (\mathbf{K}_{ff} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{K}_{f*}
\end{aligned}
$$

GP 的核心优势在于：**自带不确定性估计**（每个预测都是一个分布）、**非参数**（模型容量随数据增长）、**通过核函数编码先验知识**。

## 2. GP 的局限性

尽管 GP 在贝叶斯优化、地质统计等领域表现优异，标准 GP 有两个根本性限制：

### 2.1 表达能力受限于单一核函数

GP 的行为完全由核函数决定。比如 RBF 核：

$$
k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)
$$

这隐含假设了函数的**平稳性**——任意两点之间的协方差只依赖于它们的距离，而与具体位置无关。当数据生成过程是**非平稳**的（比如高频区域与低频区域混合），单一核函数就难以描述。

### 2.2 计算复杂度

精确 GP 回归的计算复杂度是 $\mathcal{O}(N^3)$，存储复杂度是 $\mathcal{O}(N^2)$。虽然有稀疏 GP、诱导点方法等近似手段，但这些方法同样限制了模型的表达能力。

## 3. 深度高斯过程的核心思想

深度高斯过程（Deep Gaussian Process, DGP）由 **Damianou 和 Lawrence (2013)** 提出，核心思想十分直观：**把多个 GP 层堆叠起来**。

在标准 GP 中，我们直接对输出 $\mathbf{y}$ 建模：

$$
\mathbf{y} = f(\mathbf{X}) + \varepsilon, \quad f \sim \mathcal{GP}(0, k)
$$

而在 $L$ 层 DGP 中，每一层都是一个 GP，前一层的输出成为后一层的输入：

$$
\begin{aligned}
\mathbf{F}_1 &\sim \mathcal{GP}(0, k_1(\mathbf{X}, \mathbf{X})) \\
\mathbf{F}_2 &\sim \mathcal{GP}(0, k_2(\mathbf{F}_1, \mathbf{F}_1)) \\
&\ \vdots \\
\mathbf{F}_L &\sim \mathcal{GP}(0, k_L(\mathbf{F}_{L-1}, \mathbf{F}_{L-1})) \\
\mathbf{y} &= \mathbf{F}_L + \varepsilon
\end{aligned}
$$

这可以看作对输入空间进行一系列"扭曲"（warping），每一层 GP 对输入做一次非线性变换，逐步将输入映射到更适合建模的空间。

```mermaid
graph LR
    X["输入 X"] --> F1["GP 层 1<br/>F₁ ~ GP(0, k₁)"]
    F1 --> F2["GP 层 2<br/>F₂ ~ GP(0, k₂)"]
    F2 --> F3["..."]
    F3 --> FL["GP 层 L<br/>F_L ~ GP(0, k_L)"]
    FL --> Y["输出 Y"]
```

### 3.1 与深度神经网络的关系

可以把 DGP 看作深度神经网络（DNN）的**概率对应物**：

| | 深度神经网络 | 深度高斯过程 |
|---|---|---|
| **基本单元** | 神经元（仿射变换 + 逐点激活函数） | 高斯过程（核函数定义的非线性映射） |
| **层级组合** | $\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)})$ | $\mathbf{F}_l \sim \mathcal{GP}(0, k_l(\mathbf{F}_{l-1}))$ |
| **输出类型** | 点估计 | 概率分布 |
| **训练方式** | 反向传播 + 梯度下降 | 变分推断 / 期望传播 |
| **不确定性** | 需要额外机制（Dropout, Ensemble） | 内置贝叶斯不确定性 |

事实上，无限宽的深层神经网络会收敛到一个具有特定核函数的 GP ——这被称为神经正切核（NTK）。而 DGP 从这个对偶方向出发，直接用 GP 构建层级模型。

## 4. 数学模型与关键挑战

### 4.1 联合分布与带隐变量的贝叶斯推断

DGP 的联合分布为：

$$
p(\mathbf{y}, \{\mathbf{F}_l\}_{l=1}^{L}) = p(\mathbf{y} \mid \mathbf{F}_L) \prod_{l=1}^{L} p(\mathbf{F}_l \mid \mathbf{F}_{l-1})
$$

其中 $\mathbf{F}_0 = \mathbf{X}$，每一层的条件分布 $p(\mathbf{F}_l \mid \mathbf{F}_{l-1})$ 是一个高斯分布。

这里的问题在于：**隐变量 $\mathbf{F}_l$ 和观测变量之间存在非线性的 GP 映射**，使得后验分布 $p(\{\mathbf{F}_l\} \mid \mathbf{X}, \mathbf{y})$ 不再是高斯分布，无法解析求解。

### 4.2 变分推断方法

Damianou 和 Lawrence 提出使用**变分推断**来近似后验。他们引入诱导点（inducing points）——在每个 GP 层中选取 $M$ 个伪输入 $\mathbf{Z}_l$ 及其对应的函数值 $\mathbf{U}_l$，将推断问题转换为：

$$
q(\{\mathbf{F}_l, \mathbf{U}_l\}) = \prod_{l=1}^{L} p(\mathbf{F}_l \mid \mathbf{U}_l, \mathbf{F}_{l-1}) \ q(\mathbf{U}_l)
$$

变分下界（ELBO）为：

$$
\mathcal{L} = \mathbb{E}_q\left[\log p(\mathbf{y} \mid \mathbf{F}_L)\right] - \sum_{l=1}^{L} \text{KL}\big(q(\mathbf{U}_l) \ \|\ p(\mathbf{U}_l)\big)
$$

### 4.3 双重随机变分推断

**Salimbeni 和 Deisenroth (2017)** 进一步提出**双重随机变分推断**（Doubly Stochastic Variational Inference, DSVI），使 DGP 能够扩展到大规模数据。关键技巧：

- 对层间传递进行**随机采样**
- 使用**小批量**（mini-batch）训练
- 每一层的输出对下一层输入来说是随机的，于是 ELBO 的期望可以通过蒙特卡洛采样估计

这使得 DGP 的计算复杂度从 $\mathcal{O}(NM^2)$ 降低到可处理的 $\mathcal{O}(M^3)$。

## 5. 为什么 DGP 有效

### 5.1 非平稳建模能力

标准 GP 的核函数是平稳的：$k(\mathbf{x}, \mathbf{x}') = k(\|\mathbf{x} - \mathbf{x}'\|)$。DGP 的层级结构使得有效核函数变为：

$$
k_{\text{eff}}(\mathbf{x}, \mathbf{x}') = \mathbb{E}_{p(\mathbf{F}_1 \mid \mathbf{X})}\big[k_2(f_1(\mathbf{x}), f_1(\mathbf{x}'))\big]
$$

这不再是位移不变的——在输入空间的不同区域，有效核函数的行为不同。Duvenaud 等人（2014）证明，DGP 可以建模"核函数随输入变化而变化"的非平稳过程。

### 5.2 优美地退化

当数据确实是平稳的时，DGP 的后验会迫使中间层的 GP 接近恒等映射（或线性映射），此时 DGP 退化回标准 GP。这提供了一种优雅的自适应行为。

### 5.3 更好的边际似然建模

在图像、空间统计等复杂数据上，DGP 的边际似然往往优于浅层 GP，表明它捕捉到了数据中更多的结构。

## 6. 代码实践

下面使用 GPyTorch 构建一个简单的两层 DGP：

```python
import torch
import gpytorch
from gpytorch.models import DeepGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import DeepApproximateMLL


class DGPHiddenLayer(gpytorch.models.ApproximateGP):
    """DGP 中的单个隐层"""
    def __init__(self, input_dim, output_dim, num_inducing=128):
        inducing_points = torch.randn(num_inducing, input_dim)
        variational_distribution = CholeskyVariationalDistribution(num_inducing)

        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TwoLayerDGP(DeepGP):
    """两层深度高斯过程"""
    def __init__(self, train_x_shape):
        super().__init__()

        # 第一层: 输入维度 -> 隐维度
        self.hidden_layer = DGPHiddenLayer(
            input_dim=train_x_shape[-1],
            output_dim=2,       # 隐层输出维度
            num_inducing=128
        )
        # 第二层: 隐维度 -> 输出维度
        self.output_layer = DGPHiddenLayer(
            input_dim=2,        # 接收第一层的输出
            output_dim=1,       # 最终输出维度
            num_inducing=128
        )

    def forward(self, inputs):
        hidden_outputs = self.hidden_layer(inputs)
        output = self.output_layer(hidden_outputs)
        return output


def train_dgp(model, train_x, train_y, num_epochs=1000):
    """训练 DGP 模型"""
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)

    mll = DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(
            model.likelihood, model, num_data=train_y.size(0)
        )
    )

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.3f}')

    return model


# ---- 使用示例 ----
# 生成带非平稳性的合成数据
def generate_nonstationary_data(n=500):
    torch.manual_seed(42)
    x = torch.linspace(-3, 3, n).unsqueeze(-1)
    # 左侧高频，右侧低频
    y = torch.where(
        x < 0,
        torch.sin(5 * x.squeeze()) * 0.5,            # 高频正弦
        torch.sin(x.squeeze()) * 1.5                  # 低频正弦
    ) + 0.1 * torch.randn(n)
    return x, y.unsqueeze(-1)


train_x, train_y = generate_nonstationary_data(500)

# 构建并训练 DGP
model = TwoLayerDGP(train_x.shape)
model = train_dgp(model, train_x, train_y, num_epochs=500)

# 预测
model.eval()
test_x = torch.linspace(-4, 4, 200).unsqueeze(-1)
with torch.no_grad():
    pred = model(test_x)
    mean = pred.mean
    lower, upper = pred.confidence_region()

print(f"预测完成: 均值范围 [{mean.min():.2f}, {mean.max():.2f}]")
```

这段代码演示了一个关键设计：**每一层 GP 的低维输出**（`output_dim=2`）。这实际上做了降维（bottleneck），迫使中间层提取有意义的特征表达，避免层间传递的退化。

## 7. Vecchia 近似与大规模 DGP

传统 DGP 的推断复杂度仍然较高。**Sauer 等人（2022）** 将 **Vecchia 近似**引入 DGP，这是一个重要的进展。

Vecchia 近似的核心思想是利用条件独立性：

$$
p(\mathbf{f}) \approx \prod_{i=1}^{N} p(f_i \mid \mathbf{f}_{\text{ne}(i)})
$$

其中 $\text{ne}(i)$ 是第 $i$ 个数据点的最近邻集合。通过仅对最近邻条件化，联合分布被分解为多个小型条件分布的乘积，计算复杂度降至 $\mathcal{O}(N m^3)$，其中 $m \ll N$ 是邻居数量。

Vecchia-DGP 的优势：
- 可以处理超过 **10 万** 级别的数据点
- 保留了 DGP 的非平稳建模能力
- 推断过程不需要诱导点

## 8. 应用场景

| 领域 | 应用 | 为什么 DGP 适用 |
|---|---|---|
| **贝叶斯优化** | 超参数调优, 实验设计 | 需要带不确定性的代理模型，目标函数常非平稳 |
| **空间统计** | 环境监测, 地质建模 | 空间过程的非平稳性（城市 vs 乡村） |
| **机器人** | 动力学建模, 状态估计 | 需要置信区间用于安全控制 |
| **计算机实验** | 仿真代理建模 | 仿真输出可能有非平稳区域 |
| **医学** | 疾病进展建模 | 需要不确定性量化 + 复杂非线性 |
| **气候科学** | 气象场插值 | 非平稳的空间相关性 |

## 9. 局限性与未来方向

当前 DGP 方法的局限：

1. **推断质量依赖变分近似的选择**：变分近似可能低估不确定性
2. **层数通常较浅**：实践中 2-4 层的 DGP 最常见，更深层的 DGP 面临梯度消失/爆炸和推断不稳定
3. **隐层维度的选择**：降维步骤需要人为设定，缺乏理论指导
4. **与 DNN 的竞争**：在图像、语言等大规模结构化数据上，DNN 的工程成熟度远超 DGP

活跃的研究方向包括：
- **卷积 DGP**：引入平移等变结构建模图像
- **对抗训练 DGP**：提升生成模型质量
- **与神经网络的混合模型**：用 DNN 做特征提取 + DGP 做概率输出

## 10. 总结

深度高斯过程是 GP 到深度学习的一次自然延伸。它的核心贡献在于：

- **用 GP 替代神经网络的确定性层**，保留完整的概率语义
- 通过层级结构实现**非平稳建模**
- 当数据平稳时能**优雅退化**为普通 GP

DGP 不是要替代深度神经网络，而是在**需要不确定性量化且数据量适中的场景下**提供一个更具原则性的选项。如果你想在贝叶斯框架下享受深度学习的表达能力，DGP 值得一试。

---

## 参考文献

1. Damianou, A., & Lawrence, N. D. (2013). *Deep Gaussian processes*. AISTATS 2013.
2. Salimbeni, H., & Deisenroth, M. (2017). *Doubly stochastic variational inference for deep Gaussian processes*. NeurIPS 2017.
3. Duvenaud, D., et al. (2014). *Avoiding pathologies in very deep networks*. AISTATS 2014.
4. Sauer, A., Cooper, A., & Gramacy, R. B. (2022). *Vecchia-approximated deep Gaussian processes for computer experiments*. Journal of Computational and Graphical Statistics.
5. Bui, T. D., et al. (2016). *Deep Gaussian processes for regression using approximate expectation propagation*. ICML 2016.
