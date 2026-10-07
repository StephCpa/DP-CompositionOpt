# Paper 3 × central DP：跨凸目标条件检查表

## 目的与主线

本文档把四个模拟器使用的同一主机制写成可以逐目标核查的条件表：客户端级中心 DP、全参与 `q=1`、每轮 fresh Gaussian noise 加在当前 clean aggregate release 上、bounded EControl、真实迭代，Composite Dual Averaging 的 prox 处理约束或正则项。服务器状态不整合带噪差分，最终报告真实迭代。每轮发布对象若是每客户端有界状态的平均值，替换邻接下的 L2 敏感度取 `2 B_h / n`；无压缩 dense DA 的对应半径是 `C0`。

这份表只建立候选定理所需的目标函数常数和 prox 几何，不能替代对 Paper 3 真实迭代误差项的闭合证明。关键未解决条件仍是 movement-coupled 累计误差 `E_t` 的二阶和。

## 逐目标条件

| 目标 | 可微部分 `f` | 复合部分 `psi` / prox | 可用的光滑性与梯度半径 | 需要在证明中保留的项 |
|---|---|---|---|---|
| Softmax + `l1` | `f(W)=E[log sum_c exp(x^T w_c)-x^T w_y]` | `lambda ||W||_1`；坐标 soft-threshold | 若 `||x||<=R`，可取 `L <= R^2/2`，单样本梯度 Frobenius 范数 `<= sqrt(2) R` | clipping bias `beta_t`、Top-K residual `c_t`、DP noise variance、真实迭代 movement |
| Box least squares | `f(x)=E[1/2 (a^T x-b)^2]` | indicator of `[-B,B]^d`；coordinate clipping / box projection | `L <= ||A||_op^2/m`；若 `||a||<=R_x`, `|b|<=B_y`, `||x||<=B sqrt(d)`, per-example gradient `<= R_x(R_x B sqrt(d)+B_y)` | box projection residual `rho_t`、clipping bias、DP-induced movement |
| Simplex logistic | `f(w)=E[log(1+exp(x^T w))-y x^T w]` | indicator of `Delta^d`；exact Euclidean simplex projection | 若 `||x||<=R`，`L <= R^2/4`，单样本梯度范数 `<= R`；在 simplex 上 `||w||_2<=1` | simplex projection residual（理论上为零，数值上核查）、clipping bias、DP-induced movement |
| 二分类 logistic（无约束基线） | `f(w)=E[log(1+exp(-y x^T w))]`（按实现的标签约定） | 可为 `l1` 或无正则 | `L <= R^2/4`，单样本梯度范数 `<= R` | 与 simplex 版本相同，但缺少有界域带来的半径控制 |

其中 box least squares 的 `L` 也可以用样本矩阵的谱范数直接计算，避免用过松的逐样本上界。softmax 的 `l1` 项不改变 `f` 的 smoothness，但需要在 DA prox 中显式保留正则项。

## 统一的 Candidate Theorem A 账本

设 `z_t` 是每轮 fresh Gaussian aggregate noise，`g_t` 是客户端梯度平均，`c_t` 是压缩/状态误差，`rho_t` 是 prox 或投影残差，`beta_t` 是梯度 clipping bias。用

`E_t = sum_{s<t} (c_s + rho_s + beta_s)`

表示累计误差；DP 随机量应首先进入有效 oracle 的方差

`sigma_eff,t^2 = sigma_0,t^2 + d sigma_DP,t^2`，

而不是直接当成必然线性累积的偏差。它通过真实迭代 movement 影响后续 EControl 状态，所以强形式中需要保留类似

`sum_t a_t ||x_{t+1}-x_t||^2`

的耦合项。

在逐轮噪声方差固定或受控、且可证明

`sum_{t=1}^T E ||E_t||^2 <= K_E T`

时，Paper 3 的真实迭代不等式才可能给出标准平均收敛率；这里 `K_E` 必须显式依赖 `B_h, B_e, B_r, L, C_g` 和 DP 噪声日程。若固定总隐私预算后令 T 任意增大，账本会使每轮 sigma 增大，不能继续假设同一个 `K_E`。

## 隐私与通信记录的最低字段

每个实验点必须保存：

- `q`、参与模型和 accountant 类型；
- `sensitivity`、每轮 `sigma_t` 或其调度；
- `B_h, B_e, B_r, C0, Cg` 与约束半径；
- state age 的均值、P90、最大值；
- prefix/tail clipping residual；
- `c_t, rho_t, beta_t, movement, ||E_t||, ||E_t||^2` 的逐轮记录；
- 每客户端比特数（Top-K 需要同时计值和索引）；
- 目标函数、测试指标和至少三个独立随机种子。

## 目前可以声称的范围

1. 同一机制接口已在 softmax+`l1`、box least squares、simplex logistic 三类不同复合结构上通过代码、有限差分/可行性检查和小规模多种子实验。
2. 这些实验支持把 Top-K 描述为通信效率分支；不能把效用差异单独归因于压缩去噪，因为 bounded aggregate radius 也会改变敏感度和 Gaussian 噪声尺度。
3. 真实迭代理论仍应写成条件式 Candidate Theorem A；没有完成 `sum_t E||E_t||^2` 的闭合前，不写无条件 `O(T^{-1/2})`。
4. FashionMNIST 应在这张条件表逐项实例化并通过一次小规模审计后再开展，避免把神经网络实验混入尚未闭合的凸目标证明。

## 版本 A / 版本 B 的证明入口

当前应将 Candidate Theorem A 拆成两条路线：

- **版本 A（clipped/bounded objective）**：令 `beta_t=0`，或把 clipped oracle 定义为研究目标。然后分别证明 EControl Lyapunov 递推、movement coupling、投影残差项和 fresh DP 方差项。
- **版本 B（原始目标）**：额外证明 `sum_t E||sum_{s<t} beta_s||^2 <= K_beta T`，或者使用条件零均值、衰减可加和或独立 clipping-residual feedback。持续非零的 clipping bias 不能自动并入版本 A 的 `E_t` 闭合。

投影残差应以 signed vector 进入主诊断：

`rho_t = average_i (p^h_{i,t} + p^e_{i,t} + p^r_{i,t})`。

把残差范数同向放入一个坐标只能作为保守压力测试，不应作为最终理论对象。
