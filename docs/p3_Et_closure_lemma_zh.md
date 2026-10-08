# Paper 3 × DP：`E_t` 的 movement-coupled 闭合引理草案

> 文档状态：候选理论草案。本文档把“有界 EControl 误差能否闭合”写成一个可以逐项证明和实验核查的条件式引理。它不把条件式结论写成无条件收敛定理，也不声称 Paper 3 的无投影 EControl 引理可以直接覆盖当前有界投影机制。
>
> 主线：全参与 `q=1`、客户端级中心 DP、每轮发布当前干净聚合估计加 fresh Gaussian noise、有界本地 `h/e/r` 状态、Top-K、Composite Dual Averaging、真实迭代。
>
> 当前明确排除：带噪差分的服务器端二次积分、最终未压缩上传、虚拟迭代作为最终理论对象、固定大小客户端抽样的主定理、原始无界 EControl 状态的时间无关敏感度。

---

## 1. 目的与核心结论

真实迭代分析中的累计近似误差定义为

\[
E_t=\sum_{s=0}^{t-1}(c_s+\rho_s+\beta_s),\qquad E_0=0,
\]

其中：

- `c_s` 是 Top-K 和 EControl 跟踪误差；
- `rho_s` 是残差缓存、二次裁剪和投影造成的近似误差；
- `beta_s` 是样本梯度 clipping bias。

需要控制的量是

\[
Q_T:=\sum_{t=1}^{T}\mathbb E\|E_t\|_2^2.
\]

本草案的目标不是直接声称 `Q_T = O(T)`，而是把它拆成三个接口：

1. **EControl 能量接口**：累计 `c_s` 的能量由本地输入变化、投影残差和初始状态控制；
2. **movement 接口**：本地输入变化由真实迭代 movement、局部随机变化和投影残差控制；
3. **DA 吸收接口**：Paper 3 真实迭代不等式左侧的 movement 项足够大，可以吸收由第 1、2 步反馈回来的 movement 项。

在这些接口成立时，才能得到条件式结论

\[
Q_T\le K_E T.
\]

这里的 `K_E` 依赖于 Top-K 收缩率、状态投影半径、梯度 clipping bias、movement 常数、DA 正则参数和 DP 噪声日程。固定总隐私预算时，`K_E` 一般还依赖于训练轮数 `T`；因此“固定总 `epsilon` 下的 `O(T)` 误差能量”不能直接当成统一于 `T` 的结论。

---

## 2. 当前机制和记号

### 2.1 客户端输入与有界状态

第 `t` 轮客户端 `i` 的样本梯度先进行 clipping，得到本地平均梯度 `v_{i,t}`。残差缓存和 EControl 输入为

\[
\bar r_{i,t}=\operatorname{clip}_{B_r}(r_{i,t}),
\]

\[
u_{i,t}=\operatorname{clip}_{C_g}(v_{i,t}+\bar r_{i,t}),
\]

\[
r_{i,t+1}=\operatorname{clip}_{B_r}
\big(v_{i,t}+\bar r_{i,t}-u_{i,t}\big).
\]

EControl 更新写成

\[
\bar h_{i,t}=\operatorname{clip}_{B_h}(h_{i,t-1}),
\qquad
\bar e_{i,t}=\operatorname{clip}_{B_e}(e_{i,t}),
\]

\[
\delta_{i,t}=u_{i,t}-\bar h_{i,t}-\eta\bar e_{i,t},
\]

\[
q_{i,t}=\operatorname{TopK}_k
\left(\operatorname{clip}_{C_\Delta}(\delta_{i,t})\right),
\]

\[
h_{i,t}=\operatorname{clip}_{B_h}(h_{i,t-1}+q_{i,t}),
\]

\[
e_{i,t+1}=\operatorname{clip}_{B_e}
\big(\bar e_{i,t}+h_{i,t}-u_{i,t}\big).
\]

由机制定义直接得到

\[
\|u_{i,t}\|\le C_g,
\quad \|h_{i,t}\|\le B_h,
\quad \|e_{i,t}\|\le B_e,
\quad \|r_{i,t}\|\le B_r.
\]

这四个有界性结论是已实现机制的确定性事实；它们不等于已经证明了 `E_t` 的时间无关界。

### 2.2 干净聚合与 fresh DP release

全参与 `q=1` 下，服务器的干净状态聚合为

\[
H_t=\frac1n\sum_{i=1}^n h_{i,t}.
\]

每轮公开

\[
y_t=H_t+z_t,
\qquad
z_t\sim\mathcal N(0,\sigma_{\mathrm{DP},t}^2I_d),
\]

且 `z_t` 在给定历史后均值为零、与其他轮次 fresh 独立。

当前引理不适用于

\[
y_t=\Delta_t+z_t,
\qquad
\widehat H_t=\widehat H_{t-1}+y_t,
\]

因为那会让 DP 噪声形成随机游走，再被 Dual Averaging 累积一次。那种机制的 DP 方差会出现近似 `T^3` 的二次累积项，不能套用下面的闭合式。

### 2.3 Top-K 收缩

令 `delta=k/d`。确定性 Top-K 满足

\[
\|\operatorname{TopK}_k(s)-s\|_2^2
\le (1-\delta)\|s\|_2^2.
\]

记

\[
\chi_\delta:=\sqrt{1-\delta}.
\]

Paper 3 无投影设置中的 `eta` 可取一个依赖于 `delta` 的小常数；在当前草案中，`eta` 还必须与 `B_h,B_e,C_g,C_delta` 和投影残差一起核对，不能直接把无投影参数当作已经正确的有界机制参数。

---

## 3. 误差分解

定义无偏参考 oracle `g_t` 和 clipping bias `beta_t`：

\[
\bar v_t=\frac1n\sum_i v_{i,t},
\qquad
\beta_t=\mathbb E[\bar v_t\mid\mathcal F_t]-\nabla f(x_t),
\]

\[
g_t=\bar v_t-\beta_t,
\qquad
\mathbb E[g_t\mid\mathcal F_t]=\nabla f(x_t).
\]

令

\[
\rho_t=\frac1n\sum_i(u_{i,t}-v_{i,t}),
\]

并令 `c_t` 汇总 Top-K、EControl 状态截断及跟踪误差，使得

\[
H_t=\bar v_t+\rho_t+c_t.
\]

定义有效随机 oracle

\[
g_t^{\mathrm{eff}}=g_t+z_t.
\]

于是，干净发布对象满足精确分解

\[
y_t-g_t^{\mathrm{eff}}
=c_t+\rho_t+\beta_t.
\]

因此

\[
E_t=\sum_{s<t}(c_s+\rho_s+\beta_s).
\]

DP 噪声 `z_t` 进入有效 oracle 方差：

\[
\mathbb E\left[
\|g_t^{\mathrm{eff}}-\nabla f(x_t)\|^2
\mid \mathcal F_t
\right]
\le
\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2,
\]

而不是同时进入 `E_t`。把 `z_t` 同时放进随机 oracle 项和累计误差项会重复计算 DP 噪声。

记

\[
\mathfrak C_t=\mathbb E\left\|\sum_{s<t}c_s\right\|^2,
\]

\[
\mathfrak R_t=\mathbb E\left\|\sum_{s<t}\rho_s\right\|^2,
\qquad
\mathfrak B_t=\mathbb E\left\|\sum_{s<t}\beta_s\right\|^2.
\]

则有已证的代数不等式

\[
\mathbb E\|E_t\|^2
\le
3(\mathfrak C_t+\mathfrak R_t+\mathfrak B_t).
\]

---

## 4. 需要区分的三种能量

为了避免把“逐轮误差有界”“累计误差有界”和“累计误差能量为 `O(T)`”混为一谈，定义：

### 4.1 本地输入变化

\[
D_t
:=
\frac1n\sum_{i=1}^n
\mathbb E\|u_{i,t+1}-u_{i,t}\|^2.
\]

`D_t` 包含三种来源：

1. 真实迭代移动造成的梯度变化；
2. minibatch 或 stochastic oracle 的局部变化；
3. `r`、`h`、`e` 投影和 clipping 造成的输入变化。

### 4.2 状态投影残差

定义 raw 更新和投影后更新之间的残差：

\[
p^h_{i,t}=h_{i,t-1}+q_{i,t}-h_{i,t},
\]

\[
p^e_{i,t+1}=\bar e_{i,t}+h_{i,t}-u_{i,t}-e_{i,t+1},
\]

\[
p^r_{i,t+1}=v_{i,t}+\bar r_{i,t}-u_{i,t}-r_{i,t+1}.
\]

令

\[
P_t:=\frac1n\sum_i\mathbb E\left[
\|p^h_{i,t}\|^2
+\|p^e_{i,t+1}\|^2
+\|p^r_{i,t+1}\|^2
\right].
\]

状态有界并不自动推出 `P_t=0`。只有当 raw 状态始终落在对应球内时，相关投影残差才为零。

### 4.3 局部随机变化和 clipping bias

可将 `D_t` 拆成一个 movement 部分和一个局部噪声部分。候选定义为

\[
D_t
\le
K_x M_t+K_\xi\Xi_t+K_pP_t,
\tag{MC-1}
\]

其中

\[
M_t:=\mathbb E\|x_{t+1}-x_t\|^2,
\]

`Xi_t` 是给定模型变化后的局部 stochastic variation，例如样本梯度差异、客户端批次差异或条件方差项。

`beta_t` 不应只用逐轮范数粗略替代。需要单独控制

\[
\mathcal B_T:=\sum_{t=1}^{T}\mathfrak B_t
=\sum_{t=1}^{T}
\mathbb E\left\|\sum_{s<t}\beta_s\right\|^2.
\]

仅有 `||beta_t|| <= b` 时，最坏情况下 `mathfrak B_t <= t^2 b^2`，从而 `mathcal B_T=O(T^3)`。因此 clipping bias 的逐轮有界性不够，需要 bias correction、可求和 bias、条件均值抵消或直接假设 `mathcal B_T=O(T)`。

---

## 5. 候选 EControl 能量闭合引理

### 5.1 引理需要的假设

设 `S_t` 是 EControl 跟踪误差状态，包含每个客户端的 `h/e/r` 差异以及累计压缩误差所需的辅助量。其具体坐标化方式需要从当前机制的更新式中正式定义；此处只要求它满足以下候选 Lyapunov 结构。

假设存在非负 Lyapunov 函数

\[
\mathcal L_t
\ge a_c\mathbb E\left\|\sum_{s<t}c_s\right\|^2,
\qquad a_c>0,
\]

以及常数 `mu_c>0, K_D>=0, K_P>=0, K_Xi>=0`，使得对每个 `t`：

\[
\mathcal L_{t+1}-\mathcal L_t
+\mu_c\mathfrak C_{t+1}
\le
K_DD_t+K_PP_t+K_\Xi\Xi_t.
\tag{EF-1}
\]

如果 `L_0` 只由初始化状态决定，则把

\[
\mathcal I_0:=\mathcal L_0/a_c
\]

看作初始 EControl 能量。

### 5.2 对 `EF-1` 求和

对 `t=0,...,T-1` 求和并使用 `L_T>=0`，得到

\[
\sum_{t=1}^{T}\mathfrak C_t
\le
\frac{\mathcal L_0}{\mu_c}
+\frac{K_D}{\mu_c}\sum_{t=0}^{T-1}D_t
+\frac{K_P}{\mu_c}\sum_{t=0}^{T-1}P_t
+\frac{K_\Xi}{\mu_c}\sum_{t=0}^{T-1}\Xi_t.
\tag{EF-2}
\]

记

\[
\kappa_0=\frac{\mathcal L_0}{\mu_c},
\quad
\kappa_D=\frac{K_D}{\mu_c},
\quad
\kappa_P=\frac{K_P}{\mu_c},
\quad
\kappa_\Xi=\frac{K_\Xi}{\mu_c}.
\]

则

\[
\mathcal C_T:=\sum_{t=1}^{T}\mathfrak C_t
\le
\kappa_0+\kappa_D\sum_tD_t
+\kappa_P\sum_tP_t
+\kappa_\Xi\sum_t\Xi_t.
\tag{EF-3}
\]

这就是当前最希望证明的有界 EControl 接口。它比直接声称 `||e_t-e'_t||` 时间无关有界更合适：误差状态可以随时间移动，但其能量由输入变化和投影残差供给。

### 5.3 为什么 `EF-1` 仍是候选而不是已证

`EF-1` 至少需要同时处理：

- Top-K 的非线性和坐标选择变化；
- `h/e/r` 的三处投影；
- `hbar/ebar/rbar` 使用的是投影后的状态；
- `u=clip_{C_g}(v+rbar)` 的二次 clipping；
- 不同客户端的状态不同步；
- `c_t` 是聚合后的量，而 `P_t,D_t` 是本地平均能量；
- 在 DP 下，`x_t` 的 movement 会改变下一轮 `v_{i,t}`，从而改变 EControl 输入。

因此，Paper 3 无投影 EControl 的收缩引理不能直接作为 `EF-1` 的证明。需要先在投影残差为零的特例中恢复原有收缩结构，再逐一加回三个投影残差。

---

## 6. movement 耦合接口

### 6.1 候选输入变化不等式

假设单样本梯度在模型变量上是 `L_u`-Lipschitz，clip 映射是非扩张的，且残差缓存映射的增量可由 `r` 投影残差控制，则候选有

\[
D_t
\le
K_xM_t+K_\xi\Xi_t+K_pP_t.
\tag{MC-1 revisited}
\]

例如，当

\[
v_{i,t}=\nabla \ell_i(x_t;\zeta_{i,t})
\]

并且模型变量上的梯度 Lipschitz 常数为 `L_u` 时，第一项通常可取与 `L_u^2` 成正比。这里不直接指定一个统一数值，因为 softmax、least squares 和 simplex logistic 的 `L_u` 不同，且随机批次差异需要单独计入 `Xi_t`。

### 6.2 Paper 3 真实迭代 movement 预算

令

\[
\mu_t:=\frac{\gamma_{t-1}-2L}{4},
\qquad
\mu:=\inf_t\mu_t>0.
\]

Paper 3 真实迭代分析给出候选强形式

\[
\sum_{t=0}^{T-1}
\mathbb E\left[
F(x_{t+1})-F(x^\star)+\mu_tM_t
\right]
\le
\frac{\gamma_{T-1}}2R_0^2
+\frac{6}{\underline\gamma}(\mathcal C_T+\mathcal R_T+\mathcal B_T)
+\frac{2}{\underline\gamma}\mathcal V_T,
\tag{DA-1}
\]

其中

\[
\underline\gamma=\inf_t\gamma_{t-1},
\qquad
\mathcal V_T=\sum_{t=0}^{T-1}
\left(\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2\right).
\]

由于 `x^star` 是最优解，目标差非负，于是可以从右侧得到一个 movement 预算：

\[
\mu\mathcal M_T
\le
\frac{\gamma_{T-1}}2R_0^2
+\frac{6}{\underline\gamma}(\mathcal C_T+\mathcal R_T+\mathcal B_T)
+\frac{2}{\underline\gamma}\mathcal V_T,
\tag{DA-2}
\]

其中

\[
\mathcal M_T:=\sum_{t=0}^{T-1}M_t.
\]

这一步使用了真实迭代不等式左侧的 movement 项；它是闭合证明中吸收 movement 反馈的关键。

### 6.3 代入 EControl 能量接口

由 `EF-3` 和 `MC-1`：

\[
\mathcal C_T
\le
\kappa_0
+\kappa_DK_x\mathcal M_T
+(\kappa_DK_\xi+\kappa_\Xi)\sum_t\Xi_t
+(\kappa_DK_p+\kappa_P)\sum_tP_t.
\tag{MC-2}
\]

记

\[
\mathcal N_T:=\sum_t\Xi_t,
\qquad
\mathcal P_T:=\sum_tP_t,
\]

\[
K_N=\kappa_DK_\xi+\kappa_\Xi,
\qquad
K_P=\kappa_DK_p+\kappa_P.
\]

则

\[
\mathcal C_T
\le
\kappa_0+\kappa_DK_x\mathcal M_T+K_N\mathcal N_T+K_P\mathcal P_T.
\tag{MC-3}
\]

代入 `DA-2`，得到

\[
\left(
\mu-\frac{6\kappa_DK_x}{\underline\gamma}
\right)\mathcal M_T
\le
\frac{\gamma_{T-1}}2R_0^2
+\frac{6}{\underline\gamma}
\left[
\kappa_0+K_N\mathcal N_T+K_P\mathcal P_T
+\mathcal R_T+\mathcal B_T
\right]
+\frac{2}{\underline\gamma}\mathcal V_T.
\tag{MC-4}
\]

因此，movement 可被吸收的候选条件是

\[
\boxed{
\mu>\frac{6\kappa_DK_x}{\underline\gamma}
}
\tag{Absorb}
\]

并且 `gamma_t` 必须满足 `gamma_{t-1}>2L`。若使用常数 `gamma`，则 `mu=(gamma-2L)/4`，一个简单但偏保守的充分条件是

\[
\frac{6\kappa_DK_x}{\gamma}
<
\frac{\gamma-2L}{4}.
\]

这说明 `gamma` 既要足够大以克服目标函数的 smoothness，也要足够大以吸收 EControl 对 movement 的反馈。该条件是候选充分条件，不是当前已经证明的最优阈值。

---

## 7. 条件式 `Q_T <= K_E T` 结论

### 7.1 可接受的逐轮条件

若存在与 `T` 无关的常数 `K_x,K_N,K_P,kappa_0,kappa_D`，并且满足：

1. `EF-1` 成立；
2. `MC-1` 成立；
3. `Absorb` 成立；
4. `sum_t Xi_t <= K_Xi T`；
5. `sum_t P_t <= K_P0 T`；
6. `mathcal R_T <= K_R T`；
7. `mathcal B_T <= K_B T`；
8. `V_T <= K_V T`，其中
   \[
   \mathcal V_T=\sum_t(\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2),
   \]
9. `gamma_{t-1}` 有统一下界并满足 `Absorb`；

这些条件必须同时成立。特别是，`V_T=O(T)` 或固定逐轮 `sigma` 本身不能替代 `mathcal B_T=O(T)`。如果 clipping bias 持续同向，固定 `sigma` 仍然可能产生超线性的 `Q_T`。

则由 `MC-4` 有

\[
\mathcal M_T\le K_M T+K_{M,0},
\]

再由 `MC-3` 有

\[
\mathcal C_T\le K_C T+K_{C,0}.
\]

由

\[
\mathbb E\|E_t\|^2
\le3(\mathfrak C_t+\mathfrak R_t+\mathfrak B_t)
\]

得到

\[
\boxed{
Q_T=\sum_{t=1}^{T}\mathbb E\|E_t\|^2
\le K_E T+K_{E,0}.
}
\tag{Closure}
\]

当初始状态固定且 `T>=1` 时，可以把 `K_{E,0}` 吸收到 `K_E` 中，写成 `Q_T <= K_E' T`。但 `K_E'` 依赖于上述所有常数和 DP 噪声日程；它不是仅由 `delta=k/d` 决定的常数。

### 7.2 从 `Q_T` 到条件式优化界

若 `Closure` 成立，且 `gamma_t=gamma`，则真实迭代平均值满足候选形式

\[
\mathbb E[F(\bar x_T^{\mathrm{real}})-F(x^\star)]
\le
\frac{\gamma R_0^2}{2T}
+\frac{2K_E}{\gamma}
+\frac{2}{\gamma T}\mathcal V_T
+\frac{6}{\gamma T}(\mathcal R_T+\mathcal B_T).
\]

如果 `gamma=c sqrt(T)`，且 `K_E`、`mathcal V_T/T`、`mathcal R_T/T`、`mathcal B_T/T` 均在所讨论的 horizon 范围内有界，则可以得到形式上的 `O(T^{-1/2})` 界。这个结论仍然是条件式的，因为 `Closure` 和 `EF-1` 尚未完成证明。

---

## 8. 固定总隐私预算时为什么 `K_E` 依赖 `T`

### 8.1 固定逐轮噪声与固定总隐私预算是两种不同实验

若每轮使用固定 `sigma_DP,t^2 <= sigma_bar^2`，则

\[
\mathcal V_T
\le
T(\bar\sigma_0^2+d\bar\sigma_{\mathrm{DP}}^2)=O(T).
\]

这与 `Q_T=O(T)` 的闭合假设相容，但总隐私损失会随轮数增长。对于 `q=1` 的 Gaussian 机制，RDP 或 zCDP 账本的总隐私成本大致按轮数线性累积；转换到 `epsilon` 后，典型尺度为 `sqrt(T)`。

相反，若固定总 `(epsilon,delta)`，则每轮噪声必须随着 `T` 增大。以等噪声分配为例，Gaussian/RDP 账本给出近似关系

\[
\sigma_{\mathrm{DP}}^2
\asymp
\frac{\Delta^2T\log(1/\delta)}{\epsilon^2},
\]

其中 `Delta=2B_h/n` 是当前有界聚合发布的单轮敏感度。于是

\[
\mathcal V_T^{\mathrm{DP}}
=d\sum_{t=0}^{T-1}\sigma_{\mathrm{DP},t}^2
\asymp
\frac{d\Delta^2T^2\log(1/\delta)}{\epsilon^2}.
\]

此时平均逐轮方差本身为 `O(T)`，不能再把 `V_T/T` 当作 horizon 无关常数。

### 8.2 对 movement 和 `E_t` 的影响

将上式代入 `MC-4`，即使投影残差和局部随机变化仍为 `O(T)`，右侧也包含 `O(T^2)` 的 DP 方差项。因此安全的条件式界只能写成

\[
\mathcal M_T=O(T^2),
\qquad
\mathcal C_T=O(T^2),
\qquad
Q_T=O(T^2),
\]

即

\[
Q_T\le K_E(T)T,
\qquad
K_E(T)=O(T)
\]

作为最保守的一阶结论。实际反馈可能更复杂：DP 噪声使 `x_t` 抖动，抖动增大 `D_t`，再通过 EControl 反馈增大 `C_t`。因此不应把 `K_E(T)=O(T)` 写成精确渐近定理，除非完成针对具体 `EF-1` 和 `MC-1` 的闭合证明。

### 8.3 为什么模拟中会比线性更差

当前盒约束实验的固定总隐私预算扫描中，`T=25,50,100,200` 时 DP-TopK 的平均 movement 和 `Q_T/T` 都快速增加。这与以下机制一致：

1. 总隐私预算固定，逐轮 `sigma` 增大；
2. movement 增大，局部输入变化 `D_t` 增大；
3. EControl 误差能量增大；
4. 真实 DA 迭代又受到更大的 DP oracle 方差影响。

补充的固定噪声扫描显示，即使把 `sigma` 固定在约 `0.6932`，`T=25,50,100,200` 时 `Q_T/T` 仍约为 `158.7,428.9,1212.3,2568.7`。这说明固定逐轮 DP 噪声并不足以保证 `Q_T=O(T)`；持续的 clipping bias 也可能被累积进 `E_t`，并通过 movement 反馈放大。

这些数值结果是对闭合引理的压力测试，不是 `EF-1` 的证明。论文中应报告 `sigma_t`、`D_t`、`P_t`、`M_t`、`mathfrak C_t`、`mathfrak R_t`、`mathfrak B_t` 和 `Q_T/T`，而不是只报告最终目标函数。

### 8.4 保持 `K_E` 近似稳定的可行方法

在全参与主线中，若希望 `K_E` 在不同 horizon 间近似稳定，需要至少满足下列之一：

- 允许总隐私预算随 `T` 增长；
- 减少发布轮数而增加每轮本地计算；
- 使用更适合 prefix workload 的相关噪声机制，但必须重新处理参与和敏感度结构；
- 使用 privacy filter/odometer，在达到固定预算的 knee 后停止；
- 将 utility 曲线报告为“固定总 bits、固定总 privacy”下的最优轮数，而不是强行延长训练。

当前主线优先采用最后两种报告方式，不把固定总隐私下的长 horizon 误写为标准收敛加速。

---

## 9. clipping bias 的单独条件

### 9.1 逐轮有界不够

由样本梯度 clipping，只能直接得到某个逐轮界

\[
\|\beta_t\|\le B_\beta.
\]

但

\[
\mathfrak B_t
=\mathbb E\left\|\sum_{s<t}\beta_s\right\|^2
\]

在最坏情形下可以达到 `t^2 B_beta^2`。因此不能从 `beta_t` 逐轮有界直接推出 `mathcal B_T=O(T)`。

### 9.2 三种可接受的闭合条件

论文可以选择以下任一条件，但必须明确写出：

1. **无偏或条件抵消**：`beta_t` 是条件均值为零的 residual；
2. **可求和 bias**：
   \[
   \sum_{t=1}^{T}\mathfrak B_t\le K_B T;
   \]
3. **二次 clipping residual feedback**：把 clipping residual 单独缓存并在不破坏敏感度的前提下反馈，使累计 bias 替换为一个有界 residual state。

当前机制的 `r` buffer 可以减少 `v+r` 二次 clipping 带来的残差，但它不能自动消除样本梯度第一次 clipping 的 bias。因此第一种和第三种不能被混为一谈。

### 9.3 两个必须分开的定理版本

固定或受控的逐轮 DP 噪声下，建议把研究结果分成两个版本。

#### 版本 A：bounded/clipped objective

把研究目标直接定义为被 clipping oracle 优化的目标，或假设 clipping 不产生目标梯度偏差：

\[
\beta_t=0.
\]

此时

\[
\mathfrak B_t=0,
\qquad
\mathcal B_T=0,
\]

闭合只需处理 EControl 跟踪、movement、局部随机变化和投影残差。若 `EF-1`、`MC-1`、`Absorb` 以及 `mathcal V_T, mathcal P_T, mathcal R_T, mathcal N_T=O(T)` 成立，则可以候选地推出

\[
Q_T\le K_E T.
\]

这是一条更干净、适合先完成机制定理的版本，但它认证的是 clipped/bounded objective，不能直接替换成原始未裁剪目标。

#### 版本 B：原始目标 + clipping bias

若目标仍是原始损失，则必须额外假设 clipping bias 的累计行为，例如：

\[
\mathcal B_T
=
\sum_{t=1}^{T}
\mathbb E\left\|\sum_{s<t}\beta_s\right\|^2
\le K_B T,
\tag{Bias-sum}
\]

或者证明 `beta_t` 是条件零均值 residual、具有可求和衰减，或由单独的 clipping-residual feedback 变成一个已闭合的有界状态。

在没有 `Bias-sum` 或等价条件时，即使 `sigma_DP,t` 固定、`h/e/r` 均有界，也不能推出 `Q_T=O(T)`。最坏情况下，若 `beta_t` 同方向且 `||beta_t||` 近似为常数，则

\[
\mathfrak B_t=\Theta(t^2),
\qquad
\mathcal B_T=\Theta(T^3).
\]

因此版本 B 必须把 clipping bias 作为原始目标与 clipped objective 之间的独立偏差项报告，或者在机制中增加不破坏敏感度的 clipping-residual feedback。版本 A 和版本 B 的实验结果、定理假设和结论不能合并成一个无条件定理。

---

## 10. 已证明、候选、需要验证

### 10.1 已证明或直接由机制得到

- `q=1` 下客户端全参与；
- `||h_{i,t}||<=B_h`、`||e_{i,t}||<=B_e`、`||r_{i,t}||<=B_r`；
- 当前干净聚合加 fresh Gaussian release 的条件均值和二阶矩；
- 单轮客户端替换敏感度 `2B_h/n`；
- Top-K 的确定性收缩不等式；
- `y_t-g_t^eff=c_t+rho_t+beta_t` 的代数分解；
- `E_t` 三项平方分解；
- DP 噪声应进入有效 oracle variance，而不是再次进入 `E_t` 的累计误差。

### 10.2 候选引理或候选充分条件

- `EF-1` 的 EControl Lyapunov 递推；
- `MC-1` 的局部输入变化到 movement 的不等式；
- `DA-1` 在当前有界投影机制和真实迭代下的完整常数核对；
- `Absorb` 对 movement 反馈的吸收条件；
- 在 `V_T=O(T)`、`mathcal P_T=O(T)`、`mathcal B_T=O(T)` 下的 `Q_T<=K_E T`；固定 `sigma` 只能帮助前一项，不能替代 `mathcal B_T=O(T)`；
- `gamma`、`eta`、`B_h,B_e,B_r,C_g` 的统一可行参数区间。

### 10.3 必须验证的内容

1. **无投影回归**：设 `p^h=p^e=p^r=0`，验证 `EF-1` 能够恢复 Paper 3 EControl 的收缩结构；
2. **单客户端确定性轨迹**：用固定输入序列比较 `C_t`、局部状态 Lyapunov 和 `D_t`；
3. **movement 扫描**：人工注入不同大小的 `x_{t+1}-x_t`，估计 `K_x`；
4. **投影残差扫描**：逐渐缩小 `B_h,B_e,B_r`，检查 `P_t` 是否按 `MC-1` 进入；
5. **固定 `sigma` horizon 扫描**：检验 `Q_T/T` 是否稳定；
6. **固定总 epsilon horizon 扫描**：报告 `K_E(T)=Q_T/T` 的增长，而不是把增长误判成优化不稳定；
7. **clipping bias ablation**：无 clipping、仅输入 clipping、加 residual feedback 三组对照；
8. **DA 常数检查**：对 box、simplex、softmax+`l1` 分别核查 `L`、prox 非扩张性和 movement 系数。

---

## 11. 论文中可安全使用的表述

当前可以写：

> 在全参与、状态显式有界、每轮发布当前干净聚合估计并加入新鲜 Gaussian 噪声的机制下，Paper 3 的真实迭代分析可化为一个 movement-coupled 条件界。若有界 EControl 的 Lyapunov 能量由本地输入变化和投影残差线性供给，且该 movement 反馈能被 Dual Averaging 的强正则项吸收，则累计近似误差能量满足 `Q_T <= K_E T`。在固定总隐私预算下，逐轮 DP 方差随 horizon 增长，因此 `K_E` 一般依赖于 `T`，并不能直接得到统一 horizon 的无条件 `O(T^{-1/2})` 结论。

当前不能写：

- “有界 `h/e/r` 自动推出 `E_t` 时间无关有界”；
- “Top-K contractivity 自动推出当前投影 EControl 的 Paper 3 常数”；
- “固定总 epsilon 下增加轮数仍保持相同 `K_E`”；
- “clipping bias 只要逐轮有界就会产生 `O(T)` 累计能量”；
- “Candidate Theorem A 已经是无条件真实迭代收敛定理”。

---

## 12. 下一步证明顺序

1. 先把 `S_t` 和 `L_t` 的坐标化定义写进代码和附录，不能停留在抽象状态；
2. 在无投影、无 clipping residual 的特例中，逐行恢复 Paper 3 EControl 收缩证明；
3. 一次只加入 `h`、`e`、`r` 中的一个投影残差；
4. 证明 `MC-1`，分别用于 softmax、box least squares 和 simplex logistic；
5. 将 `EF-3` 代入 Paper 3 真实迭代不等式，核对吸收系数；
6. 在固定 `sigma` 和固定总 epsilon 两种 protocol 下分别估计 `K_E`；
7. 只有当前述条件通过后，才把 FashionMNIST 作为高维实验，而不是先用大模型实验掩盖闭合缺口。

## 13. 实现对齐后的 signed residual 约定（2026-10-08）

当前三个模拟器把投影残差按 `raw - new` 存储：

```text
p_e = e_raw - e_new
p_r = r_raw - r_new
p_h = h_raw - h_new
```

在 e_0=r_0=0 且全参与时，逐轮代数恒等式给出：

```text
sum_{s <= t} c_s   = mean(e_t) + sum_{s <= t} mean(p_e,s)
sum_{s <= t} rho_s = -mean(r_t) - sum_{s <= t} mean(p_r,s)
```

如果证明草稿把投影残差定义成 `new - raw`，则相同恒等式写成：

```text
sum c_s   = mean(e_t) - sum mean(p_e,s)
sum rho_s = -mean(r_t) + sum mean(p_r,s)
```

这只是符号约定，但在 projection residual 不为零时不能混用。独立 sign-stress 运行（softmax 与 simplex 都缩小 Cg、Bh、Be、Br 以激活投影）在正确约定下的 telescoping 误差仍小于 7e-16；把 `raw-new` 当成 `new-raw` 会产生 1e-1 到 1 量级的误差。对应记录见 [`tele_scope_sign_stress.json`](../experiments/tele_scope_sign_stress.json)。后续证明统一使用上面的 `raw - new` 定义，或在代码中先显式取负，不能再把投影残差范数放入 `E_t` 的第一坐标。
