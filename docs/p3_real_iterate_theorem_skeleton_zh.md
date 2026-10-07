# Paper 3 × DP：真实迭代 Candidate Theorem A 的正式假设、误差分解与证明骨架

> 文档状态：研究候选稿，不是已经完成的定理证明。  
> 主线范围：全参与 q=1、客户端级中心 DP、有界本地 EControl 状态、服务器发布当前聚合估计加每轮新鲜 Gaussian 噪声、Composite Dual Averaging、只分析真实迭代。  
> 明确排除：虚拟迭代作为最终理论输出、Paper 3 原始算法的最终无压缩上传、把带噪差分在服务器上二次累加、固定大小客户端抽样的主定理、原始无界 EControl 状态的时间无关敏感度。

状态标记：

- [已证/可直接引用]：由 Paper 3 的真实迭代分析、确定性不等式或标准 Gaussian/RDP 组合直接得到。
- [候选/待证]：与当前机制相适配，但尚未完成证明，不能在论文中写成已建立结果。
- [实验支持]：当前模拟器、长时域测试或对抗性搜索提供的数值证据。
- [不主张]：当前版本明确不声称的结果。

## 1. 研究对象和定理目标

### 1.1 复合目标

考虑：

\[
\min_{x\in\operatorname{dom}\psi}
F(x)=f(x)+\lambda\psi(x),
\]

其中 f 是客户端损失的平均值，psi 是 proper、closed、convex 的复合项，lambda 大于等于 0，且最优解 x* 存在。psi 可以是 l1 正则，也可以是凸约束集的指示函数。

服务器使用 Composite Dual Averaging：

\[
G_t=G_{t-1}+y_t,
\]

\[
x_{t+1}
=
\arg\min_{x\in\operatorname{dom}\psi}
\left\{
\sum_{s=0}^{t}
\left[
f(x_s)+\langle y_s,x-x_s\rangle+\lambda\psi(x)
\right]
+
\frac{\gamma_t}{2}\|x-x_0\|_2^2
\right\}.
\]

当前主线取 a_t 等于 1。这样 A_t=t+1，和当前模拟器实现一致。

### 1.2 只认证真实迭代平均值

定义：

\[
\bar x_T^{\mathrm{real}}
=
\frac1T\sum_{t=0}^{T-1}x_{t+1}.
\]

Candidate Theorem A 的目标是给出：

\[
\mathbb E[
F(\bar x_T^{\mathrm{real}})-F(x^\star)
]
\le
\mathcal E_{\mathrm{DA}}
+
\mathcal E_{\mathrm{comp}}
+
\mathcal E_{\mathrm{state}}
+
\mathcal E_{\mathrm{clip}}
+
\mathcal E_{\mathrm{DP}}.
\]

这里所有 x_t 都是算法实际产生的迭代，不构造也不输出 Paper 3 的虚拟迭代。

当前模拟器最后报告的是最后一个真实迭代 x_T；理论认证对象是平均真实迭代。若要认证最后迭代，需要额外的强凸性、单调性或尾部稳定性假设，当前不自动声称这一点。

## 2. 机制和记号

### 2.1 过滤和参与模式

令 F_t 表示第 t 轮开始时的历史，包括模型、所有隐藏状态和过去发布的消息。主线为全参与：

\[
S_t=\{1,\ldots,n\}.
\]

固定大小客户端抽样只作为后续 stress test，不进入 Candidate Theorem A。

### 2.2 本地输入和有界状态

本地每样本裁剪平均为：

\[
v_{i,t}
=
\frac1{b_i}
\sum_j
\operatorname{clip}_{C_0}
(\nabla\ell_i(x_t;z_{ij,t})).
\]

残差缓存：

\[
\bar r_{i,t}=\operatorname{clip}_{B_r}(r_{i,t}),
\]

\[
u_{i,t}=\operatorname{clip}_{C_g}(v_{i,t}+\bar r_{i,t}),
\]

\[
r_{i,t+1}
=
\operatorname{clip}_{B_r}
(v_{i,t}+\bar r_{i,t}-u_{i,t}).
\]

EControl 状态：

\[
\bar h_{i,t}=\operatorname{clip}_{B_h}(h_{i,t-1}),
\qquad
\bar e_{i,t}=\operatorname{clip}_{B_e}(e_{i,t}),
\]

\[
\delta_{i,t}
=
u_{i,t}-\bar h_{i,t}-\eta\bar e_{i,t},
\]

\[
q_{i,t}
=
\operatorname{TopK}_k
\left(
\operatorname{clip}_{C_\Delta}(\delta_{i,t})
\right),
\]

\[
h_{i,t}
=
\operatorname{clip}_{B_h}(h_{i,t-1}+q_{i,t}),
\]

\[
e_{i,t+1}
=
\operatorname{clip}_{B_e}
(\bar e_{i,t}+h_{i,t}-u_{i,t}).
\]

从机制定义直接得到：

\[
\|u_{i,t}\|_2\le C_g,\quad
\|h_{i,t}\|_2\le B_h,\quad
\|e_{i,t}\|_2\le B_e,\quad
\|r_{i,t}\|_2\le B_r.
\]

### 2.3 当前估计的新鲜 Gaussian 发布

服务器内部计算干净聚合：

\[
H_t=\frac1n\sum_{i=1}^n h_{i,t}.
\]

公开：

\[
y_t=H_t+z_t,\qquad
z_t\sim\mathcal N(0,\sigma_{\mathrm{DP},t}^2I_d).
\]

给定 F_t：

\[
\mathbb E[z_t\mid F_t]=0,\qquad
\mathbb E[\|z_t\|_2^2\mid F_t]=d\sigma_{\mathrm{DP},t}^2,
\]

且 z_t 在不同轮次之间独立。

必须禁止下面的错误实现：

\[
y_t=\Delta_t+z_t,\qquad
\widehat H_t=\widehat H_{t-1}+y_t.
\]

这种实现会把带噪差分再次积分，形成随机游走，并让 Dual Averaging 对 DP 噪声二次累加。Candidate Theorem A 只适用于发布当前干净估计加新鲜噪声。

## 3. 正式假设

### A1. 目标函数 [已证条件，沿用 Paper 3]

f 和 psi 在 dom psi 上 convex、closed、proper；f 的梯度满足：

\[
\|\nabla f(x)-\nabla f(x')\|_2
\le
L\|x-x'\|_2.
\]

F 存在最优解 x*。

### A2. 无偏参考 oracle [已证条件，沿用 Paper 3]

定义：

\[
\bar v_t=\frac1n\sum_i v_{i,t},
\]

以及给定 F_t 的梯度裁剪偏差：

\[
\beta_t
=
\mathbb E[\bar v_t\mid F_t]-\nabla f(x_t).
\]

定义理论参考 oracle：

\[
g_t=\bar v_t-\beta_t.
\]

于是：

\[
\mathbb E[g_t\mid F_t]=\nabla f(x_t).
\]

假设：

\[
\mathbb E[
\|g_t-\nabla f(x_t)\|_2^2\mid F_t
]
\le \sigma_{0,t}^2.
\]

如果各客户端独立且单客户端方差不超过 sigma_loc^2，则可以使用 sigma_0,t^2 不超过 sigma_loc^2/n 的粗界。

g_t 只是理论证明中的无偏参考 oracle，不要求服务器实际计算它。

### A3. DP 噪声作为有效随机 oracle 方差 [已证代数]

定义：

\[
g_t^{\mathrm{eff}}=g_t+z_t.
\]

则：

\[
\mathbb E[g_t^{\mathrm{eff}}\mid F_t]
=
\nabla f(x_t),
\]

\[
\mathbb E[
\|g_t^{\mathrm{eff}}-\nabla f(x_t)\|_2^2\mid F_t
]
\le
\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2.
\]

因此 DP 噪声首先进入 stochastic-oracle variance，而不进入压缩误差缓存。

### A4. 有界本地状态 [已实现机制；理论使用条件]

每轮每个客户端满足：

\[
\|h_{i,t}\|_2\le B_h,\qquad
\|e_{i,t}\|_2\le B_e,\qquad
\|r_{i,t}\|_2\le B_r.
\]

这是显式状态截断保证的条件。它不是 Paper 3 原始无界 EControl 理论的自动结论。

### A5. Dual Averaging 参数 [已证条件，沿用真实迭代分析]

主定理取 a_t 等于 1，gamma_t 非递减，并满足：

\[
\gamma_{t-1}>2L,\qquad
\gamma_{-1}=\gamma_0.
\]

变化的 a_t 需要重新缩放本地误差反馈；当前不直接把任意 a_t 插入 EControl 递推。

### A6. 三类近似误差 [代数定义]

定义：

\[
\rho_t=\frac1n\sum_i(u_{i,t}-v_{i,t}),
\]

\[
c_t=H_t-\frac1n\sum_i u_{i,t}.
\]

rho_t 包含残差缓存和 C_g 二次裁剪造成的输入改变；c_t 包含 Top-K/EControl 跟踪误差和状态截断误差。

因为：

\[
\bar v_t=g_t+\beta_t,
\]

所以：

\[
y_t-g_t^{\mathrm{eff}}
=
c_t+\rho_t+\beta_t.
\]

定义真实迭代分析中的累计近似误差：

\[
E_t
=
\sum_{s=0}^{t-1}
(c_s+\rho_s+\beta_s),
\qquad
E_0=0.
\]

再定义：

\[
\mathfrak C_t
=
\mathbb E\left\|
\sum_{s=0}^{t-1}c_s
\right\|_2^2,
\]

\[
\mathfrak R_t
=
\mathbb E\left\|
\sum_{s=0}^{t-1}\rho_s
\right\|_2^2,
\]

\[
\mathfrak B_t
=
\mathbb E\left\|
\sum_{s=0}^{t-1}\beta_s
\right\|_2^2.
\]

于是：

\[
\mathbb E\|E_t\|_2^2
\le
3(\mathfrak C_t+\mathfrak R_t+\mathfrak B_t).
\]

这个分解是已证代数事实。尚未证明的是如何由有界 EControl 递推得到对 \mathfrak C_t 的 sharp bound。

### A7. EControl 闭合条件 [候选，核心开放引理]

令：

\[
\delta=k/d.
\]

确定性 Top-K 满足：

\[
\|\operatorname{TopK}_k(s)-s\|_2^2
\le
(1-\delta)\|s\|_2^2.
\]

Paper 3 无投影 EControl 的典型参数为：

\[
\eta
=
\frac{\delta}
{3\sqrt{1-\delta}(1+\sqrt{1-\delta})}.
\]

当前投影机制额外产生：

\[
p^h_{i,t}
=
h_{i,t}-(h_{i,t-1}+q_{i,t}),
\]

\[
p^e_{i,t+1}
=
e_{i,t+1}-(\bar e_{i,t}+h_{i,t}-u_{i,t}),
\]

\[
p^r_{i,t+1}
=
r_{i,t+1}-(v_{i,t}+\bar r_{i,t}-u_{i,t}).
\]

需要证明存在 K_delta，使：

\[
\sum_{t=1}^{T}\mathbb E\|E_t\|_2^2
\le
K_\delta
\left[
\mathcal U_T+\mathcal P_T+\mathcal B_T
\right],
\]

其中：

\[
\mathcal U_T
=
\sum_{t=0}^{T-1}
\frac1n\sum_i
\mathbb E\|u_{i,t+1}-u_{i,t}\|_2^2,
\]

\[
\mathcal P_T
=
\sum_{t=0}^{T-1}
\frac1n\sum_i
\mathbb E[
\|p^h_{i,t}\|_2^2+
\|p^e_{i,t+1}\|_2^2+
\|p^r_{i,t+1}\|_2^2
],
\]

\[
\mathcal B_T
=
\sum_{t=1}^{T}
\mathbb E\left\|
\sum_{s=0}^{t-1}\beta_s
\right\|_2^2.
\]

这条引理为候选，不得当作 Paper 3 Lemma 4.1 已经自动适用于投影机制。无投影极限中，K_delta 应恢复 Paper 3 的约 O(delta^{-4}) 常数阶，但这仍需核对。

## 4. Candidate Theorem A 正式表述

### 4.1 目标函数界 [候选]

在 A1–A7 下，定义：

\[
\bar x_T^{\mathrm{real}}
=
\frac1T\sum_{t=0}^{T-1}x_{t+1},
\qquad
R_0=\|x^\star-x_0\|_2.
\]

候选结论为：

\[
\boxed{
\begin{aligned}
\mathbb E[
F(\bar x_T^{\mathrm{real}})-F(x^\star)
]
\le\;&
\frac{\gamma_{T-1}R_0^2}{2T}
\\
&+
\frac6T\sum_{t=1}^{T}
\frac{\mathfrak C_t+\mathfrak R_t+\mathfrak B_t}
{\gamma_{t-1}}
\\
&+
\frac2T\sum_{t=0}^{T-1}
\frac{\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2}
{\gamma_{t-1}}.
\end{aligned}
}
\]

按来源拆开：

\[
\mathcal E_{\mathrm{DA}}
=
\frac{\gamma_{T-1}R_0^2}{2T},
\]

\[
\mathcal E_{\mathrm{comp/state}}
=
\frac6T\sum_{t=1}^{T}\frac{\mathfrak C_t}{\gamma_{t-1}},
\]

\[
\mathcal E_{\mathrm{clip}}
=
\frac6T\sum_{t=1}^{T}
\frac{\mathfrak R_t+\mathfrak B_t}{\gamma_{t-1}},
\]

\[
\mathcal E_{\mathrm{stochastic+DP}}
=
\frac2T\sum_{t=0}^{T-1}
\frac{\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2}{\gamma_{t-1}}.
\]

### 4.2 带 movement 项的强形式 [候选]

更强的候选不等式是：

\[
\begin{aligned}
&\sum_{t=0}^{T-1}
\mathbb E\left[
F(x_{t+1})-F(x^\star)
+
\frac{\gamma_{t-1}-2L}{4}
\|x_{t+1}-x_t\|_2^2
\right]
\\
&\qquad\le
\frac{\gamma_{T-1}}2R_0^2
+
6\sum_{t=1}^{T}
\frac{\mathfrak C_t+\mathfrak R_t+\mathfrak B_t}{\gamma_{t-1}}
+
2\sum_{t=0}^{T-1}
\frac{\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2}{\gamma_{t-1}}.
\end{aligned}
\]

movement 项不能在证明早期丢掉，因为它正是控制 DP 噪声如何改变下一轮 EControl 输入的接口。

### 4.3 DP 项的 horizon 形式

若 sigma_DP,t 不超过 sigma_DP，则：

\[
\mathcal E_{\mathrm{DP}}
\le
\frac{2d\sigma_{\mathrm{DP}}^2}{T}
\sum_{t=0}^{T-1}\frac1{\gamma_{t-1}}.
\]

若 gamma_t 等于 c sqrt(T)，则：

\[
\mathcal E_{\mathrm{DP}}
\le
\frac{2d\sigma_{\mathrm{DP}}^2}{c\sqrt T}.
\]

若 gamma_t 等于 c sqrt(t+1)，则：

\[
\mathcal E_{\mathrm{DP}}
\le
\frac{4d\sigma_{\mathrm{DP}}^2}{c\sqrt T}.
\]

这里没有出现带噪 prefix 的 T^3 方差项。若服务器把噪声发布累加到下一轮估计，上述推导立即失效。

## 5. Paper 3 Appendix I 真实迭代不等式的使用

Paper 3 Appendix I Theorem I.1 的核心形式是：

\[
\begin{aligned}
&\sum_{t=0}^{T-1}
\mathbb E\left[
a_t(F(x_{t+1})-F(x))
+
\frac{\gamma_{t-1}-2a_tL}{4}
\|x_{t+1}-x_t\|_2^2
\right]
\\
&\qquad\le
\frac{\gamma_{T-1}}2\|x-x_0\|_2^2
+
2\sum_{t=1}^{T}\frac{\mathbb E\|E_t\|_2^2}{\gamma_{t-1}}
+
2\sum_{t=0}^{T-1}
\frac{a_t^2\sigma_g^2}{\gamma_{t-1}}.
\end{aligned}
\]

当前主线取 a_t 等于 1，x 等于 x*，并把 Paper 3 中的随机 oracle 替换为：

\[
g_t^{\mathrm{eff}}=g_t+z_t.
\]

于是：

\[
\sigma_g^2
\rightsquigarrow
\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2.
\]

而 E_t 只保留：

\[
E_t=\sum_{s<t}(c_s+\rho_s+\beta_s).
\]

这一步是当前理论设计的关键。不能把 z_s 同时放入 g_t 的方差项和 E_t 的累计误差项，否则会重复计入 DP 噪声。

## 6. 证明骨架

### Step 1：有效无偏 oracle

用 fresh z_t 定义 g_t^eff。验证条件均值和二阶矩，得到 sigma_0,t^2+d sigma_DP,t^2。

### Step 2：精确误差分解

由 y_t=H_t+z_t、H_t=n^{-1}sum_i u_i,t+c_t、bar v_t=g_t+beta_t、sum_i u_i,t/n=bar v_t+rho_t，得到：

\[
y_t-g_t^{\mathrm{eff}}=c_t+\rho_t+\beta_t.
\]

### Step 3：套用真实迭代不等式

直接应用 Paper 3 Appendix I Theorem I.1，保留 movement 项，得到目标差、movement、累计近似误差和有效 oracle 方差四部分。

### Step 4：三项误差平方分解

使用：

\[
\mathbb E\|E_t\|^2
\le
3(\mathfrak C_t+\mathfrak R_t+\mathfrak B_t).
\]

得到 Candidate Theorem A 的形式界。

### Step 5：有界 EControl 闭合

用 Top-K 收缩性质和 EControl 递推，证明 \mathfrak C_t 受到局部输入变化量以及投影残差控制。需要特别处理：

- h 状态投影；
- e 状态投影；
- r 状态投影；
- 使用 hbar、ebar、rbar 而非未投影状态；
- DP 噪声通过 x movement 改变下一轮输入。

无投影 EControl 的 Lemma 4.1 不能直接复制。

### Step 6：movement-coupling

用平均客户端 smoothness 证明：

\[
\frac1n\sum_i
\mathbb E\|u_{i,t+1}-u_{i,t}\|_2^2
\le
C_1\mathbb E\|x_{t+1}-x_t\|_2^2
+
C_2\mathbb E[\text{local stochastic variation}]
+
C_3\mathbb E[\text{projection residuals}].
\]

利用真实迭代不等式左侧的 movement 项吸收第一项；需要 gamma 足够大。

### Step 7：平均真实迭代

利用 F 的凸性：

\[
F\left(\frac1T\sum_t x_{t+1}\right)
\le
\frac1T\sum_tF(x_{t+1}).
\]

整个证明不使用虚拟迭代，也不使用最终全量上传。

## 7. 可先证明的受限版本

先假设总误差能量满足：

\[
\sum_{t=1}^{T}\mathfrak C_t\le\mathcal C_T,
\quad
\sum_{t=1}^{T}\mathfrak R_t\le\mathcal R_T,
\quad
\sum_{t=1}^{T}\mathfrak B_t\le\mathcal B_T.
\]

固定 gamma_t 等于 gamma 时：

\[
\mathbb E[
F(\bar x_T^{\mathrm{real}})-F(x^\star)
]
\le
\frac{\gamma R_0^2}{2T}
+
\frac{6(\mathcal C_T+\mathcal R_T+\mathcal B_T)}{\gamma T}
+
\frac2{\gamma T}\sum_{t=0}^{T-1}
(\sigma_{0,t}^2+d\sigma_{\mathrm{DP},t}^2).
\]

若这些总误差为 O(T)，再取 gamma=c sqrt(T)，即可得到形式上的 O(T^{-1/2}) 界。这是第一阶段最稳妥的条件式真实迭代结果。

如果有界 EControl 只能给出超线性误差能量，例如 O(T^{1+alpha})，则不能继续声称标准 O(T^{-1/2}) 结论，必须缩小定理或改变机制。

## 8. 隐私命题

### 8.1 单轮敏感度 [已证代数]

相邻数据集只改变一个客户端 j 时：

\[
\|H_t-H_t'\|_2
=
\frac1n\|h_{j,t}-h_{j,t}'\|_2
\le
\frac{2B_h}{n}.
\]

因此每轮 Gaussian 机制的敏感度是：

\[
S_h=\frac{2B_h}{n}.
\]

不需要 Top-K 连续性，也不需要证明原始无界 EControl 的敏感度。

### 8.2 自适应 RDP 组合 [已证条件]

第 t 轮 RDP：

\[
\rho_{\alpha,t}
=
\frac{\alpha S_h^2}
{2\sigma_{\mathrm{DP},t}^2}.
\]

全轨迹：

\[
\rho_\alpha^{(T)}
=
\sum_{t=0}^{T-1}
\frac{\alpha S_h^2}
{2\sigma_{\mathrm{DP},t}^2}.
\]

转换为 approximate DP：

\[
\varepsilon_{\mathrm{DP}}
\le
\min_{\alpha>1}
\left[
\rho_\alpha^{(T)}
+
\frac{\log(1/\delta_{\mathrm{DP}})}{\alpha-1}
\right].
\]

发布轨迹、真实迭代和最终后处理都继承该 DP 保证。隐藏 h、e、r 不需要单独发布。

## 9. 已证、候选与不主张

| 内容 | 状态 | 说明 |
|---|---|---|
| Paper 3 Appendix I 真实迭代基本不等式 | 已证/引用 | 使用真实迭代，不使用虚拟迭代 |
| Top-K 的 1-k/d 收缩 | 已证/代数 | 确定性 Top-K |
| 有界聚合敏感度 2B_h/n | 已证/代数 | 全参与客户端替换邻接 |
| 新鲜 Gaussian 的有效方差 | 已证/代数 | 零均值、独立、每轮重新采样 |
| Gaussian/RDP 自适应组合 | 已证/标准机制 | 真实迭代是后处理 |
| y_t-g_t^eff 的三项误差恒等式 | 已证/代数 | 由 beta、rho、c 定义得到 |
| 有界 EControl 的累计误差 O(T) | 候选 | 需要处理投影残差 |
| DP 噪声与 EControl movement 的闭合 | 候选 | 需要 smoothness 和 movement 吸收 |
| Candidate Theorem A 完整无条件版本 | 候选 | 依赖上述核心引理 |
| 平均真实迭代的条件式界 | 候选、优先证明 | 直接组合 Paper 3 Theorem I.1 |
| 最后迭代 x_T 的同阶界 | 不主张 | 需要额外假设 |
| 原始无界 EControl 的统一敏感度 | 不主张 | 对抗性搜索已发现风险 |
| 虚拟迭代率自动适用于真实迭代 | 不主张 | Paper 3 本身区分两者 |
| 最终无压缩上传的 DP 保证 | 不主张 | 主线删除该上传 |
| q<1 固定大小抽样主定理 | 不主张 | 留作后续分析 |

## 10. 下一步证明顺序

1. 在 a_t 等于 1、全参与、固定 gamma 下，把 Paper 3 Appendix I 公式逐项改写成当前 y_t 机制；
2. 用 g_t^eff=g_t+z_t 确认 DP 噪声只进入 variance term；
3. 写出 c_t、rho_t、beta_t 的逐行递推；
4. 在投影残差全为零的特例中复现 Paper 3 的 EControl 误差界；
5. 逐一加入 p_h、p_e、p_r，获得带投影残差的稳定性不等式；
6. 用 smoothness 和 movement 项处理 DP 噪声对下一轮 u 的影响；
7. 验证总累计误差是否为 O(T)；
8. 最后再考虑变化 gamma、非均匀阈值和 q<1。

最重要的判据是：

\[
\sum_{t=1}^{T}\mathbb E\|E_t\|_2^2=O(T)
\]

是否能在有界 EControl 加投影残差的机制下成立。如果不能，Paper 3 × DP 的第一版应停留在条件式真实迭代界，而不应声称完成无条件收敛定理。

