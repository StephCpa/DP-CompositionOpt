# Paper 3 × Differential Privacy：研究记录（v3 草稿）

**方向**：`Composite Optimization with Error Feedback: the Dual Averaging Approach` 与客户端级中心差分隐私（central client-level DP）的结合。

**当前判断**：主线应改为“有界本地状态 + 当前聚合估计的逐轮新鲜高斯噪声 + 真迭代输出”。原始 EControl 的无界状态不宜直接作为敏感度证明对象；未经截断的 `h` 发布在对抗性搜索中已经出现明显超过 2 的替换敏感度。固定大小客户端抽样可以保留为次级实验，但不能默认其带来强隐私放大：在“活跃客户端平均值”的发布机制中，敏感度中的 `1/m` 会抵消相当一部分抽样收益。

本文档区分三类内容：

- **已实现/已观测**：代码和数值实验已经支持的事实；
- **候选定理或引理**：可以作为论文理论目标，但尚未证明；
- **停止条件**：如果某个条件失败，应缩小问题、切换机制或停止该实验线。

---

## 1. 研究问题与贡献目标

### 1.1 研究问题

考虑带复合正则项的凸随机优化问题

\[
\min_x F(x) = f(x)+\lambda\psi(x),
\]

其中 \(f\) 可以由多个客户端的数据平均得到，\(\psi\) 为可近端求解的复合项（实验中使用 \(\ell_1\) 正则）。每个客户端进行带误差反馈的 Top-K 压缩，服务器使用 Dual Averaging（DA）更新模型。目标是构造一个同时满足以下条件的机制：

1. 每轮消息对客户端替换邻接满足明确的有限敏感度；
2. 隐藏的误差反馈状态和释放的真实迭代都能纳入同一个隐私证明；
3. 能够给出适用于真实迭代的优化误差界，不依赖 Paper 3 尚未闭合的虚拟迭代分析；
4. 在固定总通信比特数下，与中心 DP 的无压缩 DA/FTRL 基线进行公平比较。

### 1.2 目标贡献的最小版本

最小可发表结果不应同时承诺“原始 EControl 的无界状态敏感度定理”“压缩与 DP 的联合最优界”“虚拟/真实迭代等价”。建议将主张收缩为：

> 在客户端状态显式截断、每轮释放当前有界聚合估计并加入新鲜高斯噪声的机制下，给出一个简单的客户端替换敏感度界；在 Paper 3 的真实迭代框架下，把压缩误差、截断误差和 DP 噪声分别纳入复合优化误差界；实验上研究状态截断、参与率和固定比特预算之间的权衡。

这会把隐私正确性和优化效用分开：隐私由确定性的状态半径给出，效用由有界误差反馈、噪声方差和截断残差分析给出。

---

## 2. 威胁模型和邻接关系

### 2.1 邻接关系

采用**客户端级替换邻接**：两个数据集只在一个客户端的全部本地数据上不同。每个客户端每轮先构造一个被裁剪的本地平均梯度，因此替换一个客户端最多改变两个半径受限向量的差：

\[
\|u_i-u_i'\|_2\le 2C_g,
\]

而对有界本地状态的直接聚合，替换敏感度的基本因子是 2，而不是 1。

### 2.2 主线威胁模型

- 服务器为可信 curator，但会看到每轮公开的模型/聚合消息；
- 客户端本地状态不公开；
- 每轮选择固定大小的活跃客户端集合 \(S_t\)，大小为 \(m\)；
- 输出为真实 DA 迭代 \(x_t\) 或由这些发布结果计算出的后处理量；
- DP 预算为 \((\varepsilon_{DP},\delta_{DP})\)，与压缩误差符号、DA 正则系数等记号严格区分。

如果采用全参与，令 \(m=n\)。如果采用固定大小抽样，状态未被选中时保持不变；本记录不把“所有持久化状态的全体平均”作为隐私发布对象，因为不活跃客户端的旧状态会持续携带其历史影响，敏感度需要单独分析。

### 2.3 中心 DP 与本地 DP 的分组

主表只比较中心 DP 机制。SoteriaFL 或其他本地 DP 方法放在单独的表格中；在相同 \(\varepsilon\) 下直接混合中心 DP 和本地 DP 会掩盖威胁模型差异，也会使中心噪声的客户端规模优势变成一个无信息的“基线胜出”。

---

## 3. 正式机制：有界 EControl + DA + 当前估计新鲜噪声

下述机制是当前主线。记客户端为 \(i\)，轮次为 \(t\)，模型为 \(x_t\)。

### 3.1 本地梯度和裁剪残差

客户端先计算每样本梯度并做范数裁剪：

\[
v_{i,t}=\frac{1}{b_i}\sum_{j=1}^{b_i}
\operatorname{clip}_{C_0}(\nabla\ell(x_t;z_{ij})).
\]

为了保留被 \(C_g\) 二次裁剪掉的部分，引入裁剪残差缓存：

\[
\bar r_{i,t}=\operatorname{clip}_{B_r}(r_{i,t}),
\qquad
u_{i,t}=\operatorname{clip}_{C_g}(v_{i,t}+\bar r_{i,t}),
\]

\[
r_{i,t+1}=\operatorname{clip}_{B_r}(v_{i,t}+\bar r_{i,t}-u_{i,t}).
\]

这里 \(r_{i,t}\) 只用于优化误差反馈。隐私证明可以直接依赖后续状态截断，不需要先证明 \(r\) 的无界稳定性；但 \(B_r\) 影响效用，因此应作为实验超参数报告。

### 3.2 有界 EControl 状态

令 \(h_{i,t}\) 为客户端的累计压缩估计，\(e_{i,t}\) 为 EControl 误差缓存。每轮先截断旧状态：

\[
\bar h_{i,t}=\operatorname{clip}_{B_h}(h_{i,t-1}),
\qquad
\bar e_{i,t}=\operatorname{clip}_{B_e}(e_{i,t}).
\]

构造待压缩向量：

\[
\delta_{i,t}=u_{i,t}-\bar h_{i,t}-\eta\bar e_{i,t}.
\]

为避免状态中的任意中间量产生无界敏感度，先将 \(\delta_{i,t}\) 裁剪到

\[
C_\Delta=C_g+B_h+\eta B_e,
\]

然后进行 Top-K：

\[
q_{i,t}=\operatorname{TopK}_k\left(\operatorname{clip}_{C_\Delta}(\delta_{i,t})\right).
\]

状态更新为

\[
h_{i,t}=\operatorname{clip}_{B_h}(h_{i,t-1}+q_{i,t}),
\]

\[
e_{i,t+1}=\operatorname{clip}_{B_e}(\bar e_{i,t}+h_{i,t}-u_{i,t}).
\]

Top-K 的坐标选择可能不连续，因此不应把 Top-K 当成全局 Lipschitz 映射来证明敏感度。当前机制的隐私证明绕开这一点：先用状态截断给出输出空间的直径界。

### 3.3 聚合与新鲜噪声

活跃客户端集合为 \(S_t\)，\(|S_t|=m\)。服务器在本轮只聚合活跃客户端的当前状态：

\[
h_t^{\mathrm{act}}=\frac{1}{m}\sum_{i\in S_t}h_{i,t}.
\]

服务器内部保留干净的 \(h_t^{\mathrm{act}}\)，公开的是当前聚合估计加上**每轮独立的新鲜噪声**：

\[
y_t=h_t^{\mathrm{act}}+z_t,
\qquad
z_t\sim\mathcal N(0,\sigma_t^2I_d).
\]

关键实现约束：不能先释放 \(\Delta_t+z_t\)，再在服务器上累加为下一轮的估计。那样会使噪声随机游走，并被 DA 再次累加，噪声方差从线性累积变为近似三次方累积。当前代码始终由干净状态产生下一轮的释放值，DA 直接使用当前 \(y_t\)。

### 3.4 Dual Averaging 更新

令 \(a_t\) 为权重，

\[
A_t=\sum_{s=0}^t a_s,
\qquad
G_t=G_{t-1}+a_t y_t.
\]

更新为

\[
x_{t+1}=\arg\min_x\left\{
\langle G_t,x\rangle+A_t\lambda\psi(x)
+\frac{\gamma_t}{2}\|x-x_0\|_2^2
\right\}.
\]

实验中使用 \(a_t=1\) 和 \(\ell_1\) 正则，对应逐轮增加正则权重。当前实现通过软阈值近端算子求解该子问题。

不再进行 Paper 3 原始算法中的最终无压缩累计梯度上传；最终输出直接使用真实迭代。这样可以避免在 DP 证明结尾新增一个高敏感度的“隐藏全量上传”。

---

## 4. 隐私证明目标和候选引理

### 4.1 可以立即得到的状态截断敏感度界

**候选引理 1（有界活跃状态聚合敏感度）**。若每个客户端状态在发布前满足

\[
\|h_{i,t}\|_2\le B_h,
\]

则在客户端级替换邻接下，条件于同一个活跃集合 \(S_t\) 且改变的客户端属于 \(S_t\)，有

\[
\left\|h_t^{\mathrm{act}}-(h_t^{\mathrm{act}})'\right\|_2
\le \frac{2B_h}{m}.
\]

若改变的客户端不在 \(S_t\)，本轮直接敏感度为 0，但其数据可能通过未来参与轮次影响后续状态。若选择集合与数据独立，整个轨迹可按逐轮自适应组合处理；需要明确状态的持久化和客户端再次参与时的条件敏感度。

全参与时取 \(m=n\)，得到 \(2B_h/n\)。这条界与 Top-K 的连续性无关，是当前机制采用状态截断的主要原因。

前缀 FTRL 参考的敏感度不同：代码中的 `prefix` 是 curator 侧的单个持久状态，而不是客户端局部状态的逐轮平均。即使每轮输入增量来自活跃平均值，把累计 `prefix` 直接截断到半径 \(B_s\) 只能得到发布球的直径界 \(2B_s\)，不能额外乘上 \(1/m\)。因此不能用 EControl active-average 的 \(2B_h/m\) 公式给前缀参考记账。

由于 prefix 的差异会在客户端下一次参与后持续存在，即使该客户端在后续轮次不活跃，前缀轨迹仍可能不同；因此前缀参考还不能逐轮套用客户端抽样放大。代码对 prefix 使用 accountant_q=1，即无放大的保守账本。对应的修正扫描保存在 prefix_corrected_sensitivity_scan.json，旧的带 q 放大数值不再用于比较结论。

**候选引理 1 的限制**：它只控制每轮的发布均值。若希望给出比逐轮高斯机制更紧的矩阵机制/相关噪声界，还需分析累计工作负载和参与模式；不能从该引理自动推出相关噪声优于独立噪声。

### 4.2 固定大小抽样的 RDP 账本

对每轮均匀无放回选取 \(m\) 个客户端，可以采用固定大小无放回抽样的 RDP 上界；实现中提供三种模式：

1. `without_replacement_bound`：固定大小无放回抽样的保守上界，作为默认正式账本；
2. `poisson_surrogate`：便于和常见 Poisson subsampling 结果比较的近似账本；
3. `conservative`：关闭放大，只做无抽样组合。

实验必须同时报告抽样率 \(q=m/n\)、敏感度 \(2B_h/m\)、噪声标准差和最终核算的 \(\varepsilon\)。仅报告“抽样率降低”是不够的，因为活跃平均值的敏感度也随 \(m\) 下降，二者会相互抵消一部分放大收益。

### 4.3 当前不应声称的无界 EControl 引理

原始、未截断的 EControl 状态不应直接声称存在与时间无关的 \(\|h_t-h_t'\|\) 或 \(\|e_t-e_t'\|\) 上界。对抗性搜索给出的结果如下：

| 维度/轮数 | K | \(\eta\) | 最大 \(\|h-h'\|\) | 最大 \(\|\Delta-\Delta'\|\) |
|---|---:|---:|---:|---:|
| \(d=8,T=80\) | 1 | \(1/8\) | 5.66 | 6.26 |
| \(d=16,T=150\) | 1 | \(1/16\) | 6.06 | 5.41 |
| \(d=16,T=150\) | 1 | \(1/(16\cdot400)\) | 4.48 | 4.00 |

这些结果是有限维度、有限轮数和启发式坐标/状态搜索，不能作为反例定理；但它们足以否定“可以直接假定消息敏感度约为 2”的研究路线。若对抗性搜索在更大范围继续显著增长，应优先使用发布裁剪或状态截断，而不是继续寻找无界 Top-K 的统一常数。

### 4.4 关于误差缓存的候选结论

随机流实验中，\(\|e_t-e_t'\|\) 在小 \(\eta\) 下仍随轮数增长；例如 \(T=40000\)、\(K/d=0.01\)、\(\eta=\delta/400\) 时达到约 5673，但 \(\eta\|e_t-e_t'\|\) 仍约为 1.67。因而候选理论对象应是反馈项 \(\eta e_t\) 或截断后的 \(\bar e_t\)，而不是直接声称 \(e_t\) 有时间无关的界。

这条观察不能替代证明；有界机制中仍应明确写出 \(B_e\) 对隐私与效用的作用。

---

## 5. 优化误差分解：当前拟证明的形式

### 5.1 噪声不要重复计入压缩误差

释放噪声 \(z_t\) 是每轮新鲜、零均值且与当前真实迭代条件独立的随机量。它应作为 DA 随机梯度/随机 oracle 的方差项进入，而不是被当成压缩误差缓存的一部分再累加一次。

因此，误差分解至少应包括：

- 随机梯度方差和样本裁剪偏差；
- EControl/Top-K 的压缩误差；
- 状态截断残差（\(h\)、\(e\)、\(r\) 的投影残差）；
- DP 高斯噪声方差；
- 客户端抽样造成的估计方差；
- 复合近端项 \(\lambda\psi\) 的 DA 近端误差。

### 5.2 真实迭代版本的候选定理

**候选定理 A（真实迭代的中心 DP 有界 EControl-DA 界）**。在 Paper 3 所需的凸性、可近端求解、梯度二阶矩和步长条件下，假设每轮发布

\[
y_t=h_t^{\mathrm{act}}+z_t,
\quad z_t\sim\mathcal N(0,\sigma_t^2I),
\]

且有界 EControl 的压缩误差和三个截断残差满足可加的二阶矩上界，则真实迭代输出满足如下结构的界：

\[
\mathbb E[F(\bar x_T)-F(x^*)]
\le
\underbrace{\mathcal R_{\mathrm{DA}}(T)}_{\text{Paper 3 真实迭代基线}}
+\underbrace{\mathcal E_{\mathrm{comp}}(T)}_{\text{Top-K/EControl}}
+\underbrace{\mathcal E_{\mathrm{clip}}(T)}_{\text{梯度与状态裁剪}}
+\underbrace{\mathcal E_{\mathrm{client}}(T)}_{\text{客户端抽样}}
+\underbrace{\mathcal E_{\mathrm{DP}}(T)}_{\text{新鲜高斯噪声}}.
\]

其中 \(\mathcal R_{\mathrm{DA}}(T)\) 必须使用 Paper 3 已经给出的**真实迭代**率；不应把虚拟迭代率直接冒充真实迭代率。

当 \(\gamma_t=c\sqrt{t}\) 或类似步长时，DP 噪声的直接方差项应呈现类似

\[
\mathcal E_{\mathrm{DP}}(T)
\asymp
\frac{L}{A_T}\sum_{t=1}^T w_t\,d\sigma_t^2,
\]

具体权重取决于 DA 的不等式和 \(a_t,\gamma_t\) 的定义。不要把它写成对噪声前缀的二次累加，除非机制真的在服务器上累加了带噪前缀。

### 5.3 DP 噪声与 EControl 的耦合项

即便噪声独立地进入 DA，噪声仍会改变 \(x_t\)，从而改变下一轮的梯度和本地 EControl 输入。真正需要分析的耦合项来自

\[
\|x_{t+1}-x_t\|,
\]

例如 EControl 跟踪误差中的 \(\ell\|x_{t+1}-x_t\|\) 项。建议先在强凸/光滑的合成问题上证明一个简化界，再推广到一般复合凸目标；不应在第一版中同时处理非光滑损失、任意客户端漂移和无界状态。

### 5.4 裁剪残差的隐私和效用分工

\(u_t=\operatorname{clip}_{C_g}(v_t+r_t)\) 本身有界，因此隐私证明只需依赖输入半径和状态截断。\(r_t\) 的半径 \(B_r\) 主要服务于效用分析：它控制因二次裁剪而丢失的梯度部分。无需另立一个“\(r_t\) 的隐私敏感度引理”才能完成候选定理 A，但应在实验中报告 \(r\) 的范数和截断残差。

---

## 6. 实验实现状态

### 6.1 代码和功能

当前模拟器文件：

- [`p3_bounded_sim.py`](../code/p3_bounded_sim.py)：合成二分类逻辑回归、\(\ell_1\) 复合项、有界 EControl + Top-K、中心 DP DA、有限前缀 FTRL 参考、固定大小客户端采样、状态年龄和 RDP 账本；
- [`p3_adversarial.py`](../code/p3_adversarial.py)：针对原始未截断 EControl 的有限维对抗性敏感度搜索。

模拟器的输出至少包含：目标函数、测试准确率、\(\sigma\)、单轮敏感度、核算后的 \(\varepsilon\)、总比特数、客户端平均比特数、状态年龄、消息范数、梯度裁剪比例、状态/前缀截断残差。

### 6.2 比特计数

Top-K 消息不能只计 K 个浮点数，还必须计 K 个坐标索引。当前实现按

\[
B_{\mathrm{round}}
=|S_t|K\left(b_{\mathrm{value}}+\lceil\log_2 d\rceil\right)
\]

计比特（实验默认浮点值为 32 bit），并报告总比特和每客户端比特。

无压缩中心 DA 的每轮消息按 \(d\) 个浮点数计数。比较时默认采用**固定总比特数**，而不是固定轮数；同时应单独给出固定轮数曲线，以便区分“压缩省通信”和“相同训练步数下的算法损失”。

### 6.3 数据集

- 合成逻辑回归：用于快速扫参数、调试隐私账本和研究敏感度；
- sklearn digits（64 维标准化像素，0–4 对 5–9）：作为无网络下载的图像特征冒烟测试；
- FashionMNIST 或其他更大图像任务：只有在合成和 digits 上确认机制/账本无明显问题后再加入。

Digits 当前一组 100 轮冒烟结果（默认参数附近）显示：中心 DP 的有界 EControl-TopK 与无压缩 DA 的目标值和准确率接近，有限前缀 FTRL 参考的目标值更低但准确率不同；这只是管线检查，不能作为算法结论。

---

## 7. 实验矩阵

### 7.1 第一阶段：机制正确性和隐私账本

| 轴 | 取值 | 目的 |
|---|---|---|
| 状态半径 \(B_h\) | 0.25, 0.5, 1, 2, 5 | 测试敏感度-截断-效用权衡 |
| 误差半径 \(B_e\) | 1, 2, 5 | 测试 \(\eta e\) 和截断残差 |
| Top-K 比例 | 1, 0.1, 0.01 | 对照无压缩、稀疏压缩 |
| 参与率 \(q\) | 1, 0.5, 0.2 | 研究固定大小抽样和状态年龄 |
| 账本 | 无放回上界、Poisson 近似、无放大保守账本 | 量化账本不确定性 |
| \(\varepsilon_{DP}\) | 2, 4, 8 | 隐私强度曲线 |
| 轮数 | 50, 100, 200, 500 | 查找隐私限制的最优训练轮数 |

每个配置至少 3 个随机种子；对抗性敏感度实验单独报告搜索初始化、搜索步数和是否找到更坏方向。

### 7.2 第二阶段：固定比特曲线

在每客户端比特预算相同的条件下，为每个参与率选择对应轮数：

\[
T(q)\cdot qn\cdot K\left(32+\lceil\log_2d\rceil\right)
= B_{\mathrm{client}}.
\]

比较：

1. 非私有有界 EControl-TopK；
2. 中心 DP 有界 EControl-TopK；
3. 中心 DP 无压缩 DA；
4. 有界前缀 FTRL 参考；
5. 本地 DP 基线（单独表格）。

核心图应画出目标值/准确率随总比特数变化的曲线，而不是只报一个“相同轮数”的点。

### 7.3 第三阶段：数据和几何扩展

先在合成和 digits 上验证：

- 抽样率降低时，状态年龄是否导致效用下降；
- \(B_h\) 降低是否明显增加截断残差；
- DP 噪声是否由当前估计新鲜释放正确进入，而非形成随机游走；
- 真实迭代是否稳定。

只有这四项通过，才扩展到 FashionMNIST、更多客户端异质性和非均匀参与。

一个合成数据的单配置运行示例为：

```bash
python p3_bounded_sim.py \
  --dataset synthetic --rounds 100 --n-clients 100 \
  --samples-per-client 40 --d 20 --topk-frac 0.1 \
  --Bh 1 --Be 2 --Cg 1 --C0 1 \
  --epsilon 8 --participation-rate 0.5 \
  --accountant without_replacement_bound --seed 0
```

对抗性敏感度诊断使用：

```bash
python p3_adversarial.py
```

---

## 8. 当前数值证据

### 8.1 对抗性搜索：原始无界机制风险明确

在有限维度上，坐标方向、状态方向、随机单位向量和坐标上升搜索已经找到：

- \(d=8,T=80,K=1,\eta=1/8\)：\(\|h-h'\|\) 达 5.66；
- \(d=16,T=150,K=1,\eta=1/16\)：\(\|h-h'\|\) 达 6.06；
- 缩小 \(\eta\) 到 \(1/(16\cdot400)\) 后仍达 4.48。

因此“原始 EControl 的消息敏感度约为 2”不能作为主线假设。该搜索尚非严格反例，但已经足够触发状态截断方案。

### 8.2 随机长时域：\(\eta e\) 比 \(e\) 更稳定

在 \(d=1000,T\le 40000,K/d=0.01\) 的随机单位向量邻接流中，\(\eta=\delta/400\) 时，\(\|e-e'\|\) 可达到约 5673，而 \(\eta\|e-e'\|\) 仍约 1.67；\(\hat g\) 的运行最大值约 1.58。\(\eta=\delta\) 时，\(\hat g\) 的运行最大值约 2.48。

这说明随机流的“看起来有界”不能决定理论；应先用对抗性搜索，再决定是否尝试无截断引理。

### 8.3 形式固定大小账本下的参与率观察（单种子诊断）

在 \(n=100,\text{samples/client}=40,d=20,T=100,K/d=0.1,\varepsilon=8\) 的扫描中，正式无放回上界给出的噪声随参与率下降而上升：

| 参与率 | \(B_h=0.25\) 的目标值 | 准确率 | \(\sigma\) | 单轮敏感度 |
|---:|---:|---:|---:|---:|
| 1.0 | 约 0.294 | 约 0.921 | 0.0347 | 0.005 |
| 0.5 | 约 0.297 | 约 0.912 | 0.0693 | 0.010 |
| 0.2 | 约 0.312 | 约 0.898 | 0.1068 | 0.025 |

在更大的 \(B_h\) 下效用恶化更明显。例如全参与、\(B_h=1\) 时目标值约 0.345；\(B_h=5\) 时约 1.73。半参与时相应约 0.439 和 3.46。这里是单种子诊断，且应在最终日志中补充 `samples_per_client`；数值依赖数据种子和其他超参数，但方向与机制分析一致：状态半径控制噪声规模，也控制压缩信息是否被截断。

### 8.4 修正后的正式无放回 RDP 扫描

以下结果使用 `p3_bounded_sim.py` 当前版本的 `without_replacement_bound`。这是 Wang--Balle--Kasiviswanathan 固定大小无放回抽样定理的 Gaussian 上界实现，不应表述为精确 accountant。参数为 \(n=100,\text{samples/client}=40,d=20,K/d=0.1,\varepsilon=8,\delta=10^{-5},C_0=C_g=1,B_r=2,B_e=2\)，并以 \(B_h=1\) 作为 equal-bits 主比较的状态半径。

在 \(q=0.2,T=100\) 下，\(B_h\) 扫描给出：

| \(B_h\) | 目标函数 | 准确率 | \(\sigma\) | 敏感度 | 每客户端比特 | 平均状态年龄 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7364 | 0.7015 | 0.4272 | 0.10 | 1,480 | 4.06 |
| 2 | 1.9150 | 0.5768 | 0.8544 | 0.20 | 1,480 | 4.06 |
| 5 | 6.3543 | 0.5023 | 2.1360 | 0.50 | 1,480 | 4.06 |

此扫描中 \(h\)-state clipping 没有触发；增大 \(B_h\) 主要直接放大 DP 噪声，而不是恢复被状态裁剪损失的效用。

在 equal-bits 设定 \(K=2,qT=100\)，每客户端通信量固定为 7,400 bits，并使用三个随机种子：

| \(q\) | \(T\) | 目标函数均值 ± SD | 准确率均值 ± SD | \(\sigma\) | 平均状态年龄 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 100 | 0.3395 ± 0.0127 | 0.8714 ± 0.0027 | 0.1386 | 0.00 |
| 0.5 | 200 | 0.6261 ± 0.1098 | 0.7567 ± 0.0180 | 0.3921 | 0.92 |
| 0.2 | 500 | 5.9521 ± 0.8513 | 0.6523 ± 0.0470 | 1.5500 | 4.12 |

因此正式上界账本下的 headline 结论与 Poisson surrogate 相反：在 active-average 发布中，敏感度 \(2B_h/(qn)\) 的 \(1/q\) 因子会抵消大部分抽样放大；在 equal-bits 延长轮数后，噪声和状态陈旧性占主导。\(q=1\) 应作为第一阶段理论与主实验设置，\(q<1\) 作为 stress test，而不是默认的效用改进方案。

这个结论依赖于当前的保守 formal upper bound；论文中必须同时报告 accountant 类型、\(q\)、敏感度和 \(\sigma\)，不能把 Poisson surrogate 的数字当作 headline 结果。

### 8.6 第二类凸目标：synthetic softmax+\(\ell_1\)

为检查主线是否只对二分类 logistic 有效，新增了独立的多分类 softmax 模拟器。目标函数为

\[
F(W)=\frac1N\sum_{j=1}^N
\left[
\log\sum_{c=1}^C\exp(x_j^\top w_c)
-x_j^\top w_{y_j}
\right]
+\lambda\|W\|_1.
\]

把 \(W\in\mathbb R^{d\times C}\) 展平为 Dual Averaging 变量，客户端使用 bounded EControl + Top-K，服务器发布当前 clean aggregate 加 fresh Gaussian noise，仍然使用真实迭代。参数为 \(n=20,d=8,C=3,T=80,K/D=0.1,B_h=1,\varepsilon=8,\delta=10^{-5}\)，三个随机种子。

| 方法 | 目标函数均值 ± SD | 准确率均值 ± SD | \(\sigma\) | 敏感度 | 每客户端比特 |
|---|---:|---:|---:|---:|---:|
| 非私有 EControl-TopK | 0.6284 ± 0.0187 | 0.7969 ± 0.0130 | 0 | 0.100 | 5,920 |
| 中心 DP EControl-TopK | 1.0790 ± 0.1374 | 0.6230 ± 0.0106 | 0.6200 | 0.100 | 5,920 |
| 中心 DP dense DA | 1.6615 ± 0.1774 | 0.5517 ± 0.0114 | 0.9300 | 0.150 | 71,040 |

这只是小规模 synthetic smoke evidence，不足以形成最终性能结论；它的作用是确认机制、账本和真实迭代接口可以迁移到第二个凸复合目标。

为区分 DP 噪声和 Top-K 压缩的影响，又增加了 dense EControl（\(K=D\)）作为仅保留 EControl、去掉 Top-K 稀疏性的对照：

| 方法 | 目标函数均值 ± SD | 准确率均值 ± SD | \(\sigma\) | 每客户端比特 |
|---|---:|---:|---:|---:|
| 中心 DP EControl-TopK | 1.0790 ± 0.1374 | 0.6230 ± 0.0106 | 0.6200 | 5,920 |
| 中心 DP dense EControl | 1.0779 ± 0.1366 | 0.6234 ± 0.0099 | 0.6200 | 71,040 |
| 中心 DP dense DA | 1.6615 ± 0.1774 | 0.5517 ± 0.0114 | 0.9300 | 71,040 |

在这个小规模设置下，Top-K 和 dense EControl 的 DP 效用几乎相同，而 Top-K 使用约 \(12\times\) 更少通信。这支持“Top-K 是通信效率分支”的表述，但不支持“压缩本身带来准确率提升”的表述。

softmax 梯度有限差分检查的最大绝对误差约为 \(10^{-7}\)。

若每个特征满足 \(\|x\|_2\le R\)，则单样本梯度满足 \(\|\nabla_W\ell\|_F\le\sqrt2R\)，softmax Hessian 的谱范数给出 \(L\le R^2/2\) 的光滑性上界；\(\ell_1\) 项仍由坐标 soft-threshold prox 处理。这为真实迭代候选定理在第二个目标上的验证提供了明确的 \(L\) 和梯度半径参数。

### 8.5 账本选择会改变结论

早期使用 Poisson 近似账本时，固定每客户端通信预算的扫描曾显示 \(q=0.5\) 可能优于 \(q=1\)。切换到正式固定大小无放回上界后，抽样放大明显弱化，必须重新跑等比特多种子实验，不能沿用旧结论。论文中应把 Poisson 结果标为敏感性分析，而不是主结果。

---

## 9. 停止条件和决策规则

### 停止条件 1：敏感度可证明或切换发布裁剪

在有限维、有限轮数的对抗性搜索中，若未截断的 \(h\) 或 \(\Delta\) 继续显著超过预设目标（例如约 2–3），则停止寻找原始 EControl 的时间无关敏感度常数，采用有界状态或直接发布裁剪。只有找到对任意自适应有界输入流成立的统一界，才把未截断机制升级为理论主线。

### 停止条件 2：固定比特曲线必须存在非平凡优势

在正式无放回账本、至少 3 个随机种子和相同中心 DP 威胁模型下，若压缩方法在整个固定比特预算范围都被无压缩 DA/FTRL 严格支配，则不再扩展大模型；先检查预算是否覆盖隐私受限最优轮数附近。如果只在极窄预算区间胜出，应报告完整曲线，不把单点胜出写成普遍结论。

### 停止条件 3：截断残差不能吞掉压缩收益

若选择的 \(B_h,B_e,B_r\) 使平均截断残差与原始消息范数同量级，或使有界机制明显退化为常量消息，则停止扩大该参数区间，转而研究更温和的发布裁剪/自适应半径。需要同时报告状态半径、截断残差和最终效用，不能只调到隐私噪声最小的参数。

### 停止条件 4：真实迭代稳定性

若在 DP 噪声下真实迭代出现发散、目标值随轮数单调恶化，或对噪声种子极端敏感，则不声称完成 Paper 3 的真实迭代扩展；先限制到强凸/光滑合成问题，或调整 \(\gamma_t\) 和噪声调度。不能用虚拟迭代实验替代真实迭代的稳定性证据。

### 停止条件 5：抽样状态年龄失控

若 \(q<1\) 时平均或 P90 状态年龄持续超过训练轮数的固定比例，并且效用下降无法由增加总比特数补偿，则把固定大小抽样降为附加实验，主线回到全参与。该条件直接反映“误差反馈需要频繁参与，而隐私放大需要抽样”的结构张力。

---

## 10. 下一步执行顺序

1. 已完成正式 `without_replacement_bound` 的 Bh 扫描、prefix Bs 扫描和 equal-bits 三种子实验；JSON/CSV 保存在 `formal_scan_results.json` 与 `formal_scan_results.csv`；
2. 主线理论和第一阶段主表固定为全参与 q=1。q=0.5,0.2 保留为 participation/state-age stress test，并同时标出 formal upper bound 与 Poisson surrogate 的差异；
3. 把账本模式、敏感度、\(\sigma\)、状态年龄和截断残差写入 CSV/JSON，避免只保留终端摘要；
4. 检查固定大小账本对活跃平均值机制的敏感度约定，明确“客户端抽样”和“每客户端内部小批采样”不混为同一个放大来源；
5. softmax+ℓ1 已完成小规模 smoke test；full-participation 真实迭代候选定理 A 的正式假设和证明骨架已整理到 `p3_real_iterate_theorem_skeleton_zh.md`；
6. softmax 的无噪声 / DP+Top-K / dense EControl / dense DA ablation 已完成；下一步才进入 FashionMNIST 和更多异质性实验。

---

## 11. 当前结论

当前最稳妥的研究叙事是：Paper 3 的误差反馈结构可以和中心 DP 结合，但需要把隐私证明对象从原始无界压缩轨迹改成**显式有界的本地状态和当前聚合估计**。服务器发布当前干净估计加新鲜噪声；DA 使用该发布值，不再把带噪差分二次累加。固定大小客户端抽样不能自动带来强放大，因为活跃平均值的敏感度也随参与数变化；它更适合作为通信-参与率-状态年龄的实验变量。

论文第一版应把贡献聚焦于：

- 有界 EControl 状态的简单中心 DP 敏感度界；
- 真实迭代下的压缩、截断和 DP 方差误差分解；
- 固定比特预算下的参与率与状态年龄实证。

若对抗性搜索或正式账本显示效用收益不存在，就停止扩展“DP + Top-K + EControl”的复杂组合，保留该记录作为机制诊断，并转向更适合 DP 的复合 DA/FTRL 变体。

### 8.7 第三个凸目标：盒约束最小二乘

为检查 Candidate Theorem A 是否依赖分类损失的特殊结构，新增盒约束最小二乘目标。每个客户端持有线性回归样本，服务器优化带盒约束的平方损失；本实验仍然只使用全参与 \(q=1\)、bounded EControl、Top-K、当前 clean aggregate 加 fresh Gaussian noise，并且输出真实迭代。参数为 \(n=20,m=40,d=10,T=100,K/d=0.1,\varepsilon=8,\delta=10^{-5}\)，三个随机种子，\(B_h=1,B_r=B_e=2,B_{\mathrm{box}}=2,\gamma=5\)。有限差分梯度检查的最大误差为 \(1.27\times10^{-10}\)。

| 方法 | 目标函数均值 ± SD | 测试 MSE 均值 ± SD | 参数 MSE | \(\sigma\) | 敏感度 | 每客户端比特 |
|---|---:|---:|---:|---:|---:|---:|
| 非私有 EControl-TopK | 0.01143 ± 0.00018 | 0.02286 ± 0.00036 | \(2.3\times10^{-5}\) | 0 | 0.100 | 3,600 |
| 中心 DP EControl-TopK | 0.74989 ± 0.44360 | 1.49978 ± 0.88719 | 0.14757 | 0.69318 | 0.100 | 3,600 |
| 中心 DP dense DA | 3.37148 ± 2.18619 | 6.74297 ± 4.37239 | 0.66768 | 1.03977 | 0.150 | 36,000 |

该实验的用途是诊断误差项，而不是宣称盒约束问题已经有无条件收敛定理。三个方法的状态年龄均为 0，\(h\) 的 prefix 截断残差为 0，说明在此参数点有界状态没有成为效用瓶颈。DP-TopK 的平均移动量为 0.4385，最大累计 \(E_t\) 范数为 47.58，\(\sum_t\|E_t\|^2\) 的三种子均值为 \(1.21\times10^5\)；dense DP-DA 的对应值为 0.6466、118.34 和 \(5.89\times10^5\)。这表明当前噪声尺度下，Candidate Theorem A 最需要控制的是 DP 噪声引起的 movement 与累计误差耦合，而不是状态陈旧性或盒投影残差。

和 softmax+\(\ell_1\) 的结果合在一起，现阶段可以把“机制迁移到多个凸复合目标”写成可复现实验事实；理论表述仍应保持条件式：需要进一步证明或假设 \(\sum_{t\le T} \mathbb E\|E_t\|^2=O(T)\)，才能从 Paper 3 的真实迭代不等式推出标准的平均收敛率。盒约束版本的代码和完整逐轮账本见：

- [p3_box_ls_sim.py](../code/p3_box_ls_sim.py)
- [box_ls_results.json](../experiments/box_ls_results.json)

### 8.8 固定总隐私预算下的 T 扫描：不能把 O(T) 条件直接外推

在盒约束最小二乘上进一步固定总预算 \((\varepsilon,\delta)=(8,10^{-5})\)，分别取 \(T\in\{25,50,100,200\}\)，每个点使用三个随机种子。这里的逐轮 Gaussian 标准差由同一个总预算账本重新计算，因此 \(\sigma\) 随轮数上升。DP-TopK 的关键诊断如下：

| T | \(\sigma\) | 测试 MSE | 平均 movement | \(\sum_t\|E_t\|^2/T\) |
|---:|---:|---:|---:|---:|
| 25 | 0.3466 | 0.1530 ± 0.0317 | 0.2244 | 74.1 |
| 50 | 0.4902 | 0.5538 ± 0.3065 | 0.3156 | 182.7 |
| 100 | 0.6932 | 1.4998 ± 0.8872 | 0.4385 | 1,212.3 |
| 200 | 0.9803 | 7.1515 ± 5.0996 | 0.6085 | 13,715.9 |

完整逐 seed 数值和全部统计量在 `box_T_sweep_summary.json` 和 `box_T_sweep/` 中。可靠结论是：在固定总预算下，\(\sigma\) 增长、DP movement 增长，且 \(\sum_t\|E_t\|^2/T\) 严重上升。因而 Candidate Theorem A 中的 \(\sum_t\mathbb E\|E_t\|^2=O(T)\) 只能作为固定或受控的逐轮噪声日程下的条件；它不能在固定总 \(\varepsilon\) 且任意延长 T 时被当作无条件经验规律。

这也改变实验报告方式：任何收敛曲线必须同时报告总预算、逐轮 \(\sigma_t\)、T 和 accountant。增加轮数并不等价于免费增加优化步数；当总隐私预算固定时，后续步的噪声会反过来放大 movement、clipping bias 与累计误差。下一阶段应分别做两条曲线：(i) 固定逐轮噪声，观察条件项是否近似线性；(ii) 固定总隐私预算，报告 privacy-limited utility knee，而不拟合一个独立于预算的 O(T) 定律。

### 8.9 第四个凸目标：simplex 约束 logistic 回归

最后加入带 simplex 约束的二分类 logistic 回归，检验投影几何不是盒约束的偶然特例。实验仍为全参与 \(q=1\)、bounded EControl、fresh aggregate release、真实迭代和中心 DP；参数为 \(n=20,m=40,d=8,T=100,K/d=0.2,\varepsilon=8,\delta=10^{-5}\)，三个随机种子。simplex 投影满足非负性与坐标和为 1 的最大可行性误差 \(6.66\times10^{-16}\)，有限差分梯度最大误差为 \(8.60\times10^{-11}\)。

| 方法 | 目标函数均值 ± SD | 准确率均值 ± SD | \(\sigma\) | 敏感度 | 每客户端比特 |
|---|---:|---:|---:|---:|---:|
| 非私有 EControl-TopK | 0.67209 ± 0.01064 | 0.5859 ± 0.0161 | 0 | 0.100 | 7,000 |
| 中心 DP EControl-TopK | 0.72124 ± 0.02450 | 0.5348 ± 0.0251 | 0.69318 | 0.100 | 7,000 |
| 中心 DP dense DA | 0.74203 ± 0.00934 | 0.5247 ± 0.0125 | 1.03977 | 0.150 | 28,000 |

三个配置的 state age 都为 0。DP-TopK 的平均 movement 为 0.1585，最大累计 \(E_t\) 范数为 4.569，\(\sum_t\|E_t\|^2\) 为 680.72；dense DP-DA 对应为 0.1836、4.975 和 752.48。DP-TopK 的准确率比 dense DP-DA 高约 1.0 个百分点，同时通信量少 4 倍；不过它还具有更小的有界聚合敏感度（0.10 对 0.15），因此不能把效用差异单独解释成稀疏压缩带来的去噪。

盒约束最小二乘、softmax+\(\ell_1\) 和 simplex logistic 三个目标共同支持如下有限结论：同一个 bounded-EControl + fresh-release + real-iterate 接口可以迁移到不同的凸复合结构，并且状态年龄、投影残差和通信账本可以统一记录。它们尚不足以证明跨目标的无条件真实迭代收敛；Candidate Theorem A 仍需对每个目标明确给出 \(L\)、梯度半径、prox/投影非扩张性，以及对 movement-coupled \(E_t\) 的条件。

实现与完整逐轮结果：

- [p3_simplex_logistic_sim.py](../code/p3_simplex_logistic_sim.py)
- [simplex_logistic_results.json](../experiments/simplex_logistic_results.json)

## 12. 跨目标条件表

已将 softmax+\(\ell_1\)、box least squares 与 simplex logistic 的 \(L\)、梯度半径、prox/投影非扩张性、movement-coupled \(E_t\) 和最低账本字段整理成单独核查表：[p3_cross_objective_conditions_zh.md](p3_cross_objective_conditions_zh.md)。该表是进入 FashionMNIST 前的理论审计入口。

### 8.10 固定逐轮 \(\sigma\) 对照：持续 clipping bias 是第二个闭合障碍

为区分总隐私预算账本和 EControl 本身的累计误差，固定每轮 \(\sigma=0.6931788\)，只改变轮数 T；对应的总 \(\varepsilon\) 随 T 增大，而逐轮噪声保持不变。盒约束最小二乘、三个随机种子、DP-TopK 的结果为：

| T | 对应总 \(\varepsilon\) | 测试 MSE | 平均 movement | \(\sum_t\|E_t\|^2/T\) |
|---:|---:|---:|---:|---:|
| 25 | 3.726 | 1.0274 ± 0.1532 | 0.4331 | 158.7 |
| 50 | 5.424 | 1.5125 ± 0.9684 | 0.4412 | 428.9 |
| 100 | 8.000 | 1.4998 ± 0.8872 | 0.4385 | 1,212.3 |
| 200 | 12.000 | 2.1430 ± 1.2143 | 0.4381 | 2,568.7 |

即使逐轮 \(\sigma\) 不变，\(\sum_t\|E_t\|^2/T\) 仍显著上升。当前模拟器把

\(\beta_t=\nabla F_t-\nabla F_{t,\mathrm{clipped}}\)

作为累计误差的一部分，因此只要梯度 clipping bias 长期非零，\(E_t\) 就可能呈线性增长，进而使二阶和呈超线性增长。这个结果排除了“只要控制 DP 噪声日程就自动得到 \(O(T)\)”的表述。

正式理论应拆成两个版本：

1. **Clipped-objective 版本**：把被 clipping 后的梯度定义为目标函数的随机 oracle，令 \(\beta_t=0\)，先证明 bounded EControl + DP fresh release 对 clipped objective 的条件式真实迭代界；
2. **原始-objective 版本**：额外假设 clipping bias 可加和、零均值，或加入单独的 clipping-residual feedback，使 \(\sum_t\mathbb E\|\sum_{s<t}\beta_s\|^2\)=O(T)\)。否则只能给出“优化 clipped objective 加一个可报告的原目标偏差”结论。

因此，下一版 Candidate Theorem A 不再把 \(\beta_t\) 和压缩误差、投影误差无条件地放进同一个 \(E_t\) 闭合式；会先证明 clipped-objective 版本，再把原始目标偏差作为单独项。完整扫描见 [box_fixed_sigma_sweep_summary.json](../experiments/box_fixed_sigma_sweep_summary.json)。

### 8.11 \(E_t\) 闭合引理草案：将理论拆成两个版本

已将 movement-coupled 闭合链整理成独立草案：[p3_Et_closure_lemma_zh.md](p3_Et_closure_lemma_zh.md)。核心候选链为

\[
\mathcal C_T
\le
\kappa_0+\kappa_D\sum_tD_t+\kappa_P\sum_tP_t+\kappa_\Xi\sum_t\Xi_t,
\]

\[
D_t\le K_xM_t+K_\Xi\Xi_t+K_PP_t,
\]

再用真实迭代不等式中的 movement 项吸收反馈。一个候选充分条件是

\[
\mu>\frac{6\kappa_DK_x}{\underline\gamma}.
\]

这条链仍是候选证明接口，不能当作 Paper 3 已证引理直接引用。尤其需要注意，固定逐轮 \(\sigma\) 也不足以控制 \(Q_T/T\)，因为持续的 clipping bias 可能被累积。

因此 Candidate Theorem A 现在正式拆为：

- **版本 A：clipped/bounded objective。** 将被 clipping 的 oracle 直接定义为研究目标，或令 \(\beta_t=0\)。先证明 EControl Lyapunov、movement coupling、投影残差和 DP 方差下的条件式真实迭代界。
- **版本 B：原始未裁剪目标。** 额外要求
  \[
  \sum_{t=1}^T\mathbb E\left\|\sum_{s<t}\beta_s\right\|^2\le K_\beta T,
  \]
  或给出条件零均值、衰减可加和、或单独 clipping-residual feedback 的证明。没有这个条件时，只能报告 clipped objective 与原始目标之间的偏差。

这一区分会成为后续正式证明和 FashionMNIST 实验的入口：先验证版本 A 的条件，再单独测量版本 B 的 clipping bias，而不把两者写成一个无条件收敛结论。

### 8.12 signed projection residual 诊断：修正保守代理的解释

原始 `p3_box_ls_sim.py` 为了在不指定残差方向时保守记录投影误差，把每轮 \(h/e/r\) 投影残差的范数放入 \(E_t\) 的第一坐标。这适合做上界压力测试，但会人为消除不同轮次和不同客户端之间的方向抵消。为避免把该代理误读成理论量，新增 `p3_box_ls_signed_sim.py`，保留真实的 signed residual vector：

\[
\rho_t
=
\frac1n\sum_i(p^h_{i,t}+p^e_{i,t}+p^r_{i,t}),
\qquad
E_t\leftarrow E_t+c_t+\rho_t+\beta_t.
\]

在 clipped-objective 对照中，将样本梯度、\(u\) 和 residual buffer 半径设得足够大，使 \(\beta_t=0\)，并固定逐轮 \(\sigma=0.6932\)。三个随机种子的 DP-TopK 结果为：

| T | 测试 MSE | 平均 movement | \(Q_T/T\) | 最大 \(\|E_t\|\) |
|---:|---:|---:|---:|---:|
| 25 | 0.5170 ± 0.1570 | 0.4532 | 4.26 | 2.49 |
| 50 | 0.9565 ± 0.5439 | 0.4582 | 4.75 | 2.78 |
| 100 | 0.6256 ± 0.3075 | 0.4555 | 7.85 | 3.77 |
| 200 | 0.7117 ± 0.2126 | 0.4536 | 12.26 | 5.22 |

这组结果不能直接证明 \(Q_T=O(T)\)，但说明此前 \(Q_T/T\) 从 158.7 增至 2568.7 的大部分增长来自“将残差范数同向累加”的保守代理。后续正式证明和主诊断使用 signed residual；norm proxy 只作为不允许方向抵消的压力测试。

实现与结果：

- [p3_box_ls_signed_sim.py](../code/p3_box_ls_signed_sim.py)
- [box_signed_clipped_sweep_summary.json](../experiments/box_signed_clipped_sweep_summary.json)

### 8.13 无投影、零 bias 的压缩误差对照

为进一步隔离 Top-K/EControl 本身，使用 signed-residual 模拟器，同时设置足够大的 \(C_0,C_g,B_r,B_h,B_e\)，使梯度 clipping bias 和 \(h/e/r\) 投影残差都为 0；逐轮 \(\sigma=0.6932\) 固定。DP-TopK 三种子结果为：

| T | 测试 MSE | 平均 movement | \(Q_T/T\) | 最大 \(\|E_t\|\) |
|---:|---:|---:|---:|---:|
| 25 | 0.4943 ± 0.1425 | 0.4568 | 11.01 | 3.98 |
| 50 | 0.9794 ± 0.5828 | 0.4611 | 13.22 | 4.55 |
| 100 | 0.5653 ± 0.2945 | 0.4616 | 18.33 | 5.44 |
| 200 | 0.7432 ± 0.2589 | 0.4597 | 29.36 | 7.73 |

该对照显示，在 \(\beta_t=0\)、\(\rho_t=0\) 后，\(Q_T/T\) 的增长明显减缓，但在当前有限 horizon 上仍不能拟合成常数。它支持把 EControl 的 Lyapunov 收缩和输入变化项作为真正需要证明的核心，而不是把所有增长归因于 clipping 或投影。完整摘要见 [box_signed_noprojection_sweep_summary.json](../experiments/box_signed_noprojection_sweep_summary.json)。



### 8.14 诊断修正与软最大/单纯形交叉核查（2026-10-08）

复核发现，旧版 horizon 表格把两个不同量混在了一起。理论中的 signed decomposition 是：

```text
c_t   = H_t - mean_i(u_i,t)
rho_t = mean_i(u_i,t - v_i,t)
beta_t = mean_i(v_i,t - raw_mean_i,t)
E_t   = sum_{s <= t} (c_s + rho_s + beta_s)
```

其中 `raw_mean` 是未裁剪的逐样本梯度平均。旧模拟器把每条消息的 `delta - TopK(delta)` 当成 `c_t`，并把投影残差范数强行同向累加。现在三个模拟器（box least squares、softmax+l1、simplex logistic）都同时记录上述 signed vector、投影残差和累计 `E_t`。

独立审计在 softmax 和 simplex 上验证了同样的代数恒等式：

- `sum_{s <= t} c_s` 与 `H_t`、`e` 投影残差的理论 telescoping 误差小于 1e-16；
- `sum_{s <= t} rho_s` 与 residual-buffer 状态的 telescoping 误差小于 1e-16；
- `E_t` 的逐轮更新与 `c_t + rho_t + beta_t` 的重构误差小于 1e-16。

审计输出见 [`softmax_simplex_telescoping_audit.json`](../experiments/softmax_simplex_telescoping_audit.json)。这不是收敛证明；它只确认实现现在与理论记号一致。softmax 的 `beta_t` 不应被解释为一个已知的梯度场，因而原始目标结论仍需单独处理。

旧版 box horizon sweep 中 `Q_T/T` 的大幅增长主要来自错误的 norm proxy。按理论 signed decomposition 重新计算时，clipped objective 的 `Q_T/T` 在主配置下约为常数（约 0.2 的量级）；原始 least-squares 目标仍显示明显的 clipping bias，这一偏差不能由“每步有界”自动变成 `O(T)` 的累计能量。

同时，旧版 utility knee 不能直接作为方法不稳定的证据：它使用固定 gamma=5 和 last iterate，而理论草案分析的是 averaged iterate 与随 horizon 调整的正则化尺度。后续 sweep 必须把 `gamma_protocol`、`iterate_report`、每轮 sigma、C0/Cg/Bh/Be/Br、seeds、accountant 和总 bit budget 写入 JSON metadata，并分别报告固定逐轮噪声与固定总 epsilon 两种 protocol。

基线也已按相同敏感度重新核查。dense DA 不发送 Top-K index，因此它的通信量应为 `T*d*32` bits/client；只有 Top-K 才支付 `K*(32+ceil(log2 d))` bits/client。后续表格会同时给出原报告的 C0=1.5 dense baseline 与 matched-sensitivity 的 C0=1.0 baseline，避免把中心 DP 的 `2*C0/n` 敏感度差异误报为压缩收益。

当前论文级结论收紧为：在全参与、当前干净聚合 fresh release、状态半径显式有界且真实迭代输出的机制下，理论可先针对 clipped/bounded objective 建立条件式真实迭代界；原始 objective 还需要一个可加和的 clipping-bias 条件或独立 residual feedback。没有这个条件时，不能声称统一 horizon 的无条件 `O(T^(-1/2))` 收敛。

独立的 softmax/simplex checker 还做了一个主动触发投影的 sign-stress：把 Cg、Bh、Be、Br 缩小后，`raw-new` 残差约定下的 c、rho telescoping 误差仍小于 7e-16，而把同一残差误当成 `new-raw` 会产生 1e-1 到 1 量级的误差。实现、报告和紧凑 JSON 分别见 [`tele_scope_softmax_simplex_audit.py`](../repro/tele_scope_softmax_simplex_audit.py)、[`tele_scope_softmax_simplex_audit.md`](../docs/tele_scope_softmax_simplex_audit.md) 与 [`tele_scope_sign_stress.json`](../experiments/tele_scope_sign_stress.json)。

headline driver 现在额外输出 matched-sensitivity dense DA（C0=1.0，与 Top-K 的 `2*Bh/n` 对齐）。在当前三种子设置下，box least-squares 的 test MSE 从历史 dense 6.743 +/- 4.372 降到 3.056 +/- 2.067；softmax objective 从 1.662 +/- 0.177 降到 1.255 +/- 0.088；simplex objective 从 0.7420 +/- 0.0093 降到 0.7383 +/- 0.0194。这些数字只用于暴露敏感度 confound，不能单独归因于 Top-K。
