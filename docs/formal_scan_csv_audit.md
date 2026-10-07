# `formal_scan_results.csv` 审计报告

审计对象：`upload/formal_scan_results.csv`（34 行含表头，33 条记录，22 列）。

## 结论

原 CSV 没有发现数值级明显错误；不建议直接替换或覆盖。它可以作为实验结果表继续使用，但不应单独作为可复现实验归档，因为关键运行元数据不在 CSV 中。对应的 `formal_scan_results.json` 含有完整 metadata，尤其是 `samples_per_client=40`、`n_clients=100`、`d=20` 和 accountant 配置。

## 已核查项目

### 1. Prefix 的 no-amplification 标记

`prefix_bs_scan_T10_no_amplification` 共 15 条记录：

- `q ∈ {1.0, 0.5, 0.2}`，每个 q 有 5 个 `Bs` 值；
- `accountant_q=1.0` 在全部 15 条记录中均存在；
- `sensitivity=2*Bs`，不随 q 变化；
- bits/client 分别为 7400、3700、1480，符合按活跃比例计通信量；
- section 名称中的 `no_amplification` 与数据一致。

因此 prefix 行的“持久状态、无抽样放大”标记是正确的。

### 2. Equal-bits 行数与一致性

`equal_bits_scan_Bh1` 共 9 条原始记录：

| q | rounds | seeds | bits/client |
|---:|---:|---:|---:|
| 1.0 | 100 | 0,1,2 | 7400 |
| 0.5 | 200 | 0,1,2 | 7400 |
| 0.2 | 500 | 0,1,2 | 7400 |

`equal_bits_aggregates` 共 3 条汇总记录，均与 3 个原始 seed 行重新计算的均值和样本标准差一致。这里的 `objective_sd` 与 `accuracy_sd` 是三种子样本标准差（ddof=1），不是标准误。

### 3. Bh 扫描

`bh_scan_q02_T100_seed0` 共 3 条，是单种子诊断；section 名称包含 `seed0`，但 `seed` 列为空。这不改变数值，但属于可追溯性缺口。

`bh_scan_q02_3seed_aggregates` 共 3 条，汇总值与 3 个 equal-bits seed 行之外的独立 Bh 扫描结果相符；该组同样没有填 `seed`，这是汇总行的正常表现。

### 4. 敏感度与噪声关系

- active-average Bh 行满足 `sensitivity=2*Bh/(q*n)`，其中 `n=100`；
- prefix 行满足 `sensitivity=2*Bs`；
- 同一 accountant 和 epsilon 下，sigma 随敏感度线性缩放；
- equal-bits 三个 q 的 bits/client 均为 7400。

## 发现的结构性缺口

### A. CSV 没有 samples_per_client

CSV 没有 `samples_per_client` 列。该值可从同目录 JSON metadata 恢复为 `40`，但如果 CSV 被单独分享，无法复现实验数据规模。

### B. CSV 缺少全局实验 metadata

CSV 本身没有以下关键字段：

- `epsilon=8.0`；
- `delta_dp=1e-5`；
- `n_clients=100`；
- `d=20`；
- `topk_frac=0.1`；
- `C0=1.0, Cg=1.0, Br=2.0, Be=2.0`；
- `gamma=5.0, l1=0.002`；
- accountant 名称 `without_replacement_bound`；
- seed 集合 `{0,1,2}`。

建议在以后每行加入 `run_id` 并配套 sidecar metadata，或在 CSV 前增加独立 metadata 文件；不建议把同一 metadata 重复粘贴到每一行。

### C. 汇总行字段不完全自洽

`equal_bits_aggregates` 和 `bh_scan_q02_3seed_aggregates` 缺少 `sensitivity`、`accountant_q`，部分情况下也缺少 `bits_per_client`（Bh 汇总行）。这不构成错误，但会使下游脚本必须按 section 推断机制。

建议未来汇总行保留：`mechanism`, `accountant_q`, `sensitivity`, `bits_per_client`, `n_seeds`。

### D. Bh 单种子行应填 seed=0

建议将 `bh_scan_q02_T100_seed0` 的三条记录补成 `seed=0`，或将 section 名称改为不编码 seed。两者选其一，避免重复来源。

## 推荐的未来 schema

保留当前长表结构，并增加：

- `mechanism`：`econtrol_active_average` / `prefix_ftrl`；
- `mode`：`raw` / `aggregate`；
- `n_clients`；
- `samples_per_client`；
- `dimension`（替代不够明确的 `d`）；
- `epsilon_dp`、`delta_dp`；
- `accountant`；
- `n_seeds`；
- `seed`（汇总行留空）；
- `sensitivity_formula` 或至少 `accountant_q`；
- `bits_per_client` 和 `total_bits_per_client` 的定义说明。

## 处理建议

1. 原 CSV 保持不动，作为当前版本的原始导出。
2. 论文表格引用时同时固定 JSON metadata；不要仅给 CSV。
3. 生成下一版时填入 `samples_per_client=40`、`seed=0`（Bh 单种子行），并给汇总行增加 `n_seeds=3` 与机制/accountant 字段。
