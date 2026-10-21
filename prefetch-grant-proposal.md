# 预取授予：L3→L2 预取的生命周期契约与 host 层所有权

> **提案（2026-10-21）**。本文是一份自包含的技术提案：读者无需任何前置上下文即可理解其问题、机制、方案与实验设计。全文围绕一个可在未修改代码上直接测量的判据（兑现率 \( \rho \)，§6.1）与一组分层改动（R1–R4，§5.0）展开。
>
> **一句话概括**：在 SGLang HiCache 的 cache 模式下，从 L3（远端共享存储池）预取到 L2（本机 host DRAM）的 KV 页，在「已经取到」与「已经被用」之间存在一段没有任何保护的时间窗口。高并发排队把这段窗口拉长到足以让淘汰逻辑删掉刚取上来的页，于是端到端命中率归零、存储后端带宽先有后无。本文把这段窗口命名为风险窗口 \( W \)，证明现有保护恰好覆盖 \( W \) 的补集，并给出一套「授予（grant）」语义使保护覆盖 \( W \) 本身。
>
> **阅读路线**：§0 导读（术语表 + 一段话读懂问题 + 预答六个反驳）→ §1 主张 → §2 问题（含 HiCache 背景 §2.0 与四个机制 §2.3–§2.6）→ §3 新颖性边界（含与 TempoKV / PEEK / Continuum / TierKV 的深度对照）→ §4 代码支点（含三条既有路径为何不够）→ §5 方案（R1–R4）→ §6 go/no-go 判据 → §7 实验设计 → §8 风险 → §9 投稿定位。若只想判断「这个问题是否真实存在」，读 §0.2、§2.1、§6.1 即可；若只想判断「是否与 TempoKV / PEEK 重复」，读 §3.4 即可。
>
> **贡献摘要**（四条，均可独立验证）：
>
> 1. **一个定位**：把「多轮命中率 0%、后端带宽归零」这一现象归因到一个此前未被命名的时间窗口 \( W = [t_{\text{land}}, t_{\text{admit}}) \)，并证明 SGLang 现有的 host 保护覆盖区间 \( [t_{\text{issue}}, t_{\text{land}}] \) 与 \( W \) **零重叠**（§2.3）。
> 2. **一个判据**：兑现率 \( \rho \)——落地时记录的授予长度与准入时实测的仍在长度之比。它在**未修改的代码上**即可测量（§6.3 的一行日志），且是确定性的：\( \rho < 1 \) 与 \( \rho \equiv 1 \) 之间没有解释空间。
> 3. **一套分层方案**：R1（授予生命周期，≈60 行，实质是把一次 `pop` 改成一次 `transfer`）、R2（截断即可观测失败）、R3-a（队列序仲裁稀缺容量）、R3-b（laxity 门控发放时机）、R4（部分完成重试）。最小可发表单元是 R1（§5.0）。
> 4. **一组反证设计**：六条可证伪判别 F-a–F-f（§3.3），其中 F-e 由一个**零成本**的既有开关对照臂 A7 承担——若 A7 使 \( \rho \equiv 1 \)，本文主张即被证伪；九臂对照（§7.4）含 PEEK 式软保护的可运行代理 A8；十项消融（§7.5）含 R3-b 的内建关闭端点 B10。
>
> **与已发表工作的关系（结论前置）**：本文与 TempoKV [cite:12]、PEEK [cite:13]、Continuum [cite:14]、TierKV [cite:15] 在词汇上高度重叠，但作用对象各不相同。最短的区分是三条：TempoKV 修「承诺太早」，本文修「释放太早」，两者位于同一最优时间点的两侧；PEEK 用**软**保护（改淘汰打分）护住树上**已存在**的祖先块，本文用**硬**保护（引用计数硬跳过）护住刚从 L3 拉上来、**在树上尚不存在**的新生叶子；Continuum 与 Talaria 保护「已消费、可能回访」的 KV，本文保护「已就位、尚未消费」的 KV。完整论证见 §3.2 与 §3.4。
>
> **目标代码库**：`SGLang-private-hicache-kv-backend-0.5.20`，路径根 `python/sglang/srt/`。**版本锚定说明**：该工作副本**不含 `.git` 目录**，`python/sglang/version.py` 采用动态导入，因此无法用 commit hash 锚定；本文所有行号以该副本当前状态为准。公开侧对照信息：PyPI 上 sglang 最新发布为 2026-10-01[cite:23]，官方文档称 “UnifiedRadixTree is now the default”，HiCache × UnifiedRadixTree 相关 PR 为 #23316 / #23391 / #24691，并含 “stability fixes across cascade eviction, tombstone replay, and partial-match paths”[cite:23]。**行号在投稿前必须按目标 commit 重新校验**（§8 风险 11）。
>
> **修改幅度授权**：允许大幅改动。但 §5 的分层发布把最小可发表单元（R1）收敛到约 60 行，且这 60 行的实质动作是把一次 `pop` 改成一次 `transfer`——所依赖的数据在原处已经齐备。
>
> **论断规范**：GFM；行内数学一律用 `\( \)`；`[cite:N]` 紧跟其所支撑的论断；未亲自读码或检索核实的内容标 `[unverified]`；全文不用第一人称；Sources 置于末尾。

---

## 0. 导读

### 0.1 术语表

| 术语 | 含义 | 详见 |
|---|---|---|
| L1 / L2 / L3 | 三层 KV 存储：GPU HBM / 本机 host DRAM / 远端共享存储池（Mooncake 一类） | §2.0.1 |
| HiCache | SGLang 的分层 KV 缓存子系统，负责三层之间的迁移 | §2.0.1 |
| cache 模式 | `--hicache-host-memory-mode cache`（默认）：host DRAM 作为**持久**的 L2 层，跨请求复用 | §2.0.1 |
| buffer_only 模式 | host DRAM 仅作为 GPU 与存储后端之间的**临时**中转缓冲，不做跨请求 L2 复用 | §2.8 |
| 预取（prefetch） | 请求准入前，把 L3 命中的前缀 KV 提前搬到 L2 的动作 | §2.0.2 |
| \( t_{\text{issue}} \) | 发起预取的时刻：锚点节点被钉住，占用计数自增 | §2.0.2 |
| \( t_{\text{alloc}} \) | 授予 host 页的时刻：分配失败先淘汰再重试，再失败则截断 | §2.0.2 |
| \( t_{\text{land}} \) | L3→L2 传输完成、页入树的时刻：锚点解钉、认领记录被销毁、计数递减 | §2.0.2 |
| \( t_{\text{admit}} \) | 请求准入的时刻：`load_back` 消费授予页，此时才真正 `protect_host()` | §2.0.2 |
| \( W \)（风险窗口） | \( [t_{\text{land}}, t_{\text{admit}}) \)：页已入树、引用计数为零、在淘汰候选集合中，完全无保护 | §2.2 |
| 授予（grant） | 本文提出的具名对象：一次预取对若干 host 页的所有权，带状态机与释放点 | §5.1 |
| 锚点（anchor） | 预取发起时树上已存在的那个最深节点，被钉住以防前缀在传输期间被删 | §2.3 |
| 新生叶子 | `_insert_helper_host` 在 \( t_{\text{land}} \) 创建的、承载刚落地页的树节点 | §2.3 |
| 截断 | host 池空间不足时，只预取「放得下的页对齐前缀」，其余部分静默丢弃且记为成功 | §2.6 |
| 兑现率 \( \rho \) | 准入时实测仍在的存储来源命中长度 / 落地时记录的授予长度 | §6.1 |
| \( G_{\max} \) | 反死锁钳制：已落地未消费授予总量的硬上界 | §5.3 |
| R1–R4 | 分层发布的五个改动层（R3 拆为 R3-a / R3-b） | §5.0 |
| A0–A8 | 九个对照臂 | §7.4 |
| B1–B10 | 十项消融 | §7.5 |
| C1–C4 | 四条正确性断言 | §5.7 |
| F-a–F-f | 六条可证伪判别 | §3.3 |

**编号约定**：`C` 前缀为正确性断言，`A` 前缀为对照臂，`B` 前缀为消融，`F` 前缀为可证伪判别，`R` 前缀为发布层。`R3` 仅作为 `R3-a` 与 `R3-b` 的统称出现（§5.2），任何具体主张都落在 `R3-a` 或 `R3-b` 上。

### 0.2 一段话读懂问题

请求进入系统时，框架先去 L3 查前缀命中，命中就把这段 KV 提前搬到本机 host 内存（L2）。搬运需要时间，于是请求本身在等待队列里排队。搬运完成的那一刻，框架做了三件事：**解开对树上锚点节点的保护、销毁「这个请求认领了这批页」的记录、把占用计数减回去**。但请求此时**还没被调度**——它还在队列里等。于是这批刚搬上来的页处于一个尴尬状态：已经在树上（因此是淘汰候选）、引用计数为零（因此不被淘汰逻辑跳过）、没有任何记录说明它们属于谁（因此无从追回）。等请求终于被准入、去取这批页时，页可能已经被后来者的预取淘汰掉了。并发越高、队列越长，这段间隔越长，被删的概率越高；单并发下间隔近零，问题消失。这就是「多轮命中率 0%」与「后端带宽先有后无」的来源：数据确实取上来了，也确实还在 L3 里，但取上来之后在被用之前就被扔了，而框架把截断和淘汰都记为成功，不再重试。

第二个根因与第一个同源：申请 host 页时若空间不够，框架先淘汰别人、再重试，仍不够就**只取放得下的那一截**，并把这次部分成功记为完全成功（`return True`）。因为记为成功，重试机制不会介入；因为原始命中长度被就地改写，事后连「少了多少」都无法测出。于是 128K 的需求只命中 40K，且长度分布呈「谁先占到谁多拿」。

### 0.3 预先回答的六个反驳

| 反驳 | 回答所在 | 一句话答案 |
|---|---|---|
| 「把 `--hicache-storage-prefetch-policy` 调成 `wait_complete` 不就行了？」 | §4.5.1 + A7 臂 + F-e | 不行。`wait_complete` 的门控条件是「op 仍在 `ongoing_prefetch` 里」，而 cache 模式在 \( t_{\text{land}} \) 就 `del` 掉了它，门控恰好在窗口开启的瞬间失效。**这条已被设计成可证伪的：若 A7 使 \( \rho \equiv 1 \)，本文主张作废。** |
| 「打开 `enable_hicache_prefetch_retry` 让它重试不就行了？」 | §4.5.2 + B6 消融 | 不行。该开关的官方 help 自述语义域是「因瞬时限制被 **skipped**」，而本文两个根因分别是「成功完成但被淘汰」与「成功发起但被截断」，都不属于 skipped；且截断分支根本不调 `poll_miss`——一页都没分到反而会注册重试，分到一部分反而不注册。 |
| 「仓库里已经有 `session_ref` 会话保留，官方 blog 说它作用于 L1/L2。」 | §4.5.3 | 不覆盖 \( W \)。它的注册点在**请求完成**路径（要求 `req.last_node is not None`），生命周期是**跨轮会话**，性质是**软排序**而非硬保护；且 FULL 组件的 LRU 谓词直接返回 `None`。 |
| 「直接切到 `buffer_only` 模式不就免疫了？」 | §2.8 + A1 臂 | 确实免疫（\( \rho \equiv 1 \) 由构造保证），但代价是放弃 L2 的持久层身份，即放弃跨请求与跨实例复用——这恰是本文硬约束要保住的东西；且它有四条硬前置，含「不能跑在 decode 实例上」。A1 臂给出定量代价。 |
| 「给 host radix tree 加锁不就行了？」 | A6 臂 + §5.3 | 能修 \( \rho \) 但会 wedge 准入：整树禁淘汰使 host 池可用容量骤降，写回 staging 分配失败率上升。A6 与 A4 的差别不在 \( \rho \)（都为 1.0），而在容量与前进性——这正是 \( G_{\max} \) 与 C3 断言的价值。 |
| 「PEEK / TempoKV 已经做了。」 | §3.4.1 / §3.4.2 + A8 臂 + F-f | 对象不同。PEEK 软保护树上**已存在**的祖先块，本文硬保护**新生**叶子；PEEK 的机制在原理上不能消除截断（打分不影响 alloc 成败）。TempoKV 修承诺太早，本文修释放太早。A8 把 PEEK 的淘汰钩子实现为受控臂，用数据而非论证回应。 |

---

## 1. 主张

L3→L2 的预取在 cache 模式下**没有所有权语义**：它有一个计数器，但没有一条记录；有保护原语，但保护窗口与风险窗口恰好错开；有授予动作，但授予在落地的瞬间就被销毁。

源码里已经把两种模式的差别用两个不同的动词写清楚了。[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2531-L2537) 中 `_prefetch_occupied_span` 的 docstring 逐字为：cache 模式 “**reserves** the requested span at enqueue”，buffer 模式 “**grants** at hit-alloc, sized to the allocation”。这两个词不是修辞差异，是两种记账制度的差异——`reserve` 是对**请求跨度**的限流记账，`grant` 是对**实际授予页**的生命周期记账。cache 模式停在 `reserve`，因此它既不持有授予，也就无从保护授予。

由此得到本文的中心命题：

> **保护窗口是脆弱窗口的精确补集。** 预取的 host 保护（`protect_host` / `host_ref_counter`）覆盖 \( [t_{\text{issue}}, t_{\text{land}}] \)，而页真正可被淘汰的区间是 \( [t_{\text{land}}, t_{\text{admit}}) \)。两段区间**零重叠**。这不是保护不够强，是保护根本不在正确的时间段上。

以及由它直接导出的修复方向：

> 把预取从「限流计数」升级为「带生命周期的 host 内存授予」，令 \( \text{release}(g_r) = \text{admit}(r) \) 而非 \( \text{release}(g_r) = \text{complete}(g_r) \)。授予按等待队列序发放；发放不出就 park 而不是截断；截断成为一等可观测失败而非静默成功。

一个概念同时覆盖两个根因——「预取太早被后来者挤掉」与「空间不够只预取放得下的部分」。而「按请求顺序取、取不到就等」这个被认为实现成本过高的解法，在该模型里不是独立机制，只是**授予发放策略**的一个特例：把 park 从 buffer 专属分支扩展到 cache 分支，并在扫描时对 park 者 `break` 而非 `continue`。

---

## 2. 实际问题

### 2.0 背景：SGLang HiCache 的三层结构、两种 host 模式与一次预取的完整时序

本节把后文所需的全部背景一次讲清，使 §2.1 起的内容可以自包含地阅读。

#### 2.0.1 三层与两种模式

SGLang 的 HiCache 把 KV 缓存放在三层上：

| 层 | 物理介质 | 容量量级 | 谁管理淘汰 |
|---|---|---|---|
| **L1** | GPU HBM | 最小，受静态显存分配约束 | 设备侧 radix cache 的 LRU |
| **L2** | 本机 host DRAM | 中等，`hicache_ratio × L1`（cache 模式默认 `2.0`） | `evict_host`（本文的战场） |
| **L3** | 远端共享存储池（Mooncake 一类） | 最大，官方 benchmark 用 500 GiB | 存储后端自身 |

三层由同一棵 radix tree 索引。树节点上挂两份数据指针：`value`（L1 物理槽，`None` 表示已不在 HBM）与 `host_value`（L2 物理槽，`None` 表示不在 host 上）。节点的两个派生属性因此是（[radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/radix_cache.py#L262-L302)）：

```python
    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None
```

**关键推论**：一个刚从 L3 取回、只存在于 L2 的节点，其 `value is None`，故 `evicted == True`——它在设备侧的语义里是「已被淘汰的节点」，同时它 `backuped == True`。这个组合正是 \( W \) 里那批页的状态：**在树上、可被 host 侧淘汰、且被设备侧视为已淘汰**。

host 层有两种工作模式，由 `--hicache-host-memory-mode` 选择，只有两个合法取值：

| 模式 | 语义（官方 help 逐字） | L2 是否跨请求复用 |
|---|---|---|
| `cache`（默认） | “host memory is a **persistent HiCache tier**” | 是 |
| `buffer_only` | “a **transient staging buffer** between GPU and the storage backend” | 否 |

本文全程讨论 **cache 模式**。`buffer_only` 在 §2.8 作为免疫性对照出现——它不受本文问题影响，恰恰因为它放弃了 L2 的持久层身份。

#### 2.0.2 一次预取的完整时序

设请求 \( r \) 到达，其前缀在 L3 命中 \( h \) 个 token。当 \( h \) 超过阈值（官方默认 256 token，可配[cite:24]）时触发预取。整条时间线上有四个决定性时刻：

| 时刻 | 发生了什么 | 树 / 账本状态 | 代码位置 |
|---|---|---|---|
| **\( t_{\text{issue}} \)** | 发起预取。**锚点节点**（树上当时最深的那个已存在节点）被 `protect_host()` 钉住；占用计数 `prefetch_tokens_occupied += len(prefetch_key)` | 目标页尚不存在；锚点 `host_ref_counter = 1` | `hiradix_cache.py:1985`；`unified_radix_cache.py:1990-1994`；计数自增 `hiradix_cache.py:2006`、`unified_radix_cache.py:2059` |
| **\( t_{\text{alloc}} \)** | 向 host 池申请 \( h \) 个页。失败则先 `evict_host(h)` 再重试；仍失败则**截断**到「放得下的页对齐前缀」 | 页已分配、尚未入树；锚点仍钉住 | `unified_radix_cache.py:2738-2751` |
| **\( t_{\text{land}} \)** | L3→L2 传输完成。`_insert_helper_host` 把页挂进树，成为**新生叶子**的 `host_value`；随后**锚点解钉**、认领记录 `pop` / `del` 销毁、占用计数递减 | 新生叶子 `host_ref_counter = 0`、`value is None`，并被 `_update_host_leaf_status` 立即加入 `evictable_host_leaves` | `hiradix_cache.py:1825`→`1837`→`1838`；`unified_radix_cache.py:2215`→`2216` |
| **\( t_{\text{admit}} \)** | 请求终于被准入。`load_back` 把 L2 页搬回 L1 消费；**此时才**对沿途节点 `protect_host()` | 页 `host_ref_counter ≥ 1`，`evict_host` 硬跳过 | `hiradix_cache.py:1455-1456`、`1422-1497` |

这四个时刻把预取生命周期切成三段：\( [t_{\text{issue}}, t_{\text{land}}] \) 是框架**认为**自己在保护的区间，\( [t_{\text{land}}, t_{\text{admit}}) \) 是页**实际**可被淘汰的区间。本文的全部论点建立在这两段互不重叠这一事实上（§2.3、§3.1）。

#### 2.0.3 淘汰是怎么发生的

host 侧淘汰由 `evict_host(need)` 驱动（`hiradix_cache.py:1387-1420`）。它的候选集合是 `evictable_host_leaves`，逐个检查跳过条件：

```python
            if not x.evicted:
                continue
            ...
            if x.host_ref_counter > 0:
                continue
```

第二条是**唯一的硬保护**（L1403-1404）：只要引用计数为正，无论需要淘汰多少页，该节点都被跳过。这正是本文要复用的原语——R1 不新造保护机制，只是把这次计数的**发生时刻**从 \( t_{\text{admit}} \) 提前到 \( t_{\text{land}} \)。

候选集合的维护函数 `_update_host_leaf_status`（L1222-1235）只看 `evicted` 与 `lock_ref`，**完全不看 `host_ref_counter`**，也不看是否存在 in-flight 或已落地未消费的预取。因此刚承载预取页的新生叶子在创建的那一刻就进入候选集合。这本身不构成 bug：`evict_host` 在真正摘除前有硬跳过。但它意味着**保护完全依赖引用计数**——一旦计数为零，该页就是最优先被淘汰的对象之一，因为它既是最新创建的叶子，又满足 `evicted == True`。

#### 2.0.4 谁在什么时候调用 `evict_host`

三类调用者，其中第二类是本文的对抗面：

1. **写回路径**（`hiradix_cache.py:883`）：`write_backup` 分配 host 槽失败时淘汰，为新数据腾地方。
2. **预取授予路径**（`unified_radix_cache.py:2738-2751`）：**为请求 A 授予页的动作，本身就是淘汰请求 B 已落地页的那一行**（§2.5）。这是本文最关键的因果结构：淘汰不是外部噪声，而是同一机制的自我对抗。
3. **准入路径**（`hiradix_cache.py:638`）：`load_back` 需要空间时淘汰。

三者都在每轮事件循环里被驱动：`get_next_batch_to_run` 开头调 `_process_hicache_events()` → `check_hicache_events()`（`scheduler.py:3685-3695`、`hiradix_cache.py:1594-1626`），把已完成的预取结果入树。因此 \( W \) 里的页**每轮都被扫一次**：队列越长、轮数越多，被扫的次数越多，被摘走的概率单调上升。

#### 2.0.5 为什么这个窗口在别的系统里不存在

这一点决定了本文的新颖性边界，必须先说清。在单请求 DMA 重叠模型（如 [cite:1] 的 chunk-aware prefetch）里，预取与计算在同一条请求内流水重叠，传输完成即被消费，\( t_{\text{land}} \) 与 \( t_{\text{admit}} \) 之间的间隔**恒为零**。在 lookahead 模型（如 [cite:15] 的 deterministic lookahead window）里，预取窗口按「未来 \( k \) 次迭代」对齐，落地即进入使用区间，同样没有裸露的间隔。

SGLang 的 \( W \) 之所以存在，是因为它把**预取的完成**与**请求的准入**解耦成了两个独立事件：前者由存储后端的 IO 进度决定，后者由调度器的批组装决定（`PrefillAdder` 按可用 token 预算准入）。两者之间隔着一次排队，而排队时长在多用户高并发下**没有上界**。这就是「单并发下问题消失、并发越高命中率越低」的结构性原因，也是本文与所有单请求预取工作的分界线。

### 2.1 现象

观测到的四项事实：

1. 多轮对话场景下，L3 侧明明存在命中（`storage_hit_count > 0`、预取任务成功完成、日志打印 `prefetch success`），但请求真正进入调度时 `load_back` 拿不到对应的 host 页，退化为重算，端到端表现为**多轮请求 KV 命中率接近 0**。
2. 存储后端带宽在压测初期有流量，随后迅速归零——不是因为无命中可取，而是因为**取得的东西在消费前就没了**，且框架不再重复发起（重试被三重封锁，见 §2.6）。
3. 并发度越高、排队越长，命中率越低；单并发下问题消失。
4. 同一轮压测中，第二轮的部分请求只命中一小段前缀（例如 128K 需求只命中 40K），且这一小段的长度分布呈明显的「谁先占到谁多拿」。

**公开可引的宏观同构证据**。SGLang 官方 Unified Radix Cache deep dive 给出的多轮 benchmark 中，存在一条与上述事实 1、3 同构的曲线：DeepSeek-V4-Flash（FULL+SWA）、4×H200 TP4、48 clients、60 rounds、每轮 4,096 in + 16 out，按轮次绘制的 hit rate 显示——Round 20 时 L1 ≈0% 而 L1+L2 ≈95%；Round 25–35 维持在 L1+L2 ≈96–97%；**到 Round 40–55，L1 ≈0%、L1+L2 ≈0%，而 L3（500 GiB Mooncake）保持 ≈98%**[cite:22]。第二个配置（Inkling-Small，FULL+SWA+MAMBA，8×H200 TP8，64 clients，30 rounds）崩塌得更陡：**Round 17 L1+L2 ≈95% → Round 18 ≈62% → Round 19 ≈0%**，同期 L3 保持 ≈95%[cite:22]。

这条曲线的意义不在数值本身，而在其**形状**：host tier（L2）在长程多轮下**先扛住、后突然崩塌到零**，而更慢的 L3 反而稳定。若 L2 的失效是容量不足的自然 LRU 结果，曲线应当是渐进下滑并停在某个非零平台（因为热前缀仍会被优先保留）；观测到的是**断崖式归零**，而同期 L3 命中率不变说明「数据确实还在、也确实被取上来过」。断崖 + L3 稳定 + 端侧表现为重算，三者同时成立时，最简解释是：**取上来的页在被消费前被丢弃了**——即本文的 \( W \) 窗口暴露。该 benchmark 因此既是动机图（§7.2），也是 A0 臂的预期基线形状。

用快递柜比喻描述这套现象，术语替换如下：

| 比喻 | 本文术语 | 代码位置 |
|---|---|---|
| 小区快递柜（格子有限） | host 内存池 `mem_pool_host` | `cache_controller` 持有 |
| 收件人 | 请求 `req` | `waiting_queue` 中的元素 |
| 快递员把包裹从仓库放进柜子 | 预取 L3→L2 落地 | `_try_alloc_storage_hit` + `mem_pool_host.alloc` |
| 下单排队 | 入队，发起预取的时刻 \( t_{\text{issue}} \) | [scheduler.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/managers/scheduler.py#L3339-L3340) 中 `_prefetch_kvcache(req)` 在 `waiting_queue.append(req)` **之前**调用 |
| 收件人终于来取件 | 准入 \( t_{\text{admit}} \)，真正 `load_back` | `load_back` 路径 |

比喻里有一处必须修正，且这处修正是本文的关键：**「柜子满了按 LRU 把 A 的包裹挤掉」不是被动发生的**。淘汰 A 的那一行代码，正是为 B 分配预取内存时主动调用的 `self.evict_host(alloc_len)`（§2.5）。B 的授予动作就是 A 的丢失原因，二者在同一条调用栈上、同一次事件循环里。这不是资源竞争的附带结果，是**授予路径内部的结构性副作用**。

### 2.2 三个时间点与风险窗口

根因描述给的是两个时间点错配（T1 入队发起预取 / T2 真正进调度）。读码后需要拆成四个时刻，因为「预取完成」既不等于 T1 也不等于 T2，而正是它构成了窗口的起点：

| 时刻 | 事件 | 代码 |
|---|---|---|
| \( t_{\text{issue}} \) | 发起预取；**锚点节点**被钉住；`prefetch_tokens_occupied += len(prefetch_key)` | `hiradix_cache.py:1985` `protect_host()`；`unified_radix_cache.py:1990-1994` `inc_host_lock_ref`；自增点 `hiradix_cache.py:2006`、`unified_radix_cache.py:2059` |
| \( t_{\text{alloc}} \) | 授予 host 页；**不足则先 `evict_host` 再重试分配；再不足则截断到放得下的页对齐前缀** | `unified_radix_cache.py:2738-2751` |
| \( t_{\text{land}} \) | L3→L2 传输完成，页入树成为新生叶子的 `host_value`；**锚点解钉、认领记录 `pop` 销毁、计数递减** | `hiradix_cache.py:1825`→`1837`→`1838`；`unified_radix_cache.py:2215`→`2216` |
| \( t_{\text{admit}} \) | 请求准入，`load_back` 消费授予页，此时才 `protect_host()` | `hiradix_cache.py:1455-1456` |

风险窗口 \( W = [t_{\text{land}}, t_{\text{admit}}) \)。它的三个性质决定了故障的严重度：

- **\( W \) 的长度无上界**。请求在 `waiting_queue` 中停留多久，\( W \) 就多长。停留时长由并发度、batch 组成、解码长度共同决定，与本机制无任何耦合。
- **\( W \) 内页处于「已入树、零引用」状态**。`_insert_helper_host` 创建的新节点 `value is None`，故 `evicted == True`（[radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/radix_cache.py#L286-L287)），被 `_update_host_leaf_status` 直接加入 `evictable_host_leaves`（`hiradix_cache.py:2043`），而新节点的 `host_ref_counter == 0`。
- **\( W \) 每轮事件循环被扫过一次**。`get_next_batch_to_run` 开头即执行 `_process_hicache_events()`（`scheduler.py:3685-3695`），其中 `tree_cache.check_hicache_events()`（`hiradix_cache.py:1594-1626`）触发淘汰逻辑。请求在队列里每多等一轮，就多被扫一次。
### 2.3 机制一：保护窗口是脆弱窗口的精确补集

钉住原语本身是完备的，且其**设计意图就是为存储操作服务**。[radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/radix_cache.py#L262-L302)：

```python
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        ...

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")
```

注释 “incremented when the node is referenced by a **storage operation**” 说明这套计数器就是为预取/回写这类跨层操作准备的。`release_host` 的 `RuntimeError` 是真实的不变式检查，可直接用作修复后的正确性断言（§5.7 C1）。

淘汰侧确实尊重这个计数器。[hiradix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/hiradix_cache.py#L1387-L1420) 的 `evict_host`：候选来自 `evictable_host_leaves`，`if not x.evicted: continue`，`if x.host_ref_counter > 0: continue`，通过后 `num_evicted += self.cache_controller.evict_host(x.host_value)` 并把节点从父节点摘除。另一处消费者 `hiradix_cache.py:1350` 是 `if any(n.host_ref_counter > 0 for n in nodes)`。正确用法的示范在 `hiradix_cache.py:1455-1456`：准入时 `for n in nodes_to_load: n.protect_host()`。

问题在于**钉的是谁、什么时候解钉**。四个阶段的对照：

| 阶段 | host 页状态 | 被钉住的对象 | 能否被 `evict_host` 淘汰 |
|---|---|---|---|
| \( [t_{\text{issue}}, t_{\text{alloc}}) \) | 尚未分配（懒分配） | **锚点节点**（`hiradix:1985` `protect_host()`；`unified:1990-1994` `inc_host_lock_ref`） | 无页可淘汰 |
| \( [t_{\text{alloc}}, t_{\text{land}}) \) | 已分配，operation 持有，**不在树上** | 无（仅 `ongoing_prefetch` 持引用） | 淘汰逻辑看不到它 |
| \( [t_{\text{land}}, t_{\text{admit}}) \) | **已入树成为 `host_value`** | **无**（锚点已在 `hiradix:1837` / `unified:2214` 解钉；新节点 `host_ref_counter == 0`） | **能** |
| \( [t_{\text{admit}}, \dots) \) | 入树 | `protect_host()`（`hiradix:1455-1456`）+ `inc_lock_ref(ancester_node)` | 不能 |

第一阶段钉的是**锚点**——请求前缀在树上的既有末端节点，与本次预取将得到的页毫无关系。第二阶段页不在树上，淘汰逻辑枚举不到它，所以「不需要」保护。第三阶段页刚入树，而保护已经在前一行被释放了。

完成路径的顺序把这一点写得很直白。[hiradix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/hiradix_cache.py#L1816-L1842) `_handle_prefetch_result`：L1825 `self.ongoing_prefetch.pop(req_id)` 销毁认领记录 → L1829-1834 `_insert_helper_host` 把页插入树 → L1836 `free(host_indices[:matched_length])` → L1837 `release_host()` 解钉锚点 → L1838 `prefetch_tokens_occupied -= len(prefetch_key)` → L1841-1842 记录 `loaded_from_storage = min_completed_tokens - matched_length`。

unified 分支的注释更值得注意，因为它**主动承认了准入才是真相时刻**。[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2205-L2233)：

```python
self.cache_controller.mem_pool_host.free(host_indices[: insert_result.prefix_len])
loaded_from_storage = completed_tokens - insert_result.prefix_len
# Cache mode has only completed L3 -> L2 here. Keep the usable
# storage span unresolved until admission proves that L2 -> L1
# load-back actually materialized it for this request.
self._resolve_storage_prefetch_tokens(request, insert_result.prefix_len)

self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
del self.ongoing_prefetch[request]
self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
...
logger.info(
    "HiCache prefetch %s req=%s completed=%d matched=%d loaded=%d occupied=%d",
    "dropped" if insert_result.host_insert_dropped else "success",
    request.rid, completed_tokens, insert_result.prefix_len,
    loaded_from_storage, self.cache_controller.prefetch_tokens_occupied,
)
```

“**until admission proves** that L2 -> L1 load-back actually materialized it” 这句注释说的是**指标**的生命周期——`_resolve_storage_prefetch_tokens` 把「可用存储跨度」推迟到准入才结算。但紧接其后的三行把**所有权**的生命周期在落地就结束了：解钉、删除认领、递减计数。

> 指标的生命周期对准了准入，所有权的生命周期对准了完成。这个不一致就是本文要修的东西。

顺带两个现成的可观测点：`insert_result.host_insert_dropped` 让日志已经能区分 `dropped` / `success`；`occupied=%d` 已经在 INFO 级别打印计数器值。

### 2.4 机制二：一个没有条目的账本

`prefetch_tokens_occupied` 是 cache 模式唯一的预取内存治理手段。全仓 30 处引用的分布本身就是诊断：

| 类别 | 处数 | 位置 |
|---|---|---|
| 自增 | **3** | `hiradix_cache.py:2006`（`+= len(prefetch_key)`）；`unified_radix_cache.py:2059`（`+= len(prefetch_key)`）；`unified_radix_cache.py:2801`（`+= alloc_len`，**仅 buffer 模式**） |
| 自减 | **11** | `hiradix_cache.py:547`、`1775`、`1838`、`2216`；`unified_radix_cache.py:2216`、`2323`、`2493`、`2650`；`pipeline.py:1004`、`1280`、`1307` |
| 读取（判据/预算） | 2 | `cache_controller.py:1270`（限流判据，**仅 cache 模式**）；`pipeline.py:522`（`live_cap`） |
| 读取（日志） | 3 | `hiradix_cache.py:1979-1980`；`unified_radix_cache.py:2232`；`pipeline.py:1285` |
| `max(0, …)` 钳制 | **4** | `hiradix_cache.py:548-549`、`1774-1775`；`unified_radix_cache.py:2650-2653`；`storage_attachment.py:401-407` |
| 初始化/重置 | 3 | `cache_controller.py:610`、`807`；`hybrid_cache_controller.py:418` |

**3 增 11 减 4 钳**。一个带身份的账本不可能变负——每条支出都对应一条可寻址的收入，撤销时按身份注销即可。裸整数会在异常清理路径上双减，于是必须在四处独立防御：

```python
                try:
                    cc.prefetch_tokens_occupied -= len(prefetch_key)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass
```

（[hiradix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/hiradix_cache.py#L532-L555) 的强制清理路径，`try/except Exception: pass` 包裹递减与钳制，是整个账本设计最露骨的一处自证。）

**钳制的存在本身证明记账没有身份。** 对照组就在隔壁：buffer 模式的 `_StagedPrefetch` 是一个有身份的容器，其 docstring（[pipeline.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L100-L104)）逐字为 “A completed buffer-mode fetch **parked until prefill admission**: only the op-owned host bounce exists (no device state, nothing in the tree).” 它把「已完成、未准入」显式建模为一个状态，账本挂在容器上（`pipeline.py:1004`、`1280`、`1307` 三处递减都在容器生命周期边界上），因此从不需要 `max(0, …)`。

还有一处更严重的后果：`unified_radix_cache.py:2649` 的断言

```python
        # Every revoke path runs before the bounce alloc, so buffer mode
        # holds no occupancy here; post-alloc aborts go through
        # release_aborted_request instead.
        assert _host_indices is None or self.host_memory_mode != "buffer_only"
```

这条注释的前提是「所有 revoke 路径都发生在 bounce 分配之前」。一旦 cache 模式开始持有 post-land 占用（§5.1 的 R1），该前提失效，必须新增一条 post-land 撤销路径，且要走 `release_host()` 而非仅仅递减整数。这是 R1 必须连带处理的一处，不能漏。

### 2.5 机制三：为 A 授予的动作，就是淘汰 B 的那一行

授予路径内部（[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2738-L2751)）：

```python
    alloc_len = hit_tokens
    host_indices = cc.mem_pool_host.alloc(alloc_len)
    if host_indices is None:
        self.evict_host(alloc_len)
        host_indices = cc.mem_pool_host.alloc(alloc_len)
```

`self.evict_host(alloc_len)` 淘汰的候选来自 `evictable_host_leaves`，其中包含**所有其他请求已落地、尚未准入的预取页**（§2.3 第三阶段）。`hiradix_cache.py:638` 是同型调用；`hiradix_cache.py:883`（`write_backup`，`host_indices is None` 时 `evict_host(len(node.value))`）是第三处。

关键的结构性事实是：`_update_host_leaf_status`（`hiradix_cache.py:1222-1235`）判断一个节点是否可淘汰时，**只看 `evicted` 与 `lock_ref`，完全不看 `host_ref_counter`，也不看是否存在 in-flight 或已落地未消费的预取**。因此一个刚承载了预取页的新生叶子，在被创建的那一刻就进入候选集合。

这解释了为什么并发度越高故障越确定：\( N \) 个请求同时在 \( W \) 内，每次为第 \( i \) 个请求授予都会扫一遍前 \( i-1 \) 个请求的落地页。故障不是概率性的资源竞争，是**授予次数的线性函数**。
### 2.6 机制四：截断被记为成功，而成功不可重试

第二根因（空间不够只预取放得下的部分）的代码路径在同一段里，紧随 §2.5 的 `evict_host` 之后（`unified_radix_cache.py:2742-2751`）：

```python
    if host_indices is None and not buffer_mode:
        # Memory-pressure fallback: a shorter page-aligned prefix.
        # (Cache mode only — buffer mode parks for the full hit.)
        available_size = cc.mem_pool_host.available_size()
        alloc_len = min(hit_tokens, available_size - (available_size % self.page_size))
        if alloc_len >= self.prefetch_threshold:
            host_indices = cc.mem_pool_host.alloc(alloc_len)
```

注释里 “(Cache mode only — buffer mode **parks** for the full hit.)” 已经把两种模式的处理方式对立起来：cache 模式截断，buffer 模式停下等。截断之后的走向（`unified_radix_cache.py:2793-2803`）：

```python
    self._resolve_storage_prefetch_tokens(request, hit_tokens - alloc_len, reason="host_capacity")
    operation.storage_hit_count = alloc_len
    operation.hash_value = operation.hash_value[: alloc_len // self.page_size]
    operation.host_indices = host_indices
    self.ongoing_prefetch[request] = info._replace(host_indices=host_indices)
    if buffer_mode:
        cc.prefetch_tokens_occupied += alloc_len
    cc.prefetch_buffer.put(operation)
    return True
```

`return True` 意味着**成功**。`operation.storage_hit_count` 被就地改写为 `alloc_len`，原始的 `hit_tokens` 只以一个差值形式进了 `_resolve_storage_prefetch_tokens(reason="host_capacity")`，随后在 `t_land` 被结算掉。请求最终带着 `loaded_from_storage = 40K` 走向准入，日志打印 `prefetch success`，指标显示「命中 40K」——**框架认为这是一次成功的预取，只是命中得少**。没有任何计数器记录「本来能命中 128K」。

对照完全失败的分支（`unified_radix_cache.py:2761-2771`）：它会调 `self.storage_prefetch_retries.poll_miss(...)` 注册重试并 `_log_storage_prefetch_deferred(...)`。**一页都没分到会注册重试，分到一部分反而不注册。**

而重试本身被三重封锁：

1. `PrefetchStatus.DONE` 的注释逐字为 “**Never retry**”（[schedule_batch.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/schedule_batch.py#L849-L864)）。截断路径走的是成功完成，状态即 `DONE`。
2. `poll_miss` 的五个触发点全部是「预取从未成功」：`unified:1975-1981`（too_short）、`1982-1986`（rate_limited）、`2765-2767`（host_capacity 完全失败）、`2785-2787`（aux staging 失败）、`2092-2113`（`host_indices is None`）。
3. `StoragePrefetchRetries` 的 docstring（[storage_prefetch.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/storage_prefetch.py#L11-L74)）逐字为 “**New L3 attempts only** (parked IO and retained staging retry at their owner); step deadlines keep TP ranks in lockstep, retries follow queue order, and a request past its re-issue budget is admitted with what the device holds.” 且默认关闭：`enable_hicache_prefetch_retry=False`（[memory.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/arg_groups/fields/memory.py#L177-L185)），作者注释（`scheduler.py:3135-3138`）为 “Default: one-shot judgment. A prefetch skipped by transient limits (in-flight prefetch budget / host memory pressure) is **never retried** for this request.”

唯一的自愈点是 buffer 专属的：`pipeline.py:954-958` `_refetch_staged`，触发条件 `matched_len < f.matched_len`（由 `prepare_staged_prefetch`，L901-911 计算）。cache 模式没有等价物——页已经在树上，被摘除之后连可供比较的对象都不存在了。

理论上还有一处能感知缩水的代码：`_prefetch_after_device_hit_loss`（[scheduler.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/managers/scheduler.py#L3300-L3327)）。它的 L3309 `current_match_len = len(req.prefix_indices) + req.host_hit_length` 有能力发现 host 命中变短（`match_prefix`，`hiradix_cache.py:1910-1941`，其 `while last_node.evicted: host_hit_length += len(last_node.host_value)` 在节点已被 L1411-1412 摘除后自然少算）。但这处能力被三重遮蔽：

- 函数名、docstring、warning 文案**全部只谈 device**（`device_hit_loss`），维护者与读者都不会把它联想到 host 层缩水；
- L3312-3318 被 `max_attempts` 门控（默认 8），超限后 “admitted as is”；
- L3304-3308 被 `buffer_pipeline.has_staged` 短路。

**感知能力存在，被命名、门控与短路三重遮蔽。**

### 2.7 两个根因收敛为同一缺陷

| 维度 | 根因一：预取太早被淘汰 | 根因二：空间不足被截断 |
|---|---|---|
| 发生时刻 | \( t_{\text{land}} \) 之后、\( t_{\text{admit}} \) 之前 | \( t_{\text{alloc}} \) |
| 直接代码 | `evict_host` 摘除 `host_ref_counter == 0` 的新生叶子 | `alloc_len = min(hit_tokens, available_size - …)` |
| 记账后果 | 计数已在 \( t_{\text{land}} \) 递减，淘汰不再回滚 | 计数按 `alloc_len` 记，原始 `hit_tokens` 无处可查 |
| 可观测性 | 日志 `prefetch success`，指标显示 `loaded > 0`，准入时兑现为 0 | 日志 `prefetch success`，指标显示 `loaded = 40K`，无「本应 128K」记录 |
| 可否重试 | 不可（`DONE` = “Never retry”） | 不可（未调 `poll_miss`） |
| 缺失的同一物 | **授予的生命周期** | **授予的生命周期** |

两者不是两个 bug，是同一个缺陷的两个投影：**cache 模式把「预取」建模成一次性的传输任务，而不是一份有开始、有持有期、有释放事件、有失败语义的资源授予。**

限流判据也印证这一点。[cache_controller.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/managers/cache_controller.py#L1255-L1273) `prefetch_rate_limited()`：

```python
    def prefetch_rate_limited(self) -> bool:
        """Rate limit the prefetching operations to avoid overwhelming the storage backend."""
        if self.host_memory_mode == "buffer_only":
            # buffer mode allocates hit-sized, so prefetch_tokens_occupied's
            # requested spans overstate it; read the real pool occupancy.
            ...
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
```

三点值得注意：docstring 说的是保护**存储后端**（不是内存）；buffer 分支的注释**主动承认 cache 计数是坏代理**（“requested spans **overstate** it”）并因此弃用它、改读真实池占用；末尾挂着 `todo`。

> **buffer 模式已经因为这个计数器不可靠而放弃使用它，cache 模式仍在用它做唯一的内存治理。**

`prefetch_tokens_occupied` 的初始化注释（`cache_controller.py:609-610`）为 “tracking the number of tokens **locked** in prefetching, updated by the main scheduler thread”。用词是 `locked`，实现是 `int`。

### 2.8 免疫性对照：buffer_only 为什么不受影响，以及为什么不能直接切过去

| 维度 | cache 模式 | buffer_only 模式 |
|---|---|---|
| host 层角色 | 持久 KV tier（L2），承载跨请求复用 | GPU 与存储后端之间的瞬态 staging buffer |
| 树上是否有 `host_value` | 有 | **无**，`evict_host` 直接 `return 0`（`unified_radix_cache.py:1311-1314`） |
| 授予记账语义 | `reserve` 请求跨度，\( t_{\text{issue}} \) 起、\( t_{\text{land}} \) 止 | `grant` 实际分配，\( t_{\text{alloc}} \) 起、**准入**止（`pipeline.py:1151`、`1280`、`1307`） |
| 空间不足时的行为 | 截断到放得下的前缀，记为成功 | **park**，`return False`，保留完整命中，重试不耗预算 |
| 「已完成未准入」状态 | 不存在 | 一等公民 `_StagedPrefetch` |
| 计数钳制 `max(0, …)` | 需要，4 处 | 不需要 |
| 淘汰对抗 | 存在（§2.5） | 不存在（无可淘汰对象） |
| 自愈重试 | 无 | `_refetch_staged`（`pipeline.py:954-958`） |
| 限流依据 | `prefetch_tokens_occupied` | 真实池占用（已弃用前者） |
| 跨请求 L2 复用 | **有** | **无** |

buffer_only 对本文两个根因**双重免疫**：既没有可被淘汰的树，记账生命周期又完整覆盖到准入。但它免疫的代价是**放弃了 L2 本身**——host 层不再是缓存而只是搬运缓冲，跨请求的 host 层复用能力消失。

对「集群范围共享池 + 跨实例复用」的目标（本文的立项约束），这是不可接受的：跨实例复用的价值恰恰依赖 host 层作为共享池的本地驻留面。因此正确的结论不是「切到 buffer_only」，而是「**把 buffer_only 已经验证过的授予生命周期，移植到仍然保留 L2 的 cache 模式**」。

这也直接回答了审稿人必问的一题：为什么不用现成的 buffer 模式规避问题。答案分两层——语义层（放弃 L2 等于放弃跨实例复用）与工程层（buffer_only 有四条硬前置，见下）。

四条硬前置（[hicache_hook.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/arg_groups/hicache_hook.py#L185-L229)，含强制校验）：

1. 必须配 `--hicache-storage-backend`（L210-215）；
2. 不能配 `--hicache-write-policy write_back`（L216-221）；
3. **不能跑在 decode 实例上**（L222-229），理由逐字为 “the decode-side prefetch and offload paths bypass the buffer-mode pipeline, fetching without its prefix context and never consuming its staged holds. Prefill instances share the standard scheduler path and are supported.”；
4. 必须使用 `UnifiedRadixCache`（`registry.py:244-255`）且只支持 FULL/SWA 统一树（`unified_radix_cache.py:424-429`）。

另：`hicache_ratio` 的默认值随模式改变，`1.2 if buffer_only else 2.0`（`hicache_hook.py:79-86`）。

> **两个开关的口径澄清（极易混淆，特此固定）**：`--hicache-host-memory-mode` 只有 `cache` / `buffer_only` 两个合法取值，且带强制校验（[hicache_hook.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/arg_groups/hicache_hook.py#L185-L229)）；官方 help 逐字为 “Whether host memory is a persistent HiCache tier (cache) or a transient staging buffer between GPU and the storage backend (buffer_only). buffer_only requires --hicache-storage-backend.”。`--hicache-write-policy` 是**完全独立的另一个开关**，三值，`choices=["write_back", "write_through", "write_through_selective"]`，默认 `"write_through"`（[memory.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/arg_groups/fields/memory.py#L102-L124)）。二者不可混同：前者决定 host 层**是不是一个缓存**，后者决定**什么时候往 L3 写**。本文的全部改动都在前者的 `cache` 取值路径上，对后者的三个取值均不做任何假设，也不依赖其默认值。


---

## 3. 新颖性边界

### 3.1 问题定位：本文修的是预取生命周期的哪一段

L3→L2→L1 的一次预取可以切成四段。本文只修第三段；下表同时说明其余三段为什么不在范围内，以及各段已有哪些工作：

| 段 | 区间 | 该段的决策内容 | 已有工作的覆盖情况 | 本文是否触碰 |
|---|---|---|---|---|
| **① 取什么** | 命中探测 → 决定预取跨度 | L3 上哪些前缀值得取、取多宽 | [cite:3] ECHO（稀疏注意力的 lossless intra/inter-query 预取）、[cite:4] OasisKV（decode 侧 lookahead 稀疏预取）、[cite:19] RoPE-Aware Prefetching（按注意力类型自适应窗口宽度，miss 率降 25%） | **不触碰**。本文假定上游命中探测与跨度选择已经正确，只处理它交下来的那个跨度 |
| **② 什么时候开始取** | \( t_{\text{issue}} \) 的选择 | 提前多久发起，才既不白占快层容量又赶得上消费 | [cite:12] TempoKV（laxity 非正才承诺，protected fast-tier byte-time 降 63–91%）、[cite:15] TierKV（\( T_{\mathrm{transfer}} \le \sum_j T_{\mathrm{iter}}^{(j)} \) 的 lookahead 窗口）、[cite:2] CXL Memory Rack（每请求约 128 MB 的预取窗口上限） | **借用而不主张**。§5.2.3 的 R3-b 门控用的正是 TempoKV 式 laxity 判据，但它在本文中是 R1 的成本对冲项（§5.2.2），不是新颖性所在 |
| **③ 取到之后归谁、活多久** | \( [t_{\text{land}}, t_{\text{admit}}) \) | 页已入树、请求仍在队列里，这段间隔内页受不受保护、已授予的容量记账何时释放 | **本文所指的空白**。[cite:1]、[cite:2]、[cite:15] 的 pin 在传输完成即释放，而在其模型里传输完成就等于已被消费，故这段间隔恒为零；[cite:5] Talaria 与 [cite:14] Continuum 的 reservation 覆盖「已消费、可能回访」，方向相反；[cite:16] 与 [cite:15] 的 reserved area 是静态池划分，与请求生命周期无关 | **R1 / R2 / R3-a / R4 的全部内容** |
| **④ 块该放在哪一层** | 页在 L1/L2/L3 之间的静态归属 | 驻留层级的划分策略 | [cite:20] *Where Should the KV Cache Live?*（三层 placement policy）、[cite:18] HillInfer（按 recency 静态锁定最近 \( \alpha \cdot N \) 步于 Hot pool） | **不触碰**。本文在既有三层结构内工作，不重划层级，也不从池中永久切走一块 |

三点需要明确：

**其一，本文修的不是「预取算法不够聪明」，而是「预取结果没有主人」。** 段 ①② 的全部工作都在优化「怎样让页在需要时已经在快层」；本文处理的是页**已经到了快层、却在被需要之前被拿走**。这是两类不同的失效，其可观测形状也不同：前者表现为命中率整体偏低，后者表现为命中率**在高并发下坍塌而在低并发下正常**——正是 §2.1 所述现象与 [cite:22] 官方基准 Round 40–55 处 L1/L1+L2 双双归零、L3 仍维持约 98% 的断崖形状。段 ① 或段 ② 的任何改进都无法消除这个形状，因为它们都不改变页落地之后的命运。

**其二，本文的 pin 对象在树上是新生叶子，不是内部祖先。** `_insert_helper_host` 创建的新节点 `value=None`（故 `evicted == True`），并被 `_update_host_leaf_status` 立即加入 `evictable_host_leaves`（[hiradix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/hiradix_cache.py#L2009-L2050)，其中 L2043 调用该状态更新）——它是叶子。这条结构性事实有两个后果：

- 本文的保护判据「该叶承载一个已落地但尚未被消费的授予」，与任何基于「结构上不可丢的祖先」的保护判据在树上**打不到同一个对象**。因此本文与祖先保护类工作（如 [cite:13] PEEK 对 `ancestors(t_b)` 的打分）在机制上可叠加而不冲突，两者的作用窗口首尾相接（§3.4.2）。
- 被保护的页是**刚刚从 L3 付出真实 IO 换来的**，丢失它的代价是全部已投入的传输成本外加一次完整重算，而不是一次廉价的重算。这是本文强调「硬保护」而非「软排序」的经济学理由（§3.3 F-f）。

**其三，范围界定。** 本文不改动：L3 侧的池协议与跨实例元数据（[cite:8] Mooncake 的职责域）、`evict_host` 的候选排序策略（只在候选遍历处增加一条硬跳过，见 §2.0.3）、请求的准入顺序与调度策略（§5.2.4 论证 R3-b 与「不改变请求调度」兼容）、以及 L1 device 层的任何逻辑。全部改动集中在 L2 host tier 的**授予记账**与**钉住时刻**两处，规模与依赖关系见 §5.0。

### 3.2 与最新文献对照

| cite | 工作 | 其机制 | 与本文的差异 |
|---|---|---|---|
| [cite:1] | LM-CXD / *Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management*，arXiv 2609.26828 | CAP：chunk 按窗口 “prefetched, **pinned**, and DMA'd”；“the next window starts prefetch only when another window is **unpinned**” | pin 在**单请求内部**为 DMA/计算重叠服务；释放点是 DMA 完成而非准入；无跨请求排序、无淘汰对抗 |
| [cite:2] | *A CXL Memory Rack for Multi-Turn LLM Serving*，arXiv 2607.18141v3 | 每请求预取窗口上限（“up to approximately 128 MB of prefetched KV data per request”），按 KV-block 大小调窗口 | **单请求内的固定预算**，是节流（throttling）而非所有权（ownership） |
| [cite:3] | ECHO: *Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs*，OSDI'26 | 稀疏注意力的 lossless intra-query / inter-query 预取 | 解决「**取什么**」，本文解决「**归谁、活多久**」 |
| [cite:4] | OasisKV: *Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching*，arXiv 2608.08097 | decode 侧 lookahead 稀疏预取，支持 PD 分离与多级 KV 存储 | 稀疏性 + decode 侧；本文覆盖 prefill 与非 PD 场景，且与稀疏性正交 |
| [cite:5] | Talaria: *Session-Aware Serverless Serving of Hundred-Billion-Parameter LLMs*，arXiv 2607.17181 | **Soft reservation** for returning sessions：router 记 session id / model / instance / reusable-prefix handle / \( t_{\exp} = t_{\text{done}} + \tau \)（\( \tau = 1\text{s} \)，近 p90 tool gap） | **集群 router 层**的会话亲和；按**时间过期**释放。本文是**单机 host tier 层**，按**准入事件**释放，且发放按等待队列序 |
| [cite:6] | Cascade: *Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving*，arXiv 2608.06557 | 跨类结构公平调度；含 64k context 深 tier KV 检索对 TTFT 的影响 | 调度层的类间公平；不涉及 tier 内页所有权 |
| [cite:7] | ExPAND: *CXL Topology-Aware and Expander-Driven Prefetching*，arXiv 2505.18577 | expander 驱动的 CXL LLC 预取 | 硬件层、LLC 粒度 |
| [cite:8] | Mooncake: *A KVCache-centric Disaggregated Architecture for LLM Serving*，arXiv:2407.00079 | L3 共享池背景 | 本文在其 L2/L3 边界内工作，不改动其池协议 |
| [cite:9] | arXiv 2606.15555（Ao et al.，极限环，吞吐损失达 50%） | 振荡型故障 | 本文故障**非振荡**：单次 `evict_host` 即单次损失，负载下降立即消失（§3.3 F-c） |
| [cite:10] | arXiv 2607.02640 Metronome（metastable + silent，14/20 vs 0/20） | 亚稳态故障 | 本文**非亚稳态**：无自增强回路（§3.3 F-d） |
| [cite:11] | HotStorage'26 “To Restore or To Recompute” | 恢复 vs 重算的代价权衡 | 对偶定位：本文不改代价函数，改**授予是否存活到能被代价函数评估的那一刻** |
| **[cite:12]** | **TempoKV**: *Timing-Aware KV Staging Commitment for SSD-Backed Inference*，arXiv 2609.35065v1，Samsung Semiconductor，2026-09-28 | **timing-aware resource-commitment layer**，“separates early knowledge of reuse from the acquisition of staging resources”；命中记为 **metadata-only claim**；当 \( \widehat{\mathrm{TTU}} - \widehat{\mathrm{TTR}} \le 0 \)（laxity 非正）时才承诺；承诺需在 protected-capacity budget 内容纳 claim 的 **full KV footprint**，否则留在 Planned | **最强撞题，但正交**：TempoKV 修 \( t_{\text{commit}} \) 太早，本文修 \( t_{\text{release}} \) 太早；TempoKV 在 vLLM+LMCache / SSD-backed CXL Type-3，本文在 SGLang host tier；TempoKV 有 protected capacity、无淘汰对抗的**授予身份**，本文相反。详见 §3.4.1 |
| **[cite:13]** | **PEEK**: *Proactive prEfix matChing and sCheduling for LLM serving*，arXiv 2607.02525（v1 2026-05-10 / v2 2026-09-12），26 pp.，**在 SGLang 与 vLLM 上真实实现** | 在 pending queue 上维护**增量 radix tree**；**dual-walk** 为每个等待请求求最长前缀匹配；**cluster pioneers first** 准入使兄弟继承新缓存前缀；“a co-designed **eviction hook protects blocks ancestral to queued demand**”，打分 \( \mathtt{evict\_score}(b) = \max_{v \in \mathtt{ancestors}(t_b)} \mathtt{pending\_count}(v) \cdot \mathtt{depth}(v) \)；multi-lane stride scheduler 限界饥饿 | **系统级最强竞争者，但机制强度不同**：PEEK 是**软保护**（改候选排序），本文是**硬保护**（`host_ref_counter > 0: continue` 硬跳过）；PEEK 保护树上**已存在**的祖先块，本文保护 L3 刚拉来的**新生叶子**；PEEK **不解决截断**。详见 §3.4.2 |
| **[cite:14]** | **Continuum**: *Efficient Execution of Agentic Workflows with Dynamic KV Cache Retention*，arXiv 2511.02230 / UCB/EECS-2026-234，UC Berkeley + Stanford + Tensormesh + Tsinghua | 按 reload cost 与 eviction 引起的 queueing delay 算出 **TTL**，把 KV **pin 在 GPU 内存**；TTL 到期自动可淘汰；配合 **program-level FCFS**；实现在 vLLM；avg job completion time 提升 over 8× | 三层全不同：Continuum pin **已完成请求**的 KV（end-of-turn）在 **GPU**，按**时间过期**释放；本文 pin **尚未准入请求**的预取页在 **host**，按**准入事件**释放。Continuum 的核心洞察（“even when CPU offloading is enabled... eviction causes **per-turn queueing delay**”）与本文同源，可作为动机侧的强支撑引用 |
| [cite:15] | **TierKV**: *Prefetch-Aware Three-Tier KV Cache Management*，idea2paper.org 预印本（匿名，**离散事件模拟器评估，非真实实现**） | continuous batching scheduler 的 **deterministic lookahead window**；\( T_{\mathrm{transfer}} \le \sum_j T_{\mathrm{iter}}^{(j)} \)（EMA-16）；Figure 1 Zone A 含**静态预留 staging area**；§3.3 “Blocks belonging to in-flight requests are **pinned and cannot be evicted**”；§3.4 bandwidth-aware admission control，高水位 80% 时 defer 低优先级预取，\( priority(r) = k^{-1} \) | TierKV pin 的是 **active（正在 decode）**块，**不 pin 已预取未准入的块**——正是本文缺口；其预留是**静态区域划分**而非**按请求具名授予**；模拟器 + 匿名，引用价值低，宜作同期预印本并列 |
| [cite:16] | *Mitigating KV Cache Competition to Enhance User Experience in LLM Inference*，arXiv 2503.13773 | §4 `global_reservation`：“a **reserved portion of the KVC**”；`TokenBlockManager` 加 `track_reserved_blocks`，“reserved blocks are **separate from the general allocation pool**”；`scheduler.py` 加 `allocate_reserved_kvc` | 「reservation」一词最接近的先例，但是**静态划分预留区**（池的一部分永久隔离），本文是**按请求、按时序、可撤销的具名授予**（池不分隔，只标记归属与释放点） |
| [cite:17] | **TokenCake**，arXiv 2510.18586v5 | §4.3 Predictive Upload：\( P_{upload} = I + U \)（importance + urgency），“the scheduler computes an **upload budget that protects critical waiting requests**” | 方向是**上行**（KV 上传到池）的预算，本文是**下行**（池→host→device）的授予；“protects critical waiting requests” 的意图相同，但 TokenCake 的 protection 是**上传带宽配额**，不是**已落地页的淘汰豁免** |
| [cite:18] | **HillInfer**，arXiv 2602.18750 | Bidirectional KV Cache Pools；**Temporal-Aware Pinning**：“we **statically lock** the KV cache of the most recent \( \alpha \cdot N \) steps exclusively in the Hot pool”；edge + SmartSSD | 「temporal-aware pinning」名字最近，但机制是**按 recency 静态锁定固定步数**，与请求生命周期无关；本文按**单个请求的准入事件**动态授予与释放 |
| [cite:19] | *Predictive Multi-Tier Memory Management for Long-Context LLM Inference*，arXiv 2604.26968 | §III-E **RoPE-Aware Prefetching**：预取 positions \([n, n+w]\)，\( w \) 按层自适应（local-attention 窄、global-attention 宽）；配合 Bayesian reuse prediction，“reduces cache miss rates by 25%” | 解决「**预取哪些位置、预取多宽**」，本文解决「**预取到的页归谁、活多久**」；两者可叠加（RoPE-aware 决定内容，授予决定存活） |
| [cite:20] | *Where Should the KV Cache Live?*，arXiv 2609.16215 | §III 三层 placement policy；“A prefetch engine moves blocks up the hierarchy ahead of the decode step that needs them, hiding transfer latency” | 关注**块该放在哪一层**（placement），本文关注**已经放对了层的块如何活到被用**（retention of in-flight staging）；其 prefetch engine 的描述正是本文所诊断的、缺少所有权语义的那类引擎 |
| [cite:21] | *HBF Sucks?*，arXiv 2608.11668 | High-Bandwidth Flash 全栈特性刻画 | 硬件层特性研究；可为 \( \widehat{\mathrm{TTR}} \) 的存储侧估计提供实测参数，与本文互补而不重叠 |

**空白点陈述**：截至本次复核，**没有任何已发表工作把 L3→L2 预取建模为「具名的、以准入为释放点的、带反死锁上界的 tier 内授予」，并同时把「压力下截断」处理为一等可观测失败**。需要澄清「按等待队列序发放」在本文中的确切地位：它是**稀缺容量的仲裁规则**（R3-a，决定谁先拿到页），而**不是发放时机的判据**（后者由 R3-b 的 laxity 门控负责，决定何时该去拿）。这一分工依 [cite:12] 的实证作出——TempoKV 的 Figure 2 标题 “Queue rank is not a staging deadline.”，配合其 “Despite identical rank, margins range from 2.67 s early to 6.02 s late” 的测量，共同表明队列序只给出**次序**而不给出**距检索的时间**（详见 §3.4.1）。

三条最接近的先例各自缺什么：

- [cite:12] TempoKV **有** protected capacity 与 timing 判据，**缺** capacity 不足时的仲裁策略——其摘要与 §3.2 均只说 “the claim remains in its initial Planned state”，**从未定义多个 Planned claim 之间谁先获得稀缺容量**。本文的队列序仲裁正好补这个洞。
- [cite:13] PEEK **有** 「保护排队请求所需块」的淘汰钩子，**缺**硬保护语义与截断处理——打分只改排序，当必须淘汰的页数超过非保护页总数时，保护失效；且打分不影响分配成功与否，故其机制在原理上无法消除根因二。
- [cite:1]、[cite:2]、[cite:15] 都使用 pin / window / staging 词汇，但它们的 pin 从不覆盖「**已落地、未消费**」这个状态——在 DMA 重叠模型（[cite:1]）与 lookahead 模型（[cite:15]）里该状态根本不存在，因为窗口的 pin 在传输完成时即释放，而传输完成就意味着已被消费。本文的 \( W \) 之所以存在，恰恰因为**L3→L2 的传输完成（\( t_{\text{land}} \)）与 L2→L1 的消费（\( t_{\text{admit}} \)）之间隔着一次调度准入**，这段间隔在单请求 DMA 模型里为零，在多请求排队模型里无上界。

[cite:5] Talaria 与 [cite:14] Continuum 的 reservation / pin 都覆盖「**已消费、可能回访**」，方向与本文相反（本文覆盖「**已就位、尚未消费**」）。[cite:16] 与 [cite:15] 的 reserved area 是静态池划分，本文是按请求授予——前者会永久损失一部分池容量，后者只在窗口内临时占用且带 \( G_{\max} \) 上界。

### 3.3 可证伪判别

本文的故障诊断必须与相邻故障类型区分开。六条判别全部可在实验台上直接测量：

- **F-a（与「L3 无命中」区分）**：若 `storage_hit_count > 0` 且日志出现 `prefetch success`，但准入时 `load_back` 兑现为 0，则不是命中缺失。测量点：`prefetch_loaded_tokens_by_reqid`（`hiradix_cache.py:1842` / `unified_radix_cache.py:2218`）与准入时实际 load-back 的存储来源 token 数之差。
- **F-b（与「存储后端慢」区分）**：若 L3 读带宽在压测初期非零、随后归零，而 `storage_hit_count` 仍非零，则不是后端吞吐瓶颈。测量点：后端读字节数 / 秒，与 `_prefetch_outcome_stats` 的分布。
- **F-c（与 [cite:9] 的极限环区分）**：负载阶跃下降后，命中率应**立即**恢复到单并发水平，无持续振荡。测量点：并发从 256 突降到 1 后的逐轮命中率序列。
- **F-d（与 [cite:10] 的亚稳态区分）**：不存在自增强回路——淘汰由外部授予驱动，授予速率受 `prefetch_rate_limited()` 限制，损失不会导致更多损失。测量点：撤销负载后系统是否自行恢复（应恢复）；以及固定负载下命中率方差是否随时间发散（应收敛）。
- **F-e（与「调 `prefetch_stop_policy` 即可解决」区分）** — **最重要的一条**。断言：`--hicache-storage-prefetch-policy` 的三个取值 `best_effort | wait_complete | timeout`（默认 `timeout`，`memory.py:166-172`）**无一能保护 \( W \)**。证明在 §4.5.1，核心是 `check_prefetch_progress` 的早退：`if request not in self.ongoing_prefetch: return True`（`unified_radix_cache.py:2094-2095`），而 cache 完成路径在 \( t_{\text{land}} \) 就 `del self.ongoing_prefetch[request]`（L2215），门控随之立即解除。**测量点**：把 A7 臂（`wait_complete`）在高并发下跑一遍，若 \( \rho \) 仍系统性 \( < 1 \) 则 F-e 成立、本文诊断未被既有开关覆盖；若 A7 使 \( \rho \equiv 1 \)，**本文主张被证伪**，应放弃投稿。这一条把「审稿人会说：既有开关试过吗」变成一个可判定的实验，且判据对本文不利时同样有效。
- **F-f（与 [cite:13] PEEK 的软保护区分）**。断言：软保护（改淘汰打分）在**极端压力**下必然失效，硬保护（引用计数）不会。测量点：把 A8 臂（PEEK 式打分）的 \( \rho \) 对 host 池压力（`--hicache-ratio` 从高到低扫）作图。预期 A8 在低压下 \( \rho \to 1 \)、在高压下 \( \rho \) 跌破 1，而 A2（硬保护）在整个扫描区间维持 \( \rho \equiv 1 \) 直到 \( G_{\max} \) 绑定。**两臂曲线的分叉点即软/硬保护的相变点**，是本文相对 [cite:13] 的定量增量。

F-c 与 F-d 合起来把本文的故障归入一个明确的类型：**窗口暴露（window exposure）**，既非振荡亦非亚稳态。判别用同一个负载阶跃实验：并发从高压突降到 1 之后，命中率应**立即**恢复且方差收敛。这条类型学区分是必要的而非装饰性的——振荡与亚稳态故障在卸载负载后**不自愈**，若本文故障属于那两类，则 §5 的修复路线（消除窗口）在原理上不可能奏效，必须转而设计阻尼或隔离机制。换言之，F-c/F-d 是「本文方案是否对症」的前置检验。

F-e 与 F-f 是本文的两条**新颖性防线**：前者把「既有配置就能修」这个最可能的审稿反驳变成一个可判定的实验，并附带一条对本文不利的自我证伪路径（若 A7 臂使 \( \rho \equiv 1 \)，主张作废）；后者把「[cite:13] 已经做了」这个反驳变成一条可测的相变曲线（A8 与 A2 的 \( \rho \)–压力扫描分叉点）。两条防线共同的特征是：**判据在结论对本文不利时同样有效**。

### 3.4 三方深度对照

三篇工作在标题、词汇、甚至动机上都与本文重叠，必须逐条正面处理，否则任何一位审稿人指出其中一篇都足以否定新颖性。

#### 3.4.1 TempoKV：同一问题直觉，相反的时间端

TempoKV 的摘要第一句就点出本文的直觉：“Demand staging exposes SSD latency, whereas **immediate staging can reserve fast-tier capacity long before retrieval begins**”[cite:12]。它把这一点做成了一个 timing-aware commitment layer，术语定义（§2.2 *Memory Exposure Is Not Readiness*）与本文高度同源：

> **stage-ready** = “once every required KV object is present in the fast tier **and protected against eviction**”
> **staging commitment** = “the point at which staging is authorized and **sufficient fast-tier capacity is reserved to keep the matched KV protected through retrieval**”

“protected through retrieval” 与本文的 \( \text{release}(g_r) = \text{admit}(r) \) 是**同一个要求**。差异全在实现路径：

| 维度 | TempoKV [cite:12] | 本文 |
|---|---|---|
| 被修的时间端 | \( t_{\text{commit}} \) **太早** → 早承诺导致 protected byte-time 浪费 | \( t_{\text{release}} \) **太早** → 早释放导致页在消费前被淘汰 |
| 失效表现 | 快层容量被无效占用（**性能税**） | 已投入的 IO 全部作废（**正确性/命中率损失**） |
| 系统 | vLLM + LMCache，SSD-backed CXL Type-3 | SGLang HiCache，host DRAM tier + Mooncake L3 |
| 「保护」的来源 | 新造 protected-capacity budget，由 storage-side provider 复核 | 复用既有 `host_ref_counter` / `protect_host()` 原语（§4.3） |
| 「不提前占」的来源 | metadata-only claim：commitment 前 “neither initiates staging I/O nor reserves protected fast-tier capacity” | cache 模式**已有**账面 reserve（`prefetch_tokens_occupied += len(prefetch_key)`，L2059）；本文不改 reserve，改的是 **grant 的生命周期** |
| 释放判据 | TTL 无；由 retrieval 开始触发 | 准入事件触发 |
| 是否改调度 | 明确 “**without changing request scheduling**” | 同样不改请求的准入顺序；只改**授予**的仲裁（R3-a）与发放时机（R3-b） |
| 容量不足时 | “the claim remains in its initial Planned state”——**未定义 Planned claim 之间的仲裁** | 队列序 park + LIFO 撤销 + `defer_rounds ≤ 32` 上界（§5.4） |

**关键的正交性论证**：TempoKV 的 metadata-only claim 在 SGLang 的 cache 模式里**已经存在**——它就是 `prefetch_tokens_occupied` 的账面 reserve，L2056-2059 的注释逐字为 “Cache mode **reserves** the requested span up front; buffer mode **grants** occupancy later at hit-alloc time, sized to the hit”。TempoKV 缺的、也是 SGLang 缺的，是它自己命名的另一半：**protected capacity**。SGLang 的 `prefetch_capacity_limit = int(0.5 * mem_pool_host.size)` 是一个**限流阈值**，不是**保护预算**——它限制「同时在飞的预取有多少」，但不保证「已落地的页不被淘汰」。所以：

> TempoKV 证明了「过早承诺要付 protected byte-time 的税」；本文证明了「过早释放要付**全部已投入 IO** 的税」。两个税同时存在，两个修复互不蕴含，可叠加。

**必须采纳的批评**。TempoKV §2.3 用 Figure 2（标题 “**Queue rank is not a staging deadline.**”）给出实证：FCFS + 单活跃序列，每个 target 是运行中前驱之后唯一的等待者，改变前驱剩余服务时间（0.5 s / 4 s）、target footprint（8 / 12 GiB）、staging backlog（无 / 4 个已承诺的 8-GiB 操作），结果 “Despite identical rank, margins range from **2.67 s early to 6.02 s late**”；4 s 前驱 + backlog 使两种尺寸都从 early 翻转为 late。正文结论：“Queue rank, for example, indicates **order rather than time until retrieval**.”

若把队列序当作**发放时机**判据，这条批评直接命中本文。本文的处理是把队列序的职责收窄：

- **队列序只做一件事：仲裁稀缺容量**（谁先拿到 host 页）。这仍然是正确的，因为「先入队者先获得」保证了队头前进性与公平性，而 TempoKV 并未否定公平性规则，它否定的是**用 rank 推断 deadline**。
- **发放时机改由 laxity 门控**：借用 TempoKV 的 \( \widehat{L}_{c}(t) = \widehat{\mathrm{TTU}}_{c}(t) - \widehat{\mathrm{TTR}}_{c}(t) \)，仅当 \( \widehat{L}_c(t) \le 0 \)（just-in-time 或已 late）时才把授予从 PARKED 提升为 GRANTED。§5.2 给出 SGLang 侧的可实现估计式。

 TempoKV 的 \( \widehat{\mathrm{TTU}} \) 定义（“time until **fast-tier→GPU retrieval begins**”，非 time-to-first-token；基于 active batch + per-request execution progress + scheduling order；输出长度用「对已生成 \( n \) token 的运行请求，取长度 \( N > n \) 的已完成输出中 \( N-n \) 的均值」估计；改编自 QLM 的 work-based waiting-time estimation 与 Past-Future 的 output-length conditioning）在 SGLang 里**逐项都有对应物**，这是 §5.2 可实现性的依据。

 TempoKV 还留下一个本文可直接引用的空白：**它没有说明 protected-capacity 不足时多个 Planned claim 如何竞争**。本文的队列序仲裁 + LIFO 撤销 + defer 上界，恰好是对该空白的一种具体回答；反过来，本文的 \( G_{\max} \)（§5.3）也是对 TempoKV protected-capacity budget 的一种「以准入余量为约束」的具体化。**两篇互为对方的缺失章节**，这是最强的「正交而非重复」论证形式。

其报告量级亦可作为本文实验的预期参考：protected fast-tier byte-time per request 降 **63–91%**；fast-tier 容量 100→25 GiB 时吞吐与 p95 TTFT 几乎不变；相对未改的 LMCache Device-DAX L1，p95 TTFT 降 **48.0%**、吞吐升 **27.8%**[cite:12]。

#### 3.4.2 PEEK：同一系统，软保护 vs 硬保护

PEEK 是本次复核中**唯一在 SGLang 上真实实现、且明确以「保护排队请求所需的块」为目标**的工作，因此是最高优先级的竞争者。其 eviction hook 打分式：

\[
\mathtt{evict\_score}(b) = \max_{v \in \mathtt{ancestors}(t_b)} \mathtt{pending\_count}(v) \cdot \mathtt{depth}(v)
\]

四个维度的区分：

| 维度 | PEEK [cite:13] | 本文 |
|---|---|---|
| 保护强度 | **软**：改 `evict_score`，只是候选排序偏好。当必须淘汰的页数 > 非保护页总数时，保护破裂 | **硬**：`if x.host_ref_counter > 0: continue`（`hiradix_cache.py:1403-1404`），淘汰逻辑**硬跳过**，与需淘汰量无关 |
| 保护对象 | `ancestors(t_b)`——树上**已存在**的祖先块 | `_insert_helper_host` 创建的**新生叶子**（承载刚落地页）。PEEK 的 pending tree 里可能**根本没有对应节点**，因为该前缀此前只存在于 L3 |
| 额外数据结构 | 需维护 pending queue 上的**增量 radix tree** + **dual-walk** 匹配 | 复用既有 `ongoing_prefetch` 字典（§4.2），只把 `pop` 改 `transfer` |
| 截断（根因二） | **无法解决**：打分不影响 `mem_pool_host.alloc` 的成功与否，分配失败仍走截断分支 | R2/R4 直接处理：`truncated` 标志 + `requested_hit_tokens` + TAIL 回队 |

第四条是最干净的分界：**PEEK 的机制在原理上不能消除根因二**，因为淘汰打分与分配失败是两个正交的因果链。这意味着本文的 R2/R4 相对 [cite:13] 是严格增量，且 B5 消融（只加标志不改返回值）能定量给出这条增量的价值。

第二维度的区分同样重要：PEEK 保护的是**共享前缀的祖先**，本文保护的是**独占的新叶**。在多轮对话里，第二轮请求的前缀在 L1/L2 已被淘汰（§2.1 官方曲线的 Round 40–55 情形），此时树上的祖先节点本身已 `evicted`，PEEK 的 `ancestors(t_b)` 要么为空、要么指向已无 `host_value` 的节点——而本文要保护的页此刻正在从 L3 传输上来、尚未在树上有任何节点。**PEEK 的窗口在本文的窗口之前，两者串接而非重叠。**

PEEK 的 multi-lane stride scheduler（限界饥饿）与本文 §5.4 的反饥饿上界呼应，可互引；其报告量级（SGLang 上 up to 3.0× cache hit、7.9× TTFT、6.7× E2E、3.6× throughput；vLLM 上 2.6× / 7.1× / 5.5× / 4.5×；无可利用前缀结构时 match baselines within noise；“Wins hold as KV-cache pressure and inference parallelism scale”）[cite:13] 说明该方向在 SGLang 上有可观收益空间，也说明 A8 臂是一条**强**对照而非稻草人。

#### 3.4.3 Continuum 与 TierKV：同族但不同层

**Continuum**[cite:14] 的 Table 1 用三维度（Retains KV Cache / Includes Per-Turn Queueing Delay / Bounds Retention Time）把 vLLM、Autellix、Pie、InferCept 全部标为至少一项 ✗，自己三项 ✓。本文可以用**同构的三维度表**给自己定位，且答案对本文有利：

| 维度 | 本文 |
|---|---|
| Retains KV Cache（保留已就位的 KV） | ✓（R1：`protect_host()` 提前到 \( t_{\text{land}} \)） |
| Includes Per-Turn Queueing Delay（建模排队延迟） | ✓（§5.2 的 \( \widehat{\mathrm{TTU}} \) 直接来自 waiting_queue 位置与执行进度） |
| Bounds Retention Time（限界保留时长） | ✓（\( G_{\max} \) 空间上界 + `defer_rounds ≤ 32` 时间上界 + LIFO 撤销） |

差异仍在三处：pin 的**对象状态**（Continuum：已完成请求的 end-of-turn KV；本文：尚未准入请求的预取页）、pin 的**层级**（GPU vs host）、释放的**判据类型**（TTL 时间过期 vs 准入事件驱动）。Continuum 明确说明 TTL 的必要性是 “prevent infinitely long waiting... prevent severe memory pressure or **deadlocks**”——本文用 \( G_{\max} \) 的**空间**上界达成同一目的，且额外给出 C3 断言（§5.7）作为运行时检查。Continuum 对 InferCept 的批评（“makes its KV preserve decision based **solely on the reload cost**... does not model the per-turn queueing delay”、“their preserve operation is **fixed** and could not adapt to tool use in real time”）可以原样转用于本文对 [cite:18] HillInfer 静态锁定的批评。

Continuum 报告 avg job completion time 提升 over 8×、delay 降 1.12–3.66×、吞吐升 1.10–3.22×，Tensormesh 内部真实 SWE-agent 降 delay up to 8.18×，评测覆盖 SWE-Bench / BFCL / OpenHand 与 Llama-3.1 8B/70B、Gemma-3 12B、GLM-4.5 355B[cite:14]。**其实现在 vLLM 且只 pin GPU 层**，故与本文无代码冲突，可作为 discussion 中「同一洞察在不同层、不同释放判据下的两次独立验证」。

**TierKV**[cite:15] 的 §3.3 明写 “Blocks belonging to in-flight requests are **pinned and cannot be evicted**”、“when T0 fills, the least recently accessed **non-pinned** block is demoted to T1”，这与本文的机制形式完全一致——但**对象是 active（正在 decode）的块**，即本文四阶段表里的第四阶段 \( [t_{\text{admit}}, \dots) \)，而该阶段在 SGLang 里**今天就已经被保护**（`hiradix_cache.py:1455-1456`）。TierKV 的 Figure 1 Zone A 含 “a **reserved prefetch staging area** (dashed) absorbs incoming blocks”，是**静态预留区**，对应本文 §3.2 中 [cite:16] 那一类，而非按请求授予。其 §3.4 bandwidth-aware admission control（高水位 80% 时 defer 低优先级预取，\( priority(r) = k^{-1} \)，\( k \) 为距调度的迭代数）与本文 R3-a 的 park 语义相近，且其 \( k^{-1} \) 优先级**恰好是 [cite:12] 所批评的「用 rank/迭代数推断紧迫性」**——这为 §5.2 的 R3-b 采用 laxity 而非纯序号提供了第二个独立依据。

TierKV 的评估是**离散事件模拟器**且预印本匿名，故本文对其定位是「同期、同直觉、不同对象层级、无真实实现」，在 related work 中并列一段即可，不构成新颖性威胁。

---

## 4. 代码支点

五个支点决定本方案不是新造机制，而是把仓库里已存在、已被验证、但只在 buffer 分支启用的机制接到 cache 分支上。第五个支点（§4.5）承担一项不同的任务：证明**仓内既有的三条「看似能解决」的路径都不能解决**，从而把必要性钉死。

### 4.1 支点一：`_prefetch_occupied_span` 已经把两种记账语义封装在同一函数里

[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2531-L2537) 全文：

```python
    def _prefetch_occupied_span(self, prefetch_key, host_indices) -> int:
        """Occupancy units held by a prefetch: cache mode reserves the
        requested span at enqueue; buffer mode grants at hit-alloc, sized
        to the allocation (0 while still querying / parked)."""
        if self.host_memory_mode == "buffer_only":
            return len(host_indices) if host_indices is not None else 0
        return len(prefetch_key)
```

七行代码，三点论证：

1. **两种语义的差异是已知的、被命名的、被封装的**。docstring 用 `reserves` / `grants` 两个不同动词，并把 buffer 模式的 “0 while still querying / parked” 显式写出来——parked 状态占用为 0，因为 park 者不持有页。这是一个完整的授予语义。
2. **cache 分支返回的是 `len(prefetch_key)`，即请求跨度**，其中包含 miss 部分。它既不等于实际授予的页数（`alloc_len`），也不与实际持有的 host 内存挂钩。这正是 `prefetch_rate_limited()` 的 buffer 分支注释所说 “requested spans **overstate** it” 的来源。
3. **本方案不需要说服审稿人「问题存在」**。源码 docstring 已经描述了它。论文要做的是指出：cache 模式缺的不是新机制，而是隔壁分支已有的那个机制。

### 4.2 支点二：`ongoing_prefetch` 就是缺失的预留记录

`ongoing_prefetch` 是一个以请求为键的字典，值是一个具名元组 `info`，携带授予所需的全部信息：请求身份（`rid`）、锚点（`anchor_node_id` / `anchor_lock_params`）、请求跨度（`prefetch_key`）、已授予的页（`host_indices`）。

- **创建点**：`hiradix_cache.py:2005`、`unified_radix_cache.py:2799`（后者以 `info._replace(host_indices=host_indices)` 就地更新授予结果）。
- **销毁点**：`hiradix_cache.py:1825`（正常完成，`pop`）、`unified_radix_cache.py:2215`（正常完成，`del`）、`hiradix_cache.py:553`（强制清理）、`storage_attachment.py:410`（revoke）。

> **R1 的实质动作 = 把正常完成路径上的 `pop` 改为 `transfer`。**

条目在 \( t_{\text{land}} \) 被销毁，但它携带的信息在 \( t_{\text{admit}} \) 之前一直有用。R1 不新建数据结构的内容，只把它从 `ongoing_prefetch` 迁移到一个新的 `landed_grants` 容器，同时对刚落地的节点调 `protect_host()`。这是「约 60 行」估计的来源：迁移 + 钉住 + 准入侧解钉与注销 + 反死锁钳制 + `is_idle()` 对称判定。

### 4.3 支点三：钉住原语完备且已被正确消费

五项证据（全部亲自读码核实）：

1. 原语存在且带真实不变式检查：`protect_host()` / `release_host()`，后者在计数已为零时抛 `RuntimeError`（`radix_cache.py:293-302`）。
2. 设计意图明确指向存储操作：`host_ref_counter` 的注释 “incremented when the node is referenced by a **storage operation**”（`radix_cache.py:269-271`）。
3. 淘汰路径确实尊重它：`evict_host` 的 `if x.host_ref_counter > 0: continue`（`hiradix_cache.py:1403-1404`）；另一消费者 `if any(n.host_ref_counter > 0 for n in nodes)`（`hiradix_cache.py:1350`）。
4. 准入路径已经在使用它：`for n in nodes_to_load: n.protect_host()`（`hiradix_cache.py:1455-1456`）。R1 只是把这个调用的时刻**从 \( t_{\text{admit}} \) 提前到 \( t_{\text{land}} \)**，并在 \( t_{\text{admit}} \) 保持幂等（不重复递增，或递增后由消费路径统一递减一次）。
5. unified 分支有等价的 `inc_host_lock_ref` / `dec_host_lock_ref`（`unified_radix_cache.py:951`、`956`），锚点已在用（`unified_radix_cache.py:1990-1994` 增、`2214` 减）。

**缺口只在候选集合的构造**：`_update_host_leaf_status`（`hiradix_cache.py:1222-1235`）不看 `host_ref_counter`。但这不构成障碍——`evict_host` 在实际摘除前有 `host_ref_counter > 0: continue` 的检查，候选集合里有它也会被跳过。R1 因此**不需要修改叶子状态逻辑**，只需要保证新生叶子在 \( t_{\text{land}} \) 时 `host_ref_counter \ge 1`。

### 4.4 支点四：反死锁钳制与反饥饿上界在仓内已有先例

钉住内存会减少可淘汰量，因此必须有上界，否则会死锁。buffer 模式的 `anchor_lock_cap_tokens`（[pipeline.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L240-L263)）已经把这个问题的正确解法写完了，注释逐字为：

> “Clamp by admission headroom: pins must leave room for the largest allowed request, else **a queued hold can wedge admission permanently (pool full, nothing retractable)**. No-headroom pools take no pins.”

配套实现：

- `try_lock_anchor` 返回五值枚举 `locked | no_anchor | cap_skip | cap_busy | anchor_lost`（`pipeline.py:757-820`，cap 检查在 L795-813）。**`cap_busy` 与 `cap_skip` 的区分正是「park 等位」与「放弃钉住」的区分**，R3-a 直接沿用。
- 反饥饿上界 `max_staged_admission_defers: int = 32`（`pipeline.py:266-268`），注释 “Arbitrary: a deferral means the admission budget and the allocator disagree about free slots, which waiting on decode rarely fixes.”
- ABA 防护：`reset()` 的账本（`pipeline.py:270-301`）中 `anchor_locks: dict[CacheRequestHandle, _AnchorLock]`，注释 “**Attempt-keyed locks** stop stale completions from unlocking retries.”（L299-300）。
- 降级路径：cap 无余量时 warning + `_grant_cap_skips += 1` + 不钉住（`pipeline.py:252-259`）。
- TP 一致放弃：`SGLANG_HICACHE_PREFETCH_BLOCKED_WAIT_S = 60.0`（[environ.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/environ.py#L784)）与 `_sync_prefetch_blocked_give_up`（`scheduler.py:3248-3265`，用 `ReduceOp.MIN` 做跨 rank 一致放弃）。

**本文的反死锁钳制不是新设计，是把 `anchor_lock_cap_tokens` 的公式搬到 cache 模式的授予上。** 这一点在投稿时很重要：审稿人最担心的「钉住会不会导致准入死锁」，仓库里已经有一条被生产验证过的答案。

### 4.5 支点五：仓内三条「看似能解决」的路径，逐条排除

新颖性主张的强度取决于**排他性**：不仅要说没人做过，还要说仓内现成的东西不够用。本节排除三条最可能被审稿人点名的既有路径。

#### 4.5.1 `--hicache-storage-prefetch-policy` 三值均不覆盖 \( W \)

这是一个**真实存在的运行时旋钮**，且直接暴露在 HTTP API 层（`http_server.py:1154/1172/1230`、`io_struct.py:1762`、`scheduler.py:5095/5110-5112`、`tokenizer_control_mixin.py:370/379/394-396`），意味着可以在不重启的情况下切换。其定义（[memory.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/arg_groups/fields/memory.py#L166-L172)）：

```python
    hicache_storage_prefetch_policy: A[
        str,
        Arg(
            help="Control when prefetching from the storage backend should stop.",
            choices=["best_effort", "wait_complete", "timeout"],
        ),
    ] = "timeout"
```

**默认值是 `timeout`，不是 `best_effort`。** 官方文档的三值描述为：`best_effort`（“Terminate prefetch when needed”）/ `wait_complete`（“Ensure complete prefetch, higher cache reuse”）/ `timeout`（“Balance between completion and best-effort”）[cite:24]。语义在 [unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2061-L2087) 的 `_can_terminate_prefetch`：

```python
    @rank_consensus(same_results=True)
    def _can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True
        if self.prefetch_stop_policy == "wait_complete":
            # Completion is committed by the ACK drain (sync-thread MIN-reduced
            # progress, rank-min queue drain); a vote here only adds collectives.
            return False
        if self.prefetch_stop_policy == "timeout":
            # Wall clocks differ across ranks: PP0 decides, TP takes MAX (any
            # rank timed out) and _all_reduce broadcasts the verdict along PP.
            should_terminate = False
            if self.pp_rank == 0:
                should_terminate = self._prefetch_timeout_check_linear_func(operation)
            should_terminate_tensor = torch.tensor(
                int(should_terminate), dtype=torch.int, device="cpu"
            )
            self._all_reduce(should_terminate_tensor, torch.distributed.ReduceOp.MAX)
            return should_terminate_tensor.item() == 1
        return True
```

三值控制的都是**同一个问题：什么时候允许提前终止一次尚未完成的预取**。而本文的 \( W \) 是**预取已经完成之后**的区间。三值逐条排除：

| 取值 | 语义 | 为何不覆盖 \( W \) |
|---|---|---|
| `best_effort` | 需要时立即终止 | 终止得更早 → \( t_{\text{land}} \) 提前 → \( W \) **变长**，故障加剧 |
| `timeout` | 超过 `prefetch_timeout_base + len(hash_value) * prefetch_timeout_per_page` 即终止（默认） | 同上，超时截断使 `fulfilled < requested`，同时进入 §2.6 的截断路径 |
| `wait_complete` | `_can_terminate_prefetch` 恒返回 `False`，直到 ACK drain 提交完成 | **完成的那一刻就是 \( t_{\text{land}} \)，也就是 \( W \) 的起点**。它保证「取全」，不保证「留住」 |

`wait_complete` 这一行需要更精确的论证，因为它是唯一可能让人误以为能解决问题的取值。关键在 `check_prefetch_progress` 的早退（[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2089-L2109)）：

```python
    @rank_consensus(same_params=True, same_results=True)
    def check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
        if request not in self.ongoing_prefetch:
            return True

        operation = self.ongoing_prefetch[request].operation
        ...
```

而 cache 模式的正常完成路径在 \( t_{\text{land}} \) 就执行 `del self.ongoing_prefetch[request]`（L2215）。于是 `check_prefetch_progress` 在 \( t_{\text{land}} \) 之后**恒返回 `True`**——门控解除，与该请求是否已被准入无关。请求此刻可以仍在 `waiting_queue` 里停留任意多轮。

代码里有一处注释看似矛盾（L2702-2706）：

```python
            if buffer_mode and cc.prefetch_rate_limited():
                # Pool is load-saturated: hold the KNOWN hit until staged
                # prefetches ahead of us are consumed. The op stays in
                # ongoing_prefetch, so wait_complete keeps gating admission.
                return False
```

“wait_complete keeps gating admission” 成立的前提是 **“The op stays in `ongoing_prefetch`”**，而这行本身在 `if buffer_mode and ...` 分支内——即**只有 buffer 模式的 park 路径才让 op 留在 `ongoing_prefetch` 里**。cache 模式没有 park 路径（§5.2），op 一旦完成即被 `del`。因此该注释恰好从反面证明了本文的论点：**门控能力来自「条目仍在」，而 cache 模式的条目在 \( t_{\text{land}} \) 就没了**。R1 的 `pop` → `transfer` 正是把「条目仍在」这一条件补回 cache 模式。

> **结论（F-e 的依据）**：`prefetch_stop_policy` 是一个**终止时机**旋钮，本文需要的是一个**释放时机**契约。前者三值穷尽，无一可用。这排除了「改配置即可」的反驳，并给出 A7 臂（§7.4）作为定量验证。

顺带记录两个与本节相关的预算常量（[cache_controller.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/managers/cache_controller.py#L265-L271)）：

```python
# Buffer-mode staging budgets. Prefetch staging is latency-critical
# (wait_complete gates TTFT), so loads may fill the pool up to this fraction
# before new prefetches are declined.
HICACHE_LOAD_POOL_USAGE_FRACTION = 0.9
# Write-staging floor: writes are deferrable, so the flush gate grows the
# write window dynamically into whatever load staging is not using.
HICACHE_WRITE_STAGING_POOL_FRACTION = 0.2
```

注释首行 “**Buffer-mode staging budgets**” 明确了 \( 0.9 \) 只服务 buffer 模式，与 cache 模式的 `prefetch_capacity_limit = int(0.5 * mem_pool_host.size)` 是**两套不同预算**。§5.3 的 \( \phi \) 取的是后者语义上的 0.5，不应误引 0.2。

#### 4.5.2 `enable_hicache_prefetch_retry` 的官方 help 文本自证不够

第二个旋钮的 help 文本（[memory.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/arg_groups/fields/memory.py#L177-L185)）逐字为：

```python
    enable_hicache_prefetch_retry: A[
        bool,
        Arg(
            help="Retry storage prefetch at schedule time when it was skipped "
            "due to transient limits (in-flight prefetch budget or host "
            "memory pressure). Disabled (default): a blocked request never "
            "retries and recomputes its full prefix.",
        ),
    ] = False
```

两点值得引用：

1. **“Disabled (default): a blocked request never retries and recomputes its full prefix.”** —— 这是**官方 help 文本级别的自证**，比代码注释更强：默认配置下，被 host 内存压力挡住的请求**永久重算全部前缀**。§2.1 的现象 1 由此得到上游确认。
2. **“when it was skipped due to transient limits”** —— 重试的语义域是「**从未成功发起**」，而本文根因二是「**成功发起但被截断**」、根因一是「**成功完成但被淘汰**」。两者都不属于 skipped，因此即使打开该开关也不覆盖（B6 消融验证）。

#### 4.5.3 已存在的 `session_ref` 机制为何不覆盖 \( W \)

这是本文最重要的排除性论证之一，也是**审稿人最可能提出的质疑**：SGLang 官方 blog 在 Unified Radix Cache deep dive 中明写 “**Session retention applies to L1/L2**; the storage backend manages L3 eviction”[cite:22]，而仓内确有完整实现（`session_ref_tracker.py` 全文 118 行，30 处命中）。若它已经保护 L2，本文的主张就落空。

三条排除理由，全部有代码依据：

**(a) 注册时机在请求完成时，不在准入时。** `register_session_ref` 的唯一调用点是 [unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L1100) L1100，处于请求完成路径；其 docstring 自述为 “Register a **non-streaming request's reusable leaves** with each component”（`session_ref_tracker.py:54`），且要求 `req.last_node is not None` 并断言 `last_node is not root_node`。而 \( W = [t_{\text{land}}, t_{\text{admit}}) \) 里的请求**尚未完成、甚至尚未准入**，其 `last_node` 还不存在。session_ref 的时间轴整体位于 \( W \) **之后**。

**(b) 生命周期是会话，本文是请求内跨阶段。** `session_ref_tracker.py:1-4` 的模块 docstring：“tag each request's KV by **session_id** for each tree component; `release_radix_session` (close) releases a session's tagged reference.” 释放点是 `/close_session`（`tokenizer_control_mixin.py:970`）或 tombstone（`_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192`）。它保护的是「**这轮结束后、下轮回来前**」的跨轮窗口；本文保护的是「**这轮预取落地后、这轮准入前**」的轮内窗口。两个窗口首尾相接、互不重叠——恰是 §3.4.2 末尾对 PEEK 的同一结论。

**(c) 它是软排序，不是硬保护。** 官方 blog 对 FULL 组件的描述是 “FULL orders unlocked, eligible entries by **reference presence, reference count**, and then the base eviction policy, usually reclaiming unreferenced data first”[cite:22]——即 `session_ref > 0` 只改变**排序**，不改变**是否可被淘汰**。代码侧的对应物是 [unified_tree_core.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L455-L458) 的 LRU 谓词：

```python
    def _session_lru_predicate(self, ct: ComponentType):
        if not self.enable_session_radix_cache or ct is ComponentType.FULL:
            return None
        return lambda node: node.component_data[ct].session_ref > 0
```

**FULL 组件直接返回 `None`**，即对 FULL 连谓词都不启用，退化为 `eviction_strategy.get_priority`（[components/full.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_cache/components/full.py#L45-L60) 的 `_ensure_eviction_strategy` 懒绑定）。这与 §3.4.2 对 PEEK 软保护的批评（F-f）是同一类失效：极端压力下排序偏好必然被数量压垮。

> **但 `session_ref` 是本文的实现载体，不只是被排除的对象。** 三处可直接复用：
>
> 1. **`component_protected_size_` 已存在**（`unified_tree_core.py:481`，与 `component_evictable_size_` 并列在 `reset()` 里初始化）——这是「protected」记账的现成容器，\( G_{\max} \)（§5.3）可以直接挂上去而无需新建全局计数器。
> 2. **引用计数沿父路径增减的三个函数已写好**（`components/full.py:62-102` 的 `_dec_session_coverage` / `_advance_session_coverage` / `_recede_session_coverage`），且带 `assert cd.session_ref > 0` 不变式检查——这正是本文 C1 断言（§5.7）所需的模式，可照抄。SWA 与 MAMBA 组件有等价实现（`components/swa.py:148-151`、`components/mamba.py:87-95`）。
> 3. **R1 的挂载点已经存在**：由于 `_session_lru_predicate` 对 FULL 返回 `None`，FULL 走的是 `self.tree_core.eviction_strategy.get_priority`（`components/full.py:56-60`）。要让硬保护对 FULL 生效，只需把该懒绑定改为一个「先检查 `host_ref_counter`、再委托 `get_priority`」的组合谓词——**这是 R1 在 unified 分支上的最小接入点**，且不必触碰 `evict_host` 本体（§4.3 已说明 `evict_host` 侧的硬跳过检查已存在）。

#### 4.5.4 Rust TreeCore 双后端：行号与生效范围的前提条件

`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND` 默认 `"python"`（[environ.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/environ.py#L671-L672)，注释 “Registered TreeCore backend serving the unified radix cache”），因此本文对 Python 侧 `evict_host` / `_session_lru_predicate` 的分析在默认配置下成立。但仓内已存在 Rust 后端（`unified_radix_cache.py:81/86-89` 的 `create_tree_core` 与 `tree_core_registry`；`unified_radix_cache.py:209-233` 读取该环境变量并据此打印 `Tree Core: {type(self.tree_core).__name__}`），官方 blog 说明其职责划分为：Rust 侧持有 “radix topology, **per-component lock accounting**, intrusive LRU lists, **eviction walks**”，Python 侧持有 “request-to-token mappings, physical KV allocation, pool ops and orchestration, match/insert/evict calls → ← deferred actions”[cite:22]。

对本方案的三条影响，必须在论文中如实陈述：

1. **“lock bookkeeping” 与 “eviction walks” 都在 Rust 侧**。若实验跑在 Rust 后端上，R1 的 `protect_host()` 提前将**不生效**，除非在 Rust 侧同步实现授予的引用计数。这是本文范围界定的一部分：**R1 主张在 Python TreeCore 后端上验证**。
2. **官方 blog 明确 “Session-aware caching continues to use the Python tree core.”**[cite:22] —— 即 §4.5.3 的 `session_ref` 机制本身也只在 Python 后端生效。这反而使本文与官方现状一致：两者共享同一个后端前提。
3. Rust 后端有平台限制（`rust_extensions/torch_build.py:37`：“the Rust TreeCore extension currently supports **Linux only**”）与 PyTorch 版本区间校验（同文件 L44/L56）。实验台若为非 Linux，则默认落在 Python 后端。

因此 §7.1 的实验台必须显式固定 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`，并把它作为一条可复现性声明写入 artifact。

---

## 5. 方案

本节给出完整方案。五个子节各自可独立阅读，建议顺序如下：

- **§5.0 分层发布**先给全景：五层改动（R1 / R2 / R3-a / R3-b / R4）各自修什么、约多少行、依赖谁、能单独投到哪里。这一节回答「最小可发表的改动是什么」。
- **§5.1 状态机与数据结构**是全文技术核心：把今日缺失的 `LANDED` 状态补上，并给出承载它的数据结构与四条正确性断言 C1–C4。**R1 的全部内容就在这一节。**
- **§5.2 授予仲裁与发放时机**处理根因二：当 host 池不足以同时满足所有在飞预取时，按什么**顺序**发（R3-a）、在什么**时刻**发（R3-b）。§5.2.2 用 Little 定律说明 R3-b 为什么是 R1 的必需配套而非可选优化。
- **§5.3–§5.5** 是三条安全边界：反死锁钳制 \( G_{\max} \)、反饥饿与队头保护、截断即可观测失败（含 R4 的尾部回队）。
- **§5.6–§5.7** 是两处工程连带改动：TP-lockstep 与 `is_idle()` 的正确性、撤销与异常路径的账目回收。

贯穿全节的组织原则只有一条：**把 buffer_only 模式已经验证过的授予生命周期，移植到仍然保留 L2 缓存语义的 cache 模式**。§2.8 已论证 buffer_only 对本文两个根因双重免疫，但代价是放弃 L2 本身；因此本文不是发明一个新机制，而是把同一仓库另一条分支上已经跑通的机制搬到需要它的那条分支上。这一点同时是 §4 全部代码支点的来源，也是 R1「≈60 行」这个规模估计的依据——需要新写的只有授予容器与状态迁移，钉住原语、限流钳制、park 与重试骨架在仓内都已存在且已被正确使用。

### 5.0 分层发布

| 层 | 内容 | 规模 | 依赖 | 投稿目标 |
|---|---|---|---|---|
| **R1** | 授予生命周期：`pop` → `transfer`，落地即钉住，准入才释放 | ≈60 行 | 无 | HotStorage'27 / HotOS'27 |
| **R2** | 截断即可观测失败：`truncated` 标志 + 差额计数器 + 截断分支日志 | ≈50 行 | R1 | 与 R1 合并投 |
| **R3-a** | 队列序**仲裁**：cache 分支启用 park + 扫描 `break`，授予稀缺容量时按入队序 | ≈80 行 | R1 | ATC'27 / EuroSys'27 |
| **R3-b** | laxity **门控发放时机**：仅当 \( \widehat{L}_r(t) \le \theta \) 时才把 PARKED 提升为 GRANTED | ≈50 行 | R3-a | 与 R3-a 合并投 |
| **R4** | 部分完成重试：尾部 `TAIL` 授予回队，`queue_seq` 继承 | ≈70 行 | R2 | 与 R3-a 合并投 |

**R1 是 go/no-go 层**。它单独就能消除根因一（被淘汰），且判据（§6 的 \( \rho \)）是确定性的：修复前 \( \rho < 1 \) 且随并发单调下降，修复后 \( \rho \equiv 1 \)。R3-a 才是消除根因二（截断）的层，但它的收益依赖 R1 提供的授予身份，无法单独评估。

**R1 与 R3-b 是一对成本对冲**。§5.2.2 用 Little 定律说明：R1 把授予的占用时长 \( D = t_{\text{admit}} - t_{\text{land}} \) 从今日的「不计账（页无保护，随时被抢）」变成「真实的正数」，这必然在固定 \( G_{\max} \) 下压低可持续授予速率；R3-b 的职责正是把 \( \overline{D} \) 压回最小。因此 R3-b 不是装饰性的调度优化，而是 R1 的成本对冲项。

### 5.1 状态机与数据结构

今日 cache 模式的预取只有三个状态（未发起 / 进行中 / 已完成），缺的是 `LANDED`。补上后：

```
(无状态) --t_issue--> PENDING --t_alloc--> GRANTED --t_land--> LANDED --t_admit--> CONSUMED
                        |                    |                   |
                     空间不足              截断              中止/超时/超 defer 上界
                        v                    v                   v
                     PARKED            GRANTED(p)+TAIL 回队     REVOKED
                    (不耗重试)                                  (可观测)
```

```python
class GrantState(IntEnum):
    PENDING = 0    # 已发起，未授予页
    GRANTED = 1    # 已授予 host 页，传输中
    PARKED = 2     # 授予被推迟（空间不足 / cap_busy），不持有页，不耗重试预算
    LANDED = 3     # 页已入树，钉住中，等待准入          ← 今日缺失
    CONSUMED = 4   # 准入已 load_back，授予注销
    REVOKED = 5    # 主动撤销（超时 / 超 defer 上界 / 请求中止），可观测


class _PrefetchGrant(msgspec.Struct):
    rid: str
    queue_seq: int              # 授予序号；R1 恒为 0（不排序），R3-a 起生效
    attempt: int                # ABA 防护，对应 pipeline.py:299 “Attempt-keyed locks”
    requested_tokens: int       # len(prefetch_key)，保持与今日记账的平价
    granted_tokens: int         # alloc_len（hit-sized），R1 新增
    truncated: bool             # granted_tokens < 已命中的 L3 跨度
    node_ids: list[int]         # LANDED 后承载授予页的树节点
    state: GrantState
    defer_rounds: int           # 反饥饿计数，上界 max_staged_admission_defers
```

核心不变式：

\[
\text{release}(g_r) = \text{admit}(r), \qquad \text{complete}(g_r) \prec \text{admit}(r)
\]

其中 \( |\{ r' : t_{\text{land}}(r') \le t < t_{\text{admit}}(r') \}| \) 即窗口内授予数，随并发度增长且无上界——这正是必须引入 \( G_{\max} \)（§5.3）的原因。

**记账平价（R1 的保守性设计）**。R1 刻意**不**改动既有计数器：

- 保留 `prefetch_tokens_occupied -= len(prefetch_key)` 于 \( t_{\text{land}} \)。这维持了 `prefetch_rate_limited()` 在 cache 分支的既有语义，避免在同一次改动里同时变更两件事而混淆归因。
- **另加**一个新计数器 `granted_tokens_landed`，生命周期严格为 \( [t_{\text{land}}, t_{\text{admit}}] \)，覆盖旧计数器不覆盖的区间。

于是 R1 是纯增量改动：旧计数器逐位不变（可用现有单元测试与日志 `occupied=%d` 回归验证），新计数器补上旧计数器的盲区。R3-a 再把两者合并为单一授予预算，届时 `prefetch_rate_limited()` 的 cache 分支也能像 buffer 分支一样改读真实占用。

### 5.2 授予仲裁与发放时机（R3）

一种看似自然的表述是「队列序发放」，即**用入队序同时决定「谁先拿到页」和「什么时候去拿页」**。本文不采用这种表述，理由是 TempoKV 的实证批评：其 Figure 2 标题为 “Queue rank is not a staging deadline.”，同一 rank 下余量从提前 2.67 s 到滞后 6.02 s [cite:12]。因此这两个职责在本文中被拆开：

- **R3-a：队列序只做仲裁**——当 host 授予预算稀缺、多个授予争同一批页时，按入队序决定谁先拿到。这条规则的正确性不依赖时间估计，TempoKV 也没有否定公平性规则，它否定的是**用 rank 推断 deadline**。
- **R3-b：laxity 门控发放时机**——授予何时从 PARKED 提升为 GRANTED，由一个显式的时间余量判据决定，而非由队列位置隐式决定。

#### 5.2.1 R3-a：队列序仲裁

R3-a 的全部要点是一个词：**`break` 而非 `continue`**。

`_drain_and_alloc_storage_hit`（[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2805-L2812)）已有的实现：

```python
def _drain_and_alloc_storage_hit():
    # Parked hits first (FIFO fairness with retries; buffer only).
    if buffer_mode:
        parked = self.buffer_pipeline.pending_hit_allocs
        while parked:
            if not _try_alloc_storage_hit(parked[0]):
                break
            parked.popleft()
    for operation in _drain_queue(cc.prefetch_hit_queue, n_storage_hit):
        ...
```

`break` 已经在 parked 队列的扫描里存在，注释也已写明理由 “FIFO fairness with retries”。但 `if buffer_mode:` 这个门把它挡在 cache 模式之外。

**R3-a 的实质动作 = 删掉 `if buffer_mode:` 这个门**，让 cache 模式在 `_try_alloc_storage_hit` 的三处 park 点也 `return False` 而非走 revoke：

| 行 | 今日 cache 模式行为 | R3-a 后 |
|---|---|---|
| `unified:2706` | 无 park（限流判据只在 buffer 分支） | 授予预算耗尽 → `return False` park |
| `unified:2720` | 无（`cap_busy` 是 buffer 专属） | cap 被前序授予占满 → `return False` park |
| `unified:2760` | 走 L2761-2771：revoke + `poll_miss` + `return True` | `return False` park，保留完整 `hit_tokens` |

`break` 生效后的语义正是本文出发点里描述的那个解法：**「后面的请求取不到了就不要取，等着，有位置了再取」**。前序请求 park 时，后续请求不再抢内存，从而不再触发 §2.5 的 `evict_host`；等前序请求准入释放授予，parked 队列按 FIFO 依次获得内存。

`queue_seq` 的来源已经在代码里：`scheduler.py:3339-3340` 中 `_prefetch_kvcache(req)` 在 `waiting_queue.append(req)` **之前**调用，因此发起时刻的 `len(waiting_queue)` 即该请求的初始序号。重试路径也已严格按 `waiting_queue` 顺序遍历（`storage_prefetch.py:57`），并有队头保护（L45-49，注释 “A speculative miss must never delay the queue head.”）。R3-a 不需要新建排序机制，只需要在授予时把这个已存在的序号记录下来。

#### 5.2.2 R3-b：为什么必须有一个时机门控

R3-a 只解决「谁先拿」，没有解决「什么时候拿」。在 cache 模式下「什么时候拿」不再是无关紧要的，因为 R1 引入的硬保护让**授予变成一种被计量的稀缺资源**。用 Little 定律把这件事写成等式：

\[
\overline{G} = \lambda_{\text{grant}} \cdot \overline{D} \le G_{\max}, \qquad D_r \triangleq t_{\text{admit}}(r) - t_{\text{land}}(r)
\]

即

\[
\lambda_{\text{grant}} \le \frac{G_{\max}}{\overline{D}} .
\]

三个系统的坐标：

| 系统 | \( D \) 的处理 | 后果 |
|---|---|---|
| SGLang cache 模式今日 | \( D \) **不被记账**（页在 \( W \) 内无保护，随时被抢走） | 已投入的 L3 IO 作废，\( \rho < 1 \)（根因一） |
| SGLang + R1 | \( D \) 被记账且受硬保护，\( \overline{D} \) 从 0 变为正 | 修复根因一，但在固定 \( G_{\max} \) 下压低 \( \lambda_{\text{grant}} \)，加剧截断（根因二） |
| SGLang + R1 + R3-b | \( D \) 被记账，且 \( \overline{D} \) 被压到 \( \widehat{\mathrm{TTR}} \) 量级 | 在不放大 \( G_{\max} \) 的前提下把 \( \lambda_{\text{grant}} \) 抬回来 |

这张表也给出了本文与 TempoKV 的关系的定量版本：TempoKV 的系统**已经有保护**，它的问题是承诺过早、\( \overline{D} \) 过大，因此它付的是 protected byte-time 的税（其报告的降幅为 63–91% [cite:12]）；SGLang cache 模式**根本没有保护**，它的问题是 \( D \) 完全不被承认，因此它付的是全部已投入 IO 的税。两者位于同一个最优点的两侧，用的是同一个不等式，方向相反。

#### 5.2.3 R3-b 的判据：SGLang 侧的可实现估计式

沿用 TempoKV 的余量形式，但两个分量都用 SGLang 已有的量构造，不引入新的估计器：

\[
\widehat{L}_r(t) \triangleq \widehat{\mathrm{TTU}}_r(t) - \widehat{\mathrm{TTR}}_r, \qquad \text{PARKED} \to \text{GRANTED} \iff \widehat{L}_r(t) \le \theta
\]

其中 \( \theta \) 为安全边界（默认 0，即 just-in-time 发放；\( \theta < 0 \) 为提前发放，\( \theta > 0 \) 为延后发放），由消融 B10 扫描。

**\( \widehat{\mathrm{TTR}}_r \)：复用仓库里已有的线性传输时间模型。** `_prefetch_timeout_check_linear_func`（[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2061-L2087)）已经在用

```
elapsed > prefetch_timeout_base + len(operation.hash_value) * prefetch_timeout_per_page
```

来判定预取是否超时。这个表达式的斜率项 `prefetch_timeout_per_page` 就是**代码库自己对「一页 L3→L2 需要多久」的估计**，截距项 `prefetch_timeout_base` 就是它自己的固定开销估计。R3-b 直接把同一对参数拿来当 \( \widehat{\mathrm{TTR}} \)：

\[
\widehat{\mathrm{TTR}}_r = \beta_0 + n_{\text{page}}(r) \cdot \beta_1, \quad \beta_0 = \texttt{prefetch\_timeout\_base},\ \beta_1 = \texttt{prefetch\_timeout\_per\_page}
\]

语义上这是一次反转：**同一个线性时间模型，今日被用来决定「何时放弃」，R3-b 用它来决定「何时开始」**。这不新增任何超参数，也不新增任何估计器，因此不会引入 TempoKV 那样的一整套 TTU/TTR 估计栈（其估计栈改编自 QLM 与 Past-Future [cite:12]）。需要如实说明的是：该模型是为「超时」标定的，其绝对值可能偏保守（超时阈值通常留裕量），因此 \( \widehat{\mathrm{TTR}} \) 偏大会使发放偏早——这一偏差方向是**安全的**（退化为 R3-a 的行为），且 B10 的 \( \theta \) 扫描能把它吸收掉。

**\( \widehat{\mathrm{TTU}}_r \)：用「还需多少个事件循环轮」而非绝对时间。** SGLang 的准入是离散的：请求在 `waiting_queue` 中，只有当 device token 预算与 running batch 容量允许时才被 `get_next_batch_to_run` 取走。因此把 TTU 分解为

\[
\widehat{\mathrm{TTU}}_r(t) = \widehat{N}_{\text{round}}(r, t) \cdot \hat{T}_{\text{round}}
\]

- \( \widehat{N}_{\text{round}}(r,t) \)：请求 \( r \) 还需多少个事件循环轮才被准入。可实现的廉价代理是 \( r \) 在 `waiting_queue` 中的位置 \( k_r(t) \) 除以每轮平均准入请求数 \( \hat{a} \)（对最近 \( m \) 轮实际准入数做 EMA）：\( \widehat{N}_{\text{round}} = \lceil k_r / \hat{a} \rceil \)。**必须显式声明这是一个 rank 型代理**，它继承了 TempoKV Figure 2 所批评的全部弱点；R3-b 用它只是为了在「完全不估时间」（今日）与「精确估时间」（TempoKV 全栈）之间取一个中间点，而不是宣称它精确。这一取舍的可辩护性由 B10 检验：若 \( \theta \) 扫描显示收益对估计精度不敏感，则廉价代理足够；若敏感，则应在 §8 风险 12 的路径上换成基于 `PrefillAdder` 干跑的预算型估计 [unverified: `PrefillAdder` 是否支持在不改变调度器状态的前提下做干跑，本次未核实]。
- \( \hat{T}_{\text{round}} \)：单个事件循环轮的墙钟时长，对 `time.monotonic()` 相邻轮差做 16 点 EMA（与 TierKV 的 EMA-16 传输时间估计同型 [cite:15]）。它是各 rank 独立的本地观测量，不进入任何集合通信的**载荷**，但会进入发放判定，因此其跨 rank 差异必须被显式消除——见下。

**TP 一致性：R3-b 复用 `_can_terminate_prefetch` 已有的 MAX all_reduce。** 这是 R3-b 唯一需要认真对待的工程约束。\( \widehat{L}_r \) 的两个分量都是本地估计（\( k_r \) 与 \( \hat{a} \) 来自本 rank 的队列观察，\( \hat{T}_{\text{round}} \) 来自本 rank 的时钟），各 rank 的读数**不会逐位相同**；而 `PENDING → GRANTED / PARKED` 的转移发生在 `prefetch_hits_sync_groups` 这个同步点上（§5.6 表第 2 行），一旦各 rank 对同一个 \( r \) 得出不同的发放决定，就会落入 [pipeline.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L12-L24) 所警告的 “a violation surfaces as an unexplained collective hang”。解法不是自建协商，而是照抄仓内已有的一处完全同构的先例：`_can_terminate_prefetch` 在 `timeout` 策略下先做本地判定，再对布尔结果取 **TP MAX all_reduce**（[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2061-L2087)）。R3-b 对 \( \widehat{L}_r(t) \le \theta \) 的判定采用同一形式：本地算出 \( \widehat{L}_r \)，在同一个同步组内对「是否达标」取 MAX（即**任一 rank 认为还早，就全体继续 PARKED**）。这带来两个后果，都需在论文中如实陈述：其一，R3-b **不新增同步点**，只在既有同步点的载荷里多带一个标量，因此 A4 相对 A3 的 TP 开销增量接近于零；其二，MAX 语义使发放时刻由**最悲观的 rank** 决定，\( \overline{D} \) 的实测值会略高于单 rank 估计所预测的值，Figure 3 应以 all_reduce 后的实际发放时刻为准来计算 \( D \)，而不是以本地 \( \widehat{L}_r \) 过零的时刻。

**钳制与失效模式。** R3-b 有两种对称的失效：

| 失效 | 原因 | 后果 | 钳制 |
|---|---|---|---|
| **过晚发放** | \( \widehat{\mathrm{TTU}} \) 被系统性低估（长队列、大 prefill） | 准入时预取尚未落地 → 请求退化为重算，等价于 A0 臂 | \( \theta \le 0 \) 时不设提前量；对 \( \widehat{N}_{\text{round}} \) 施加下界 1；PARKED 的 `defer_rounds` 仍受既有 `max_staged_admission_defers = 32` 上界约束（[pipeline.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/managers/pipeline.py#L266-L268)），到界即强制发放 |
| **过早发放** | \( \widehat{\mathrm{TTR}} \) 被低估 | \( \overline{D} \) 回升，退化为 TempoKV 描述的过量承诺 | \( G_{\max} \)（§5.3）本身是硬上界，即使全部授予同时提前发放也不会死锁 |

关键是这两种失效都**不破坏 R1 的正确性**：R1 保证「只要落地了就不会被抢」，R3-b 只影响「什么时候去让它落地」。因此 R3-b 是一个可插拔的效率层，可以在不触碰 R1 的前提下被完整消融掉（B10 的 \( \theta = -\infty \) 端点即等价于关闭 R3-b，退化为 R3-a）。

#### 5.2.4 R3-b 与「不改变请求调度」的兼容性

TempoKV 强调其方案 “without changing request scheduling” [cite:12]。R3-b 保持同一性质：它只改变**授予的发放时刻**，不改变 `waiting_queue` 的顺序、不改变 `PrefillAdder` 的准入判据、不改变任何请求的优先级。因此 R3-b 与 PEEK 的排队侧重排（增量 radix tree + dual-walk + pioneer-first 准入 [cite:13]）在原理上可叠加，两者作用在不同的决策上——这也是 §3.4.2 中「窗口串接而非重叠」的实现层含义。

#### 5.2.5 R3 的可测量预测

R3-a 与 R3-b 各自给出一条独立、可证伪的预测，构成 §7 的两组主结果：

- **R3-a**：截断率（§5.5 的 `truncated` 计数器）随并发上升的斜率显著变缓；`hit_tokens - alloc_len` 的差额分布左移。若 R3-a 后截断率与 A2 臂无差异，则 §2.4 的因果链（抢内存 → `evict_host` → 截断）不成立。
- **R3-b**：平均占用时长 \( \overline{D} = \overline{t_{\text{admit}} - t_{\text{land}}} \) 下降，且在 \( G_{\max} \) 不变的前提下可持续授予速率 \( \lambda_{\text{grant}} \) 上升。这条预测直接对应 Little 定律的等式，因此**即使端到端指标没有变化，\( \overline{D} \) 与 \( \lambda_{\text{grant}} \) 本身也是一张可发表的图**。

### 5.3 反死锁钳制

钉住授予会减少可淘汰的 host 内存。若钉住总量吃掉了准入所需的余量，就会出现 `pipeline.py:240-263` 注释所警告的 “a queued hold can wedge admission permanently (pool full, nothing retractable)”。因此授予总量必须有上界：

\[
G_{\max} = \max\!\Big(0,\ \min\big(\gamma \cdot S_{\text{host}},\ S_{\text{host}} - L_{\max} - \phi \cdot S_{\text{host}}\big)\Big)
\]

| 符号 | 含义 | 代码来源 |
|---|---|---|
| \( S_{\text{host}} \) | host 池总容量（token 数） | `mem_pool_host.size` |
| \( L_{\max} \) | 单请求最大上下文长度 | `max_context_len` |
| \( \gamma \) | 授予占池比例上限 | 对应 `SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP` |
| \( \phi \) | 写回 staging 保留比例 | `HICACHE_WRITE_STAGING_POOL_FRACTION = 0.2`（`cache_controller.py:268-271`） |

第二项 \( S_{\text{host}} - L_{\max} - \phi S_{\text{host}} \) 是「准入余量」项：必须给最大的单个请求留出可分配空间，否则钉住的授予会永久卡住准入。第一项 \( \gamma S_{\text{host}} \) 是「不与写回 staging 抢池」项。

**R1 的简化实现**（避免同时引入两个自由参数）：

\[
G_{\max}^{\text{R1}} = \min\big(\texttt{prefetch\_capacity\_limit},\ S_{\text{host}} - L_{\max}\big)
\]

其中 cache 模式的 `prefetch_capacity_limit` 已经等于 `int(0.5 * mem_pool_host.size)`（`cache_controller.py:592-610`，注释 “Budget speculative prefetch at half the host pool, leaving the rest for the write-back staging path”）。也就是说 **\( \gamma = \phi = 0.5 \) 已被既有代码选定**，R1 只是把它复用为授予上界，并额外减去 \( L_{\max} \) 保证准入余量。

**降级路径照抄先例**（`pipeline.py:252-259`）：当 \( G_{\max} = 0 \)（池太小，无任何余量）时，打 warning、`_grant_cap_skips += 1`、**不钉住**——即退回今日行为。这保证 R1 在小池配置下不会比现状更差，是严格的安全降级。

**cap-bind 率是 R3-a 的主要风险指标**（§8）：若 \( G_{\max} \) 在高并发下频繁绑定，授予会大量 park，预取吞吐下降，TTFT 可能反而变差。因此必须把 `_grant_cap_busy` / `_grant_cap_skips` 作为一等指标上报，与 `pipeline.py:519-523` 的 `live_cap` 三项式（`pool_tokens - cc.prefetch_tokens_occupied - pool_tokens // 10`）并列对照。

### 5.4 反饥饿与队头保护（R3-a）

park 是「等」，等必须有界，否则队头请求可能永久 park。五项设计：

1. **park 不消耗重试预算**。今日 cache 模式的 revoke 路径会调 `storage_prefetch_retries.poll_miss(...)`（`unified:2765-2767`），而 `max_attempts` 默认为 8（`memory.py:200-209`），超限后请求被 “admitted with what the device holds”。park 是一次**推迟**而非一次**新 L3 尝试**，符合 `StoragePrefetchRetries` docstring 里 “**New L3 attempts only** (parked IO and retained staging retry at their owner)” 的自我界定——parked IO 由其 owner 重试，不计入 L3 尝试预算。
2. **`defer_rounds` 上界复用既有常量** `max_staged_admission_defers = 32`（`pipeline.py:266-268`）。超过上界时走可观测的 `REVOKED`：注销授予、释放钉住、注册 `poll_miss`、打 warning 并递增 `_grant_defer_giveup`。
3. **撤销选 `queue_seq` 最大者（LIFO）**。当必须回收内存时，撤销最后授予的那个，理由是双重的：保住队头的前进性（队头 `queue_seq` 最小，永不被选中），以及保住已投入 IO 最多的请求（后授予者投入最少）。
4. **队头保护已有先例**：`storage_prefetch.py:45-49`，注释 “A speculative miss must never delay the queue head.”
5. **TP 一致放弃复用既有机制**：`SGLANG_HICACHE_PREFETCH_BLOCKED_WAIT_S = 60.0`（`environ.py:784`）与 `_sync_prefetch_blocked_give_up`（`scheduler.py:3248-3265`，`ReduceOp.MIN`）。R3-a 与 R3-b 都不新增同步点：R3-a 的仲裁在 `prefetch_hits_sync_groups` 既有同步点内完成，R3-b 的发放判定复用 `_can_terminate_prefetch` 已有的 TP MAX all_reduce 形式（§5.2.3），各 rank 的本地估计差异由此收敛为单一决定。

### 5.5 截断即可观测失败（R2 + R4）

R2 四步，全部在 `unified_radix_cache.py:2742-2751` 的截断分支与其下游：

1. **置标志**：`grant.truncated = True`，并保留原始 `hit_tokens` 于 `grant.requested_hit_tokens`（今日它被 `operation.storage_hit_count = alloc_len` 就地覆盖，无处可查）。
2. **差额计数**：新增 `_prefetch_truncation_stats`，上报「截断发生次数」与「截断损失的 token 总量」\( \sum (\text{hit\_tokens} - \text{alloc\_len}) \)。这是根因二的**唯一直接测量手段**——今日框架完全不记录它。
3. **补日志**：在截断分支也调 `_log_storage_prefetch_deferred(alloc_len, "host_capacity")`。今日只有完全失败分支（L2757、L2768、L2788）调它，截断分支静默。同时 `_resolve_storage_prefetch_tokens(..., reason="host_capacity")`（L2793-2795）已在做结算，R2 只需让 reason 出现在指标里而不只是日志里。
4. **改返回值**：`return False`（park）而非 `return True`。这一步把根因二与根因一在策略层连起来——**只要有请求处于截断状态，就不再为后续请求授予**，避免「A 被截断的同时 B 抢走了 A 的尾部所需内存」。

R4（部分完成重试）在 R2 之上补一步：**尾部回队**。为未取到的 \( \text{hit\_tokens} - \text{alloc\_len} \) 构造一个 `TAIL` 授予，`queue_seq` **与原授予相同**，追加到紧随其后的位置。相同序号保证公平性不被破坏（TAIL 不插队到别人前面，也不落到队尾），紧随位置保证「有位置了再取」。这需要放宽 `PrefetchStatus.DONE` 的 “Never retry” 语义——改为「完整命中不可重试，部分命中可续取」，是本方案唯一需要触碰状态机枚举语义的地方，故排在 R4 而非 R1。

### 5.6 TP-lockstep 与 `is_idle()` 连带改动

[pipeline.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L12-L24) 的契约逐字为：“There is no runtime verification; a violation surfaces as an unexplained collective hang.”

授予状态的五个变更点：

| 转移 | 发生线程 / 同步点 | 代码 |
|---|---|---|
| 创建 `PENDING` | scheduler 线程（非同步点，各 rank 各自创建，序号由 `len(waiting_queue)` 决定，天然一致） | `scheduler.py:3339-3340` |
| `PENDING → GRANTED / PARKED` | `prefetch_hits_sync_groups` | `cache_controller.py:612-614` |
| `GRANTED → LANDED` | `prefetch_completion_sync_groups` | `cache_controller.py:615` |
| `LANDED → CONSUMED` | `load_back`（准入路径，各 rank 同步推进） | `hiradix_cache.py:1422-1497` |
| `* → REVOKED` | 与上述同一同步点内的超时/上界判定 | `scheduler.py:3248-3265` |

> **R1–R4 不引入新的同步点，只改既有变更点上的语义。** 这两个 gloo 组的创建注释（`cache_controller.py:612-615`）为 “dedicated gloo groups so storage prefetch sync is isolated from other collectives and consistent across CPxTP participants”——一致性机制已经就位。
>
> **一处必要的澄清（R3-b）**：表中第 2 行 `PENDING → GRANTED / PARKED` 在 R3-b 下不是一次性的，而是**每一轮在该同步点上重新求值**——PARKED 的授予在后续轮次被提升为 GRANTED（§5.2.3）。这不新增同步点，也不新增状态变更的**种类**（提升仍是「发放」这一动作），但确实在该同步点的载荷里多带一个标量：本地 \( \widehat{L}_r \) 的达标布尔值，按 `_can_terminate_prefetch` 的既有形式取 TP MAX。第 5 行 `* → REVOKED` 同样在该同步点内判定，因此 `defer_rounds` 超过 32 的强制发放与超时撤销共享同一次 all_reduce，不会出现「一个 rank 发放、另一个 rank 撤销」的分裂。

一处必须连带修改、否则会产生新 bug 的地方：**`is_idle()`**。`pipeline.py:303-311` 的 buffer 侧判定已包含 `staged_prefetches`（即「已落地未准入」的容器非空则不 idle）。cache 模式必须对称地加入 `landed_grants` 判定，否则停机流程（`scheduler.py:5051-5055` 消费 `is_idle()`）会在授予尚未消费时提前返回，导致正在钉住的页被强制清理路径（`hiradix_cache.py:532-555`）以 `try/except: pass` 的方式处理掉。

### 5.7 撤销与异常路径

三条清理路径必须全部覆盖，且都必须走 `release_host()` 而非仅递减整数：

1. **正常消费**（`LANDED → CONSUMED`）：准入 `load_back` 后注销授予、递减 `granted_tokens_landed`、对 `node_ids` 逐个 `release_host()`。注意 `hiradix_cache.py:1455-1456` 已在准入时 `protect_host()`，R1 需保证不重复递增（或递增后成对递减）。
2. **主动撤销**（`LANDED → REVOKED`）：超时、超 `defer_rounds` 上界、请求中止。此路径今日不存在，是 R1 新增。它必须处理 §2.4 提到的 `unified_radix_cache.py:2646-2649` 断言前提失效问题——新增 post-land 撤销分支，且断言条件需相应放宽。
3. **异常清理**（强制释放 pending prefetch）：`hiradix_cache.py:532-555` 与 `storage_attachment.py:388-410`。这两处今日都用 `max(0, …)` 钳制整数；R1 之后它们应按 `attempt` 键控注销授予条目，钳制降级为断言（若触发即为 bug）。

四条正确性断言（建议在 debug 构建中启用，release 构建中为 no-op）。编号用 **C** 前缀以与 §7.4 的对照臂 A0–A8 区分：

| 编号 | 断言 | 检测的 bug |
|---|---|---|
| **C1** | 无双减：同一 `(rid, attempt)` 至多注销一次 | ABA / stale completion 误解钉；由 `release_host()` 的 `RuntimeError` 天然兜底 |
| **C2** | 无泄漏：`landed_grants` 为空 \( \iff \) 所有授予节点的 `host_ref_counter` 之和为 0 | 钉住未释放导致 host 池永久缩水 |
| **C3** | 无死锁：\( \sum_{g \in \text{LANDED}} g.\text{granted\_tokens} \le G_{\max} \le S_{\text{host}} - L_{\max} \) | 准入余量被吃光 |
| **C4** | TP 一致：各 rank 的 `len(landed_grants)` 与 `queue_seq` 序列逐轮相同 | 集体通信挂起；debug 构建用 all-reduce MIN/MAX 相等性验证 |

C2 与 C3 是本方案与「朴素加锁」方案（用户提出的第二解法）的分界：朴素加锁既无 C2 的泄漏检测，也无 C3 的上界，因此在高并发下会以「host 池逐渐缩水直到准入卡死」的形式失败，且失败模式是静默的。C4 在 R3-b 下还多覆盖一项：各 rank 对 \( \widehat{L}_r \le \theta \) 的本地判定必须在 all_reduce 之后一致（§5.2.3），否则发放决定分歧会直接表现为集体通信挂起。

---

## 6. go/no-go：一个可在未修改代码上直接测量的判据

### 6.1 兑现率 \( \rho \)

定义：

\[
\rho \;=\; \frac{\text{准入时重测的存储来源 host 命中长度}}{\text{落地时记录的 loaded\_from\_storage}}
\]

分子与分母**今日都已存在，且在同一段代码里相隔 12 行**。[scheduler.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/managers/scheduler.py#L4048-L4062)：

```python
                    # Pop the L3-loaded span. Unified cache exposes its absolute
                    # start so cache-mode L2/L3 attribution survives L3-tail eviction.
                    loaded_tokens, loaded_start = (
                        self.tree_cache.pop_prefetch_loaded_span(
                            req.cache_request_handle
                        )
                    )
                    if loaded_tokens > 0:
                        req.storage_hit_length = loaded_tokens
                        req.storage_hit_start = loaded_start
                        # Cache-mode host memory is a resident L2 tier. Buffer mode
                        # marks the staged span below once it is surfaced.
                        req.host_hit_is_storage = False

            req.init_next_round_input(self.tree_cache)
```

- **分母** = `loaded_tokens`，来自 `pop_prefetch_loaded_span`（[unified_radix_cache.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/mem_cache/unified_radix_cache.py#L2440-L2447)，docstring “Pop the loaded L3 token count and its absolute prefix start.”），其值在 \( t_{\text{land}} \) 写入 `prefetch_loaded_tokens_by_reqid`（`hiradix_cache.py:1842` / `unified_radix_cache.py:2218`）。**关键性质：该数字在页被淘汰之后不会回滚**——`pop_prefetch_loaded_tokens` 的 docstring（`hiradix_cache.py:1902-1908`）只承诺 “Returns 0 if no prefetch was done or was revoked”，被淘汰不属于 revoked。
- **分子** = `req.host_hit_length`，由紧随其后的 `init_next_round_input` 重算。该函数在 [schedule_batch.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/managers/schedule_batch.py#L1596) 调用 `tree_cache.match_prefix(...)`，而 `match_prefix` 的 host 段累加逻辑（`hiradix_cache.py:1926-1941`）是 `while last_node.evicted: host_hit_length += len(last_node.host_value)`——**节点若已在 `evict_host` 的 L1411-1412 被从父节点摘除，这里就再也累加不到它**。

于是 \( \rho \) 的语义是精确的：**落地时以为拿到了多少，准入时树上还剩多少。**

L4048-4049 的注释值得单独引用：“so cache-mode L2/L3 attribution **survives L3-tail eviction**”。作者已经意识到淘汰会破坏归因，并用「暴露绝对起点」来修补。但该修补保住的是**账面归因**，不是**页本身**——`storage_hit_start` 依然指向一段已经不在树上的区间。这正是 §2.6 所说「截断/丢失被记为成功」的最后一环。

### 6.2 判据表

| 观测组合 | 诊断 | 归属 |
|---|---|---|
| `loaded_tokens > 0` 且准入时 `host_hit_length` 中存储来源部分 \( = 0 \) → \( \rho = 0 \) | 页在 \( W \) 内被全部淘汰 | **本文根因一** |
| `loaded_tokens > 0` 且 \( 0 < \rho < 1 \) | 页在 \( W \) 内被部分淘汰（尾部先失） | **本文根因一（部分）** |
| `loaded_tokens` 显著小于 `storage_hit_count` 的历史值 → 需 R2 的 `requested_hit_tokens` 才能区分 | \( t_{\text{alloc}} \) 处被截断 | **本文根因二**（今日不可测，R2 使其可测） |
| `loaded_tokens = 0` 且 `query_storage_hit_length = 0` | L3 侧本就无命中 | 非本文问题 |
| `loaded_tokens = 0` 但 L3 有命中，且卸载负载后不自愈 | 信念闩锁（不自愈型故障） | **非本文问题**，由 §3.3 的 F-c / F-d 判据排除 |

第三行是 R2 的必要性论证：**根因二在今日的代码里原则上不可测**，因为 `operation.storage_hit_count` 被就地改写为 `alloc_len`，原始的 L3 命中跨度只以差值形式进入 `_resolve_storage_prefetch_tokens` 并随即结算。没有 R2 的 `requested_hit_tokens`，实验只能测到根因一。

### 6.3 最小测法（零代码修改）

go/no-go 实验**不需要实现任何 R 层**，只需在 `scheduler.py:4062` 之后加一行日志，把两个已经存在的数字并排打出来：

```python
req.init_next_round_input(self.tree_cache)
# ρ probe: granted-at-land vs. present-at-admit
if req.storage_hit_length > 0:
    logger.info(
        "GRANT_RHO rid=%s granted=%d present=%d rho=%.3f qdepth=%d",
        req.rid, req.storage_hit_length, req.host_hit_length,
        req.host_hit_length / req.storage_hit_length, len(self.waiting_queue),
    )
```

（`req.storage_hit_length` 与 `req.host_hit_length` 的口径差异需在实现时对齐：前者是纯 L3 来源计数，后者含 L2 既有命中。严格版本应减去本轮 `match_prefix` 中非存储来源的部分，或用 `storage_hit_start` 做区间交集。此为探针实现细节，不影响判据成立。）

**判别规则**：

- 若在高并发下 \( \rho \) 系统性 \( < 1 \) 且随 `qdepth` 单调下降，而在单并发下 \( \rho \equiv 1 \) → **根因一确证，R1 是 go**。
- 若 \( \rho \equiv 1 \) 恒成立但端到端命中率仍为 0 → 本文诊断被证伪，问题在别处（应转向 L3 命中缺失、不自愈型的信念闩锁，或调度层准入策略）。
- 若 \( \rho \equiv 1 \) 且 `loaded_tokens` 本身就小 → 根因一不成立，但根因二可能成立，需先实现 R2 才能判定。

这条判据的强度在于它是**确定性的、单点的、无需对照组即可自我证伪的**：\( \rho < 1 \) 与 \( \rho \equiv 1 \) 之间没有解释空间。

---

## 7. 实验设计

### 7.1 实验台

单机多实例即可复现，无需真实集群：本文的两个根因都发生在**单实例的 host 层内部**，跨实例共享池只提供 L3 命中的来源。

**（a）官方基准配置作为 Tier-A 实验台。** 本文的动机现象已经有一组公开、可引、可对齐的官方测量 [cite:22]：

| 项 | 官方配置（DeepSeek-V4-Flash） | 官方配置（Inkling-Small） |
|---|---|---|
| 注意力组件 | FULL + SWA | FULL + SWA + MAMBA |
| 硬件 / 并行 | 4×H200，TP4 | 8×H200，TP8 |
| 客户端 / 轮数 | 48 / 60 | 64 / 30 |
| 输入 / 输出 token | 4,096 / 16 | 1,216 / 64 |
| L3 | 500 GiB Mooncake | 同 |
| 有效输入吞吐 | L1 9.4K → L1+L2 14.3K → L1+L2+L3 145.5K tok/s（15.5×） | L1 15.5K → L1+L2 21.1K → L3 67.1K tok/s |
| 崩塌形态 | Round 40–55：L1 ≈0%、**L1+L2 ≈0%**、L3 ≈98% | **Round 17→18→19：≈95%→62%→0%**，同期 L3 保持 ≈95% |

采用它的理由是三点：其一，Tier-A 与官方数字同配置，A0 臂的预期形状因此不是本文自设的，而是**可被第三方对照的**；其二，Round-wise 曲线是逐轮测量，天然给出「崩塌是断崖而非渐进平台」的证据（§2.1）；其三，500 GiB Mooncake 满足本文的硬约束（集群范围共享池 + 跨实例复用）。

**（b）Tier-B：单机缩小配置。** Tier-A 需要 4×H200，不具普适可复现性。Tier-B 用同一形状但缩小的配置承担消融与对照臂的全部工作量：

- **模型**：一个 KV 体积足够大的模型（使 \( S_{\text{host}} \) 与 \( L_{\max} \) 处于同一量级，从而 \( G_{\max} \) 会真实绑定）。建议 context ≥ 128K，与观测到的「128K 需求只命中 40K」现象对齐。
- **配置**：`--hicache-host-memory-mode cache`（默认）、`--hicache-storage-backend <L3 后端>`、`--hicache-write-policy write_through`（默认，[memory.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/arg_groups/fields/memory.py#L118-L124)）、`--hicache-ratio` 作为扫描轴（默认 cache 模式为 `2.0`，`hicache_hook.py:79-86`）。
- **树后端必须显式固定**：`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`（默认值，[environ.py](file:///c:/Users/tesst/Desktop/sglang/SGLang-private-hicache-kv-backend-0.5.20/python/sglang/srt/environ.py#L671-L672)）。理由见 §4.5.4：R1 的钉住/释放挂载点在 Python 侧的 `host_value` 记账上，而 Rust TreeCore 持有独立的 per-component 锁记账与淘汰遍历，且该扩展当前仅支持 Linux（`rust_extensions/torch_build.py:37`）。这条必须作为 artifact 的可复现性声明写出，否则审稿人在 Rust 后端上无法复现。
- **负载**：多轮对话 trace，轮间共享长前缀（保证 L3 侧真实命中）。到达过程用泊松 + 突发两种，突发型用于放大 \( W \)。轮数需覆盖到官方曲线的崩塌区（≥60 轮），否则可能落在 L1+L2 仍有 95% 命中的「未崩塌段」而测不到效应。
- **重试开关**：主实验保持默认 `enable_hicache_prefetch_retry=False`，以呈现真实的 one-shot 语义；消融 B6 中单独打开，验证「打开重试也不足以修复」（因为截断路径根本不调 `poll_miss`，§2.6）。
- **预取策略开关**：主实验保持默认 `--hicache-storage-prefetch-policy timeout`；A7 臂改为 `wait_complete`（§7.4）。

**（c）明确的范围限制。** 本文的机制作用于承载 `host_value` 的树节点，其论证基于 FULL 组件的淘汰路径。SWA 组件的 tombstone 独立淘汰、MAMBA 组件的 recurrent state 淘汰、以及 sidecar 池（compressed KV / indexer buffers / compressor states）的从属淘汰规则 [cite:22]，本文**不覆盖**，也未逐行核实其与 R1 的交互 `[unverified: SWA tombstone 与 MAMBA 组件的 host 侧淘汰是否会绕过 host_ref_counter 判定，本次未核实]`。因此 Tier-A 若采用 Inkling-Small（三组件）配置，必须把「安全复用边界取所有组件都接受的最深边界」这一官方规则 [cite:22] 列为一条待验证的交互项；Tier-B 建议优先选 FULL-only 模型以规避该不确定性。

### 7.2 三张图：动机曲线、\( \rho \) 阶跃与 Little 定律验证

**Figure 1（动机图）：逐轮命中率的断崖。** 直接复用 §7.1(a) 的官方测量口径，绘制 Round-wise 的 L1 / L1+L2 / L3 三条命中率曲线，并在同一张图上叠加本文的两个内部计数器：窗口内 `evict_host` 命中授予页的次数（§2.5）与截断差额 \( \sum(\text{hit\_tokens} - \text{alloc\_len}) \)（§2.4）。论证目标是让「L1+L2 断崖归零」与「L3 命中率不变」并置出现，并把断崖的轮次与本文两个计数器的起飞轮次对齐。预期形状：官方 Inkling-Small 的 Round 17→18→19 = 95%→62%→0% [cite:22]；若 A0 臂复现出同一形状而 A2 臂把它拉平为非零平台，Figure 1 就完成了从「官方现象」到「本文根因」的归因闭环。

**Figure 2（主图）：\( \rho \) 与负载阶跃。** 一张图承担三个论证任务：

1. **\( \rho \) 随并发 \( N \) 单调下降**（根因一的量化）：横轴 \( N \in \{1, 4, 16, 64, 256\} \)，纵轴 \( \rho \) 的分布（p50 / p90 / 均值），曲线对应 A0 / A2 / A3 / A4 / A7 / A8。
2. **负载阶跃后立即自愈**（F-c，排除 [cite:9] 的极限环）：\( N \) 从 256 突降到 1，逐轮绘制 \( \rho \)。A0 应立即回到 1.0（因为窗口消失），这一「看似正常」的恢复正是该故障长期未被发现的原因——它只在高并发下出现，且卸载即消失。
3. **固定负载下方差不发散**（F-d，排除 [cite:10] 的亚稳态）：在 \( N = 64 \) 恒载下长时间运行，绘制 \( \rho \) 的滑动方差。应收敛而非发散。

第 2 点是本文的**类型学分界实验**，也是投稿时的关键一击：它用同一次负载阶跃把「窗口暴露」与所有**不自愈型**故障（[cite:9] 的极限环、[cite:10] 的亚稳态、以及任何基于持久信念的闩锁机制）分开。这一自愈性不只是描述性的——它是 §5 修复路线之所以对症的前提：只有当损失完全由「在飞授予的并发规模」驱动时，消除窗口才足以消除损失。若实验中 A0 臂在负载下降后仍不恢复，则诊断必须重做，且本文方案不成立。

**Figure 3：Little 定律的直接验证。** 绘制 \( \overline{D} = \overline{t_{\text{admit}} - t_{\text{land}}} \) 与可持续授予速率 \( \lambda_{\text{grant}} \) 随 \( N \) 的变化，并叠加 \( G_{\max} / \overline{D} \) 这条解析上界曲线（§5.2.2）。若实测 \( \lambda_{\text{grant}} \) 贴着上界走，则「R1 抬高 \( \overline{D} \)、R3-b 压低 \( \overline{D} \)」的因果链被定量确认；这张图也是本文与 TempoKV 的 protected byte-time 图（其报告降幅 63–91% [cite:12]）的**同一坐标不同侧**的直接对照，因此是 §3.4.1 正交性论证的实验支撑。

### 7.3 指标

| 类别 | 指标 | 来源 |
|---|---|---|
| **判据** | \( \rho \)（p50/p90/均值）、\( \rho = 0 \) 的请求占比 | §6.3 探针 |
| **端到端** | 多轮命中率（逐轮）、TTFT、TPOT、goodput、重算 token 总量 | 框架既有指标 |
| **后端** | L3 读带宽随时间曲线、L3 读字节 / 有效消费字节之比（IO 放大） | 后端计数 |
| **窗口** | \( \lvert W \rvert \) 的分布（\( t_{\text{land}} \to t_{\text{admit}} \) 实测）、窗口内 `evict_host` 命中授予页的次数 | 需插桩，R1 后由 `landed_grants` 免费提供 |
| **Little 定律（R3-b）** | \( \overline{D} \)、\( \lambda_{\text{grant}} \)、实测 \( \lambda_{\text{grant}} \) 与 \( G_{\max}/\overline{D} \) 之比 | 新增计数器，Figure 3 |
| **估计误差（R3-b）** | \( \widehat{\mathrm{TTU}} \) 与实测 TTU 的残差分布、\( \widehat{\mathrm{TTR}} \) 与实测传输时长的残差分布、因过晚发放而退化为重算的请求占比 | 新增探针，§5.2.3 失效模式表 |
| **截断（R2）** | 截断发生次数、\( \sum(\text{hit\_tokens} - \text{alloc\_len}) \)、`requested_hit_tokens` 分布 | `_prefetch_truncation_stats` |
| **授予（R1/R3-a）** | `granted_tokens_landed` 时序、`_grant_cap_busy` / `_grant_cap_skips` 率（cap-bind 率）、park 时长分布、`defer_rounds` 分布、`REVOKED` 次数 | 新增计数器 |
| **淘汰对抗** | 由 `evict_host` 摘除的、属于 in-flight/landed 授予的页数（§2.5 的直接测量） | 需插桩 |
| **既有可观测点** | `_prefetch_outcome_stats["declined_anchor_lost" / "declined_device_covered"]`、`_log_storage_prefetch_deferred(…, "host_capacity")` 频次、日志 `occupied=%d`、`insert_result.host_insert_dropped` | 已存在，无需改动 |

「IO 放大」一项尤其重要：它是**被否决的 force 重预取方案（A5）的判死指标**——A5 会把 \( \rho \) 拉回 1.0，但 IO 放大随并发线性增长，因为每次重预取都重读一遍已经在 L2 落地过（又被淘汰）的数据。

### 7.4 对照组（九臂）

| 臂 | 配置 | 实现成本 | 作用 |
|---|---|---|---|
| **A0** | `cache` 模式，今日行为 | 0 | baseline |
| **A1** | `buffer_only` 模式 | 0 | **免疫参照**：\( \rho \equiv 1 \) 由构造保证（§2.8）。用于证明判据本身有效，同时暴露其代价——跨请求 L2 复用消失，端到端命中率在多轮场景下反而更低 |
| **A2** | A0 + R1 | ≈60 行 | 授予生命周期单独的效果（根因一） |
| **A3** | A0 + R1 + R3-a | ≈140 行 | 加队列序仲裁（根因二的竞争面收敛） |
| **A4** | A0 + R1 + R3-a + R3-b + R2 + R4 | ≈260 行 | 完整方案 |
| **A5** | A0 + 入队时 force 重预取 | ≈40 行 | **strawman 臂**：治标方案。预期 \( \rho \to 1 \) 但 IO 放大与 L3 读带宽随并发线性恶化，且高并发下重预取本身也会触发 `evict_host`，可能出现 \( \rho \) 不收敛 |
| **A6** | A0 + 整树 host 加锁 | ≈30 行 | **粗锁臂**：给 host radix tree 全局加锁禁止淘汰。预期 \( \rho \equiv 1 \) 但 host 池可用容量骤降，写回 staging（`hiradix_cache.py:883`）分配失败率上升，高并发下准入被 wedge（正是 `pipeline.py:240-263` 注释警告的 “pool full, nothing retractable”） |
| **A7** | A0 + `--hicache-storage-prefetch-policy wait_complete` | 0（改开关） | **既有旋钮臂**：验证 F-e。预期 \( \rho \) 仍 < 1，因为 `wait_complete` 只在 op 仍留在 `ongoing_prefetch` 时门控准入，而 \( t_{\text{land}} \) 处 `del self.ongoing_prefetch[request]`（§4.5.1）使门控恰好在窗口开启的瞬间失效。**若 A7 使 \( \rho \equiv 1 \)，本文主张被证伪** |
| **A8** | A0 + PEEK 式软保护打分淘汰 | ≈120 行 | **最强竞争者的可运行代理**：把 host 淘汰候选的排序从 LRU 改为按 \( \mathtt{evict\_score}(b) = \max_{v \in \mathtt{ancestors}(t_b)} \mathtt{pending\_count}(v) \cdot \mathtt{depth}(v) \) 降序淘汰（即优先淘汰低分块）[cite:13]。预期显著改善但不彻底，且**截断率与 A0 无差异**（打分不改变 alloc 成败），从而量化「软保护 vs 硬保护」的差距并定位 F-f 的相变点 |

A5 与 A6 是必须存在的两臂，因为它们对应被否决的两个解法。审稿人会问「为什么不直接重预取 / 直接加锁」，而这两臂给出的是定量答案而非论证性答案。A6 尤其关键：它与 A4 的差别不在 \( \rho \)（都为 1.0），而在 **host 池可用容量**与**准入前进性**——这正是 \( G_{\max} \) 与 C3 断言（§5.7）的价值所在。

**A7 与 A8 各自封堵一条审稿路径。** A7 的成本为零（只改一个开关），却是 F-e 判别的唯一实验证据；如果本文不跑 A7，审稿人只要指出「把 `wait_complete` 打开就行了」就能拒稿。A8 的成本较高，但它是本文面对 PEEK 时唯一诚实的回应方式：与其在相关工作里论证差异，不如把对方的机制实现为一个受控臂，让数据说话。A8 的实现需如实标注两点局限：其一，PEEK 的完整机制包含增量 radix tree over pending queue、dual-walk 最长前缀匹配与 pioneer-first 准入 [cite:13]，A8 只移植其**淘汰钩子**这一半；其二，\( \mathtt{pending\_count}(v) \) 在 A8 中用淘汰时刻对 `waiting_queue` 的一次扫描近似，而非维护增量结构，因此 A8 是 PEEK 的**下界代理**而非等价实现 `[unverified: 等待队列中的请求是否在淘汰时刻已持有可用的 last_node / 前缀匹配结果，本次未核实]`。这两点局限必须在论文中写明，否则 A8 的负面结果会被读作对 PEEK 的不公平比较。

### 7.5 消融

| 编号 | 消融 | 检验的假设 |
|---|---|---|
| B1 | R1 去掉 \( G_{\max} \) 钳制（无限钉住） | 反死锁上界的必要性；预期高并发下准入 wedge |
| B2 | R1 钉住但不迁移授予记录（只 `protect_host`，无 `landed_grants`） | 授予身份的必要性；预期 \( \rho \equiv 1 \) 但 C2 断言失效、异常路径出现 `RuntimeError` |
| B3 | R3-a 用 `continue` 而非 `break` | 队列序仲裁的必要性；预期 park 者被后续请求绕过，\( \rho \) 部分恢复但尾部请求仍失败 |
| B4 | R3-a 撤销策略改为 FIFO（撤 `queue_seq` 最小者） | LIFO 撤销的队头保护价值；预期队头饥饿 |
| B5 | R2 只加标志不改返回值（仍 `return True`） | 「截断即 park」与「截断即可观测」的分离；预期可观测但根因二的竞争面不收敛 |
| B6 | 打开 `enable_hicache_prefetch_retry=True`（不实现任何 R 层） | 既有重试机制能否自愈；预期**不能**，因为截断路径不调 `poll_miss` 且 `DONE` = “Never retry” |
| B7 | 扫描 `hicache_ratio`（host 池大小） | \( G_{\max} \) 绑定阈值；预期存在一个临界 ratio，低于它 cap-bind 率骤升、TTFT 反恶化 |
| B8 | 扫描 \( \gamma \) 与 \( \phi \) | 钳制公式的敏感度；给出推荐默认值 |
| **B9** | R1 的硬保护替换为 PEEK 式软保护打分（= A8 的机制叠加在 A2 之上） | **软/硬保护相变点（F-f）**：与 B2 联合，给出「打分能否替代 `host_ref_counter`」的定量答案。预期在低并发下两者接近、在 \( \lvert W \rvert \) 超过某个阈值后软保护击穿 |
| **B10** | 扫描 R3-b 的安全边界 \( \theta \in \{-\infty, -\widehat{\mathrm{TTR}}, 0, +\widehat{\mathrm{TTR}}, +2\widehat{\mathrm{TTR}}\} \) | **laxity 门控的价值与估计精度敏感性**：\( \theta = -\infty \) 端点等价于关闭 R3-b（退化为 A3），因此 B10 同时是 R3-b 的开关消融。预期 \( \theta \approx 0 \) 附近最优；若曲线对 \( \theta \) 极敏感，则说明 §5.2.3 的廉价 rank 型 TTU 代理不足，需换预算型估计（§8 风险 12） |

B6 与 A7 是审稿人前两个必问项的答案，且都是**零实现成本**的消融（只改一个开关）。B10 的 \( \theta = -\infty \) 端点让 R3-b 这一层自带一个内建的关闭开关，这使它不会成为「加了但没人信有用」的悬空设计。

---

## 8. 风险

| # | 风险 | 触发条件 | 缓解 |
|---|---|---|---|
| 1 | **钉住减少可淘汰 host 内存，别处分配失败** | 写回 staging 路径 `write_backup`（`hiradix_cache.py:883`）在 `host_indices is None` 时调 `evict_host(len(node.value))`；若可淘汰集合被钉住页占满，写回失败 | \( G_{\max} \) 的 \( \phi S_{\text{host}} \) 项显式为写回 staging 保留配额；上报写回分配失败率作为一等指标 |
| 2 | **cap 在高并发下频繁绑定 → 授予大量 park → 预取吞吐下降 → TTFT 反而变差** | \( N \) 大到 \( \sum \text{granted\_tokens} \) 触及 \( G_{\max} \) | 上报 cap-bind 率（`_grant_cap_busy` / `_grant_cap_skips`）；R3-a 的队列序保证 park 者按序获得内存而非随机；R3-b 从另一侧压低 park 总量（缩短 \( \overline{D} \)，抬高 \( G_{\max}/\overline{D} \) 上界，§5.2.2）；B7/B8 消融给出 ratio 与 \( \gamma \) 的推荐区间，B10 给出 \( \theta \) 的推荐值 |
| 3 | **死锁：准入余量被 LANDED 授予吃光，且无可 retract 的对象** | \( G_{\max} \) 配置错误（未减 \( L_{\max} \)），或 retract 路径在 cache 模式下不存在 `[unverified: cache 模式的准入 retract 是否覆盖已钉住的 host 页，本次未逐行核实]` | \( G_{\max} \) 公式的 \( S_{\text{host}} - L_{\max} \) 项；\( G_{\max} = 0 \) 时降级为「不钉住」（退回今日行为）；`defer_rounds` 上界 32 强制 REVOKED；C3 断言在 debug 构建中持续检查 |
| 4 | **TP 分歧导致集体通信挂起** | 授予状态转移发生在非同步点，或各 rank 的 `queue_seq` 不一致 | R1–R4 只在既有的 `prefetch_hits_sync_groups` / `prefetch_completion_sync_groups` 上变更语义（§5.6）；`queue_seq` 来源于 `len(waiting_queue)`，各 rank 天然一致；C4 断言用 all-reduce MIN/MAX 相等性验证；R3-b 的 \( \widehat{L}_r \) 判定复用 `_can_terminate_prefetch` 既有的 TP MAX all_reduce（§5.2.3），不引入新的分歧源 |
| 5 | **双减 bug：`release_host()` 抛 `RuntimeError`** | 同一授予被两条清理路径注销（例如 revoke 与强制清理竞争） | `attempt` 键控（沿用 `pipeline.py:299` “Attempt-keyed locks” 先例）；C1 断言；`release_host()` 的 `RuntimeError`（`radix_cache.py:302`）本身就是兜底检测 |
| 6 | **与 `_prefetch_after_device_hit_loss`（`scheduler.py:3300-3327`）交互产生双授予** | R4 的 TAIL 授予与该函数的重发起路径同时生效 | `attempt` 键控使 stale completion 无法解锁新尝试；TAIL 继承原 `queue_seq` 但分配新 `attempt` |
| 7 | **停机提前返回，钉住页被 `try/except: pass` 路径清理** | `is_idle()` 未包含 `landed_grants` 判定 | §5.6 明确列为必须连带修改项；对称于 `pipeline.py:303-311` 已有的 `staged_prefetches` 判定 |
| 8 | **审稿人质疑「为何不直接用 buffer_only」** | — | §2.8 已备答：语义层（放弃 L2 = 放弃跨实例复用）+ 工程层（四条硬前置，含不能跑在 decode 实例上）；A1 臂给出定量代价 |
| 9 | **审稿人质疑「仓库已有 `session_ref` 会话保留，为何还要新机制」** | 官方 blog 明写 “Session retention applies to L1/L2”[cite:22]，且 `session_ref_tracker.py` 确实存在并在淘汰侧生效 | §4.5.3 三条排除理由：注册时机在**请求完成**路径（`unified_radix_cache.py:1100`，`register_session_ref` 要求 `req.last_node is not None`）；生命周期是**会话**（直到 `/close_session`）而非授予（\( t_{\text{land}} \to t_{\text{admit}} \)）；性质是**软排序**而非硬保护（`_session_lru_predicate` 只改候选顺序，`unified_tree_core.py:455-458` 对 FULL 组件直接返回 `None`）。同时把 `component_protected_size_` 与 `_dec/_advance/_recede_session_coverage` 识别为 R1 的现成载体（§4.5.3），避免「重复造轮子」的读感 |
| 10 | **Rust TreeCore 后端下 R1 不生效** | `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`；官方称 Rust 侧持有 “per-component lock accounting” 与 “eviction walks”[cite:22] | §4.5.4 + §7.1(b)：显式固定 `python`（默认值，`environ.py:671-672`）并写入 artifact 可复现性声明；Rust 后端的等价改动列为 future work，并注明该扩展当前仅支持 Linux（`rust_extensions/torch_build.py:37`）。官方同时称 “Session-aware caching continues to use the Python tree core”[cite:22]，即 Rust 后端目前也**不覆盖**会话保留这一相邻机制，本文的范围限制与上游一致 |
| 11 | **本文全部行号会随上游演进而失配** | 工作副本无 `.git`，`version.py` 动态导入，无法用 commit hash 锚定；PyPI 最新发布为 2026-10-01，且 “UnifiedRadixTree is now the default”[cite:23]，说明上游仍在快速演进（含 “stability fixes across cascade eviction, tombstone replay, and partial-match paths”） | 投稿前必须按目标 commit 重新校验「代码行号索引」全表（Sources 节）；每条引用改为**函数名 + 行号双锚**，函数名比行号稳定；在 artifact 中固定 commit hash 与 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` |
| 12 | **R3-b 的 TTU 估计精度不足，导致过晚发放** | 廉价 rank 型代理 \( \lceil k_r/\hat{a}\rceil \) 继承了 TempoKV Figure 2 所批评的全部弱点（同一 rank 下余量从提前 2.67 s 到滞后 6.02 s）[cite:12] | B10 的 \( \theta \) 扫描直接量化敏感性；`defer_rounds` 上界 32 强制发放（§5.2.3 钳制表）；若 B10 显示敏感，换用基于 `PrefillAdder` 干跑的预算型估计 `[unverified]`；最坏情况下 R3-b 可被完整关闭（\( \theta = -\infty \) 退化为 R3-a）而不影响 R1 的正确性 |
| 13 | **A8 被读作对 PEEK 的不公平比较** | A8 只移植淘汰钩子，不含增量 radix tree over pending queue、dual-walk 前缀匹配与 pioneer-first 准入 [cite:13] | §7.4 显式标注 A8 为 PEEK 的**下界代理**而非等价实现，并写明 `pending_count` 的一次扫描近似；论文中同时引用 PEEK 自报的数字（SGLang 上 up to 3.0× cache hit / 7.9× TTFT）作为参照，而非只给 A8 的实测；结论只对「软保护能否替代 `host_ref_counter`」负责，不对 PEEK 整体优劣负责 |

风险 3 中的 `[unverified]` 项是投稿前必须补做的读码工作：需确认 cache 模式的准入路径在 host 池耗尽时是否存在可撤销已钉住页的 retract 机制。若不存在，\( G_{\max} \) 的余量项就从「保守」升级为「唯一防线」，需在论文中如实陈述。

风险 9–13 的共同特征是**不改变方案的可行性，只改变论文的写法**：9 与 10 要求把「已有机制为何不覆盖」写成正面论证而非脚注（§4.5 已完成）；11 是一条工程纪律；12 与 13 要求 R3-b 与 A8 在设计上就自带「可关闭 / 可标注局限」的退路，这一点已在 §5.2.3 与 §7.4 落实。

---

## 9. 投稿定位

- **R1 单独**（≈60 行，\( \rho \) 判据确定性，A2 + A7 两臂即可支撑）→ HotStorage'27 / HotOS'27。卖点是一个**位置类**（position paper）主张：tier 内跨层预取需要所有权语义，而现有系统只有计数语义；证据是同一仓库的两个分支已经分别实现了两者，且实现 `grant` 的那个分支免疫、实现 `reserve` 的那个分支故障。A7 的加入让这个 position 更硬：既有旋钮的三个取值都不能覆盖 \( W \)（§4.5.1），说明缺口不是「参数没调对」而是「这个语义不存在」。
- **R1–R4 全评估**（九臂 + 十消融 + 三张主图）→ ATC'27 / EuroSys'27。卖点是完整的授予制度：生命周期、队列序仲裁、laxity 门控、反死锁上界、反饥饿上界、截断可观测性，每一项都有仓内先例支撑其可实现性，每一项都有对照臂或消融证明其必要性；Figure 3 的 Little 定律验证给出一个不依赖端到端指标的、自足的定量结论。
- **在先工作 TempoKV 的处理策略**。TempoKV 于 2026-09-28 提交 [cite:12]，早于本文的复核时点，因此它是**在先工作而非并发工作**，必须正面引用而非回避。建议的定位写法：TempoKV 建立了「预取承诺是一个有价格的时序决策」这一框架，并给出了承诺**过早**一侧的完整解（其 protected fast-tier byte-time 降幅 63–91%）；本文处理承诺**释放过早**一侧，并指出在一个连保护都不存在的 tier 上，时序优化无从谈起——**先有所有权，才谈得上所有权的时机**。这句话可直接用作 abstract 的收尾。
- **竞争者 PEEK 的处理策略**。PEEK 是系统级竞争者且已在 SGLang 上真实实现 [cite:13]，因此不能只做相关工作层面的论证，必须给 A8 臂。两种实验结果都可发表，这是设计上刻意保留的退路：
  - 若 A8 显示软保护在 \( \lvert W \rvert \) 超过某阈值后击穿 → 结论是「PEEK 的淘汰钩子与本文的授予契约互补：前者减少候选，后者保证不被摘」，本文主张不变。
  - 若 A8 显示软保护已足以使 \( \rho \equiv 1 \) → 本文主张收窄到**截断侧**（根因二），因为打分不改变 alloc 成败这一点是结构性的（§3.4.2 第四条），不受实验结果影响。
- **与 Continuum 的关系可作为一节 “Retention across tiers”**。Continuum 在 GPU 层用 TTL pin + program-level FCFS [cite:14]，本文在 host 层用授予契约 + 队列序仲裁，两者在其 Table 1 式的三维度自评上都得 ✓✓✓（§3.4.3），但作用对象与失效模式完全不同。若同一会议同时收到两篇，这一节是「不是同一件事」的第二个证据（第一个是 §7.2 的负载阶跃图）。
- 与 [cite:11] 的关系可作为 discussion 一节：那篇讨论「恢复还是重算」的代价权衡，本文指出**在授予活不到被权衡的那一刻时，代价函数根本没有输入**。这是对偶而非竞争。

---

## Sources

**文献**

- [cite:1] LM-CXD / *Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management*. Chung, Noh, Kim, Hwang, Kim, Kim, 2026. https://arxiv.org/abs/2609.26828
- [cite:2] *A CXL Memory Rack for Multi-Turn LLM Serving*. arXiv 2607.18141v3. https://arxiv.org/abs/2607.18141
- [cite:3] ECHO: *Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs*. OSDI'26. https://www.usenix.org/system/files/osdi26-liu-guangda.pdf
- [cite:4] OasisKV: *Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching*. arXiv 2608.08097. https://arxiv.org/abs/2608.08097
- [cite:5] Talaria: *Session-Aware Serverless Serving of Hundred-Billion-Parameter LLMs*. arXiv 2607.17181. https://arxiv.org/abs/2607.17181
- [cite:6] Cascade: *Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving*. arXiv 2608.06557. https://arxiv.org/abs/2608.06557
- [cite:7] ExPAND: *CXL Topology-Aware and Expander-Driven Prefetching*. arXiv 2505.18577. https://arxiv.org/abs/2505.18577
- [cite:8] Mooncake: *A KVCache-centric Disaggregated Architecture for LLM Serving*. arXiv:2407.00079. https://arxiv.org/abs/2407.00079
- [cite:9] Ao et al., arXiv 2606.15555（极限环型故障，吞吐损失达 50%）. https://arxiv.org/abs/2606.15555
- [cite:10] Metronome, arXiv 2607.02640（metastable + silent，14/20 vs 0/20）. https://arxiv.org/abs/2607.02640
- [cite:11] “To Restore or To Recompute”, HotStorage'26 `[unverified: 本次未取得可引用 URL]`
- **[cite:12]** TempoKV: *Timing-Aware KV Staging Commitment for SSD-Backed Inference*. arXiv 2609.35065v1, Samsung Semiconductor, 2026-09-28. https://arxiv.org/abs/2609.35065 `[unverified: 标题依检索页与摘要自述拼接，投稿前需与 arXiv 元数据逐字核对]`
- **[cite:13]** PEEK: *Proactive prEfix matChing and sCheduling for LLM serving*. Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zhen Zheng. arXiv 2607.02525（v1 2026-05-10 / v2 2026-09-12），26 pp., 21 figures, 21 tables. https://arxiv.org/abs/2607.02525
- **[cite:14]** Continuum: *Efficient Execution of Agentic Workflows with Dynamic KV Cache Retention*. Hanchen Li*, Runyuan He*, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Alvin Cheung, Joseph Gonzalez, Ion Stoica. arXiv 2511.02230；技术报告 UCB/EECS-2026-234. https://arxiv.org/abs/2511.02230
- [cite:15] TierKV: *Prefetch-Aware Three-Tier KV Cache Management*. https://idea2paper.org/assets/papers/tierkv.pdf `[unverified: 匿名预印本，非同行评审，评估基于离散事件模拟器而非真实实现]`
- [cite:16] *Mitigating KV Cache Competition to Enhance User Experience in LLM Inference*. arXiv 2503.13773. https://arxiv.org/abs/2503.13773
- [cite:17] TokenCake. arXiv 2510.18586v5. https://arxiv.org/abs/2510.18586
- [cite:18] HillInfer. arXiv 2602.18750. https://arxiv.org/abs/2602.18750
- [cite:19] *Predictive Multi-Tier Memory Management for Long-Context LLM Inference*. arXiv 2604.26968. https://arxiv.org/abs/2604.26968
- [cite:20] *Where Should the KV Cache Live?*. arXiv 2609.16215. https://arxiv.org/abs/2609.16215
- [cite:21] *HBF Sucks?*. arXiv 2608.11668. https://arxiv.org/abs/2608.11668

**SGLang 官方资料**

- **[cite:22]** SGLang Team, *Unified Radix Cache in SGLang: one tree for hybrid model prefix caching*, SGLang Deep Dive, 2026-09-23. https://www.sglang.io/blog/unified-radix-cache （本文的官方 benchmark 配置、Session-aware eviction 描述、Rust TreeCore 分工说明均出自此页）
- **[cite:23]** SGLang 官网首页，*UnifiedRadixTree is now the default* 与 HiCache + UnifiedRadixTree 变更说明（PR #23316 #23391 #24691）。https://www.sglang.io/ ；PyPI `sglang` 最新发布 2026-10-01. https://pypi.org/project/sglang/
- **[cite:24]** SGLang 官方文档，*HiCache Best Practices*（`--hicache-storage-prefetch-policy` 三值）。https://docs.sglang.io/docs/advanced_features/hicache_best_practices ；另 *HiCache System Design and Optimization*（L3 命中超过阈值即触发预取，默认 256 tokens）。https://sgl-project.github.io/advanced_features/hicache_design.html

**代码行号索引**（路径根 `python/sglang/srt/`，全部行号均在目标代码库上逐一读码核实）

| 文件 | 行 | 内容 |
|---|---|---|
| `mem_cache/unified_radix_cache.py` | 2531-2537 | `_prefetch_occupied_span`，reserve/grant docstring |
| 同上 | 2689-2803 | `_try_alloc_storage_hit` 全文 |
| 同上 | 2738-2751 | 授予分配、`evict_host`、截断分支 |
| 同上 | 2793-2803 | 截断后 `return True`（记为成功） |
| 同上 | 2805-2812 | `_drain_and_alloc_storage_hit`，`break` 与 “buffer only” 门 |
| 同上 | 2205-2233 | cache 完成路径，“until admission proves” 注释 |
| 同上 | 2636-2654 | revoke 路径与 post-alloc 断言 |
| 同上 | 2440-2447 | `pop_prefetch_loaded_span` |
| 同上 | 1311-1314 | buffer 模式 `evict_host` 直接返回 0 |
| 同上 | 424-429 | buffer_only 仅支持 FULL/SWA 统一树 |
| 同上 | 951 / 956 | `inc_host_lock_ref` / `dec_host_lock_ref` |
| 同上 | 2056-2059 | cache reserve / buffer grant 分界注释（§4.2 支点） |
| 同上 | 2061-2087 | `_can_terminate_prefetch` 三值分支；`_prefetch_timeout_check_linear_func`（R3-b 的 \( \widehat{\mathrm{TTR}} \) 直接复用此线性模型） |
| 同上 | 2089-2109 | `check_prefetch_progress` 早退路径（F-e 与 A7 臂的代码支点） |
| 同上 | 2215 | cache 完成路径 `del self.ongoing_prefetch[request]`，门控在 \( t_{\text{land}} \) 解除 |
| 同上 | 2702-2706 | buffer 分支 “wait_complete keeps gating admission” 注释及其 “The op stays in ongoing_prefetch” 前提 |
| 同上 | 1100 | `register_session_ref(req)` 的唯一调用点（请求**完成**路径） |
| `mem_cache/radix_cache.py` | 262-302 | `TreeNode` 字段、`evicted`、`protect_host` / `release_host` |
| `mem_cache/hiradix_cache.py` | 532-555 | 强制清理路径，`try/except: pass` + 钳制 |
| 同上 | 1222-1235 | `_update_host_leaf_status`，不看 `host_ref_counter` |
| 同上 | 1387-1420 | `evict_host`，候选、跳过条件、摘除 |
| 同上 | 1422-1497 | `load_back` |
| 同上 | 1455-1456 | 准入时 `protect_host()`（正确用法示范） |
| 同上 | 1816-1842 | `_handle_prefetch_result`：pop → insert → free → release → 递减 |
| 同上 | 1902-1908 | `pop_prefetch_loaded_tokens`，revoke 才返回 0 |
| 同上 | 1910-1941 | `match_prefix`，`host_hit_length` 累加 |
| 同上 | 638 / 883 | 另两处 `evict_host` 调用（授予路径、写回路径） |
| 同上 | 2009-2050 | `_insert_helper_host`，新生叶子入候选集合 |
| `mem_cache/unified_cache/storage_attachment.py` | 388-410 | unified revoke 路径与钳制 |
| `mem_cache/buffer_mode/pipeline.py` | 12-24 | TP-lockstep 契约，“no runtime verification” |
| 同上 | 100-104 | `_StagedPrefetch` docstring，“parked until prefill admission” |
| 同上 | 240-263 | `anchor_lock_cap_tokens`，“wedge admission permanently” |
| 同上 | 266-268 | `max_staged_admission_defers = 32` |
| 同上 | 270-301 | `reset()` 账本，“Attempt-keyed locks” |
| 同上 | 303-311 | `is_idle()` 含 `staged_prefetches` |
| 同上 | 519-523 | `live_cap` 三项式 |
| 同上 | 757-820 | `try_lock_anchor` 五值枚举 |
| 同上 | 954-958 | `_refetch_staged`（buffer 专属自愈） |
| 同上 | 1004 / 1151 / 1280 / 1307 | buffer 侧记账递减点 |
| `managers/cache_controller.py` | 268-271 | `HICACHE_LOAD_POOL_USAGE_FRACTION=0.9`、`HICACHE_WRITE_STAGING_POOL_FRACTION=0.2` |
| 同上 | 592-615 | `prefetch_capacity_limit`、`prefetch_tokens_occupied` 初始化、两个 gloo 同步组 |
| 同上 | 1255-1273 | `prefetch_rate_limited()`，“overstate” 注释与 `todo` |
| `managers/scheduler.py` | 3135-3138 | “one-shot judgment … never retried” |
| 同上 | 3248-3265 | `_sync_prefetch_blocked_give_up`，`ReduceOp.MIN` |
| 同上 | 3300-3327 | `_prefetch_after_device_hit_loss` 及其三重遮蔽 |
| 同上 | 3339-3340 | `_prefetch_kvcache(req)` 在 `waiting_queue.append(req)` 之前 |
| 同上 | 3685-3702 | 事件循环顺序 |
| 同上 | 4048-4062 | `pop_prefetch_loaded_span` → `storage_hit_length` → `init_next_round_input`（\( \rho \) 探针位置） |
| 同上 | 5051-5055 | 停机消费 `is_idle()` |
| `managers/schedule_batch.py` | 1536-1602 | `init_next_round_input` 调 `match_prefix` |
| `mem_cache/schedule_batch.py` | 849-864 | `PrefetchStatus.DONE`，“Never retry” |
| `mem_cache/storage_prefetch.py` | 11-74 | `StoragePrefetchRetries` docstring、队列序遍历、队头保护 |
| `mem_cache/session_ref_tracker.py` | 1-118 | 全文；模块 docstring、`register_session_ref`、`release_radix_session`、`_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192`（§4.5.3） |
| `mem_cache/unified_tree_core.py` | 455-458 | `_session_lru_predicate`，FULL 组件直接返回 `None`（R1 的挂载点） |
| 同上 | 480-481 | `component_evictable_size_` 与 `component_protected_size_` 并列初始化（\( G_{\max} \) 的现成记账载体） |
| `mem_cache/components/full.py` | 45-60 | `session_ref_eviction_strategy` 懒绑定与 `_ensure_eviction_strategy` |
| 同上 | 62-102 | `_dec_session_coverage` / `_advance_session_coverage` / `_recede_session_coverage` |
| `mem_cache/components/swa.py` | 148-151 | 等价 session coverage 实现 |
| `mem_cache/components/mamba.py` | 87-95 | 等价 session coverage 实现 |
| `arg_groups/fields/memory.py` | 102-124 | `--hicache-host-memory-mode` 与 `--hicache-write-policy` 官方定义 |
| 同上 | 166-172 | `hicache_storage_prefetch_policy` 三值，默认 `timeout`（§4.5.1 / A7 臂） |
| 同上 | 177-185 | `enable_hicache_prefetch_retry` 官方 help 文本（F-e 的官方级自证） |
| 同上 | 177-209 | 重试三开关默认值 |
| `arg_groups/hicache_hook.py` | 79-86 / 185-229 | `hicache_ratio` 默认值、mode 校验、buffer_only 四条硬前置 |
| `mem_cache/registry.py` | 244-255 | buffer_only 要求 `UnifiedRadixCache` |
| `environ.py` | 784 | `SGLANG_HICACHE_PREFETCH_BLOCKED_WAIT_S = 60.0` |
| 同上 | 671-672 | `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("python")`（默认后端，§4.5.4 / §7.1(b)） |
| `rust_extensions/torch_build.py`（**仓库根相对路径，不在 `python/sglang/srt/` 下**） | 37 / 44 / 56 | Rust TreeCore “currently supports **Linux only**”、需先装 PyTorch、PyTorch 版本区间 |

---

## 未验证项汇总

1. **cache 模式的准入 retract 是否覆盖已钉住的 host 页**（§8 风险 3）。若不存在，\( G_{\max} \) 的余量项是唯一防线，需在论文中如实陈述。**投稿前必须补做。**
2. **\( \rho \) 探针的口径对齐**（§6.3）。`req.storage_hit_length` 是纯 L3 来源计数，`req.host_hit_length` 含 L2 既有命中；严格版本需用 `storage_hit_start` 做区间交集。实现时需验证。
3. **`_insert_helper_host` 在 unified 分支的对应实现行号**。本文引用的是 `hiradix_cache.py:2009-2050`；unified 分支的等价路径未逐行核实。
4. **[cite:11] 未取得可引用 URL**，HotStorage'26 的具体论文标识待补。
5. **`host_insert_dropped` 的完整触发条件**。本文只核实它被用于日志区分 `dropped` / `success`（`unified_radix_cache.py:2225-2233`），未追踪其置位路径。
6. **\( G_{\max} \) 中 \( \gamma \) 与 \( \phi \) 的推荐默认值**待 B8 消融给出；本文只论证了 \( \gamma = \phi = 0.5 \) 是既有代码已选定的值。
7. **多实例场景下 \( \rho \) 是否会因跨实例池协议引入额外损失**。本文全程在单实例 host 层内论证；跨实例池协议不在本文范围内。
8. **A5（force 重预取）在极高并发下是否会出现 \( \rho \) 不收敛**。§7.3 给出的是推测机制（重预取本身触发 `evict_host`），需实验证实或证伪。
9. **`rank_consensus` 装饰器（`unified_radix_cache.py:3332`）对 `host_hit_length` 的一致性约束**是否会影响 \( \rho \) 探针在多 rank 下的读数。已知该装饰器带 `same_params=["params.host_hit_length"]`，但未核实其分歧处理策略。

**其余未验证项（10–16）**

10. **[cite:12] TempoKV 的标题与 arXiv 元数据需逐字核对**。本文标题依检索页与摘要自述拼接；另需确认该预印本（2026-09-28）是否已有会议接收版本，以便引用正式版。**这是本文最强撞题对象，引用信息必须准确。**
11. **[cite:15] TierKV 为匿名预印本且评估基于离散事件模拟器**，非真实实现。正文已按「同期预印本并列」定位，不作为方法学依据。
12. **R3-b 的 \( \widehat{\mathrm{TTU}} \) rank 型代理精度**（§5.2.3 / §8 风险 12）。\( \lceil k_r/\hat{a}\rceil \cdot \hat{T}_{\text{round}} \) 未考虑请求间的 token 数异质性与 chunked prefill 的分段准入；B10 的 \( \theta \) 扫描是唯一的定量判据。若敏感，需换用基于 `PrefillAdder` 干跑的预算型估计。
13. **`component_protected_size_`（`unified_tree_core.py:480-481`）的现有更新点未核实**。本文只确认它与 `component_evictable_size_` 并列初始化，可作为 \( G_{\max} \) 的记账载体；其现有语义、增减时机与是否已被 session 保留占用，需在实现 R1 时逐一读码确认，避免与既有账目冲突。
14. **A8 的 `pending_count` 一次扫描近似是否足以代表 PEEK**（§8 风险 13）。PEEK 的打分建立在 pending queue 上的**增量 radix tree** 与 dual-walk 之上，A8 不含这两者；两者的 \( \rho \) 差异中有多少来自「软 vs 硬」、多少来自「近似 vs 精确匹配」，需在论文中如实分离，或补充一个带增量树的 A8' 臂。
15. **Rust TreeCore 后端下 R1 的等价改动**（§4.5.4 / §8 风险 10）。官方称 Rust 侧持有 “per-component lock accounting” 与 “eviction walks”，本文全部改动只在 Python 后端论证；Rust 侧的对应实现路径未读码，列为 future work。该扩展当前仅支持 Linux。
16. **官方 benchmark（[cite:22]）的崩塌是否可归因于本文机制**。官方只给出命中率曲线，未给出 host 池占用、`evict_host` 调用次数或截断计数；Figure 1 的两个内部计数器是本文自行添加的观测，需要复跑官方配置才能把「Round 40–55 的 L1+L2 ≈0%」与 \( W \) 窗口的淘汰建立因果链，而非仅时间上的共现。

**已核实项**：[cite:8] Mooncake 的 arXiv 标识符 `https://arxiv.org/abs/2407.00079` 已核实有效（全文另可经 `https://arxiv.org/html/2407.00079v1` 访问），因此 Sources 中该条不带 `[unverified]` 标记。
