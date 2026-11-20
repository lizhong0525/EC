# Gold 准备与指南参考层验收记录

日期：2026-10-23；实现：`guideline-reference-v1.0.0`。
协议：[指南局部参考协议](GUIDELINE_REFERENCE_PROTOCOL_V1.md)；
操作：[指南参考与 Gold 手册](GUIDELINE_REFERENCE_OPERATIONS_V1.md)。

## 本轮实际改动

复用已有研究 Schema、SQLite、expert-a/expert-b/adjudicator 页面、
逐节点锁定、字段级分歧、裁决及 Gold 冻结门禁，没有另造一套专家系统。

新增：

- `research/reference_contracts.py`：来源目录、条款、局部规范、可见原文片段、
  条款适用性、真实整理者/AI 草稿来源、规则复核与节点复核契约。
- `research/references.py`：独立 SQLite 账本、完整性审计、来源快照、逐节点映射、
  第二人复核、不可覆盖纠错、状态分级与单独的局部 Silver 评分。
- `research/reference_cli.py` 与原研究 CLI 的 `references` 子命令。
- 来源目录、操作手册、独立版本协议、测试。

指南记录不写入专家 judgments/attestations 表；未改旧 Judgment 契约、
原专家手册、Judge 提示词、历史评分、历史实验报告或旧试标库。
专家优选动作在原 v2 中仍为必填；新的局部 Silver 契约允许留空，
没有为兼容新路径而静默改变旧语义。
当前文件夹没有 Git 元数据，因此本记录不宣称进行了 Git diff 或提交。

## 实际创建的产物

`artifacts/research/guideline-sources-20261023-v1/`

- 固定三个来源的抓取状态、时间和文件哈希。
- 两个 WHO 页面成功取得，均标记 `FETCHED_NOT_EXPERT_REVIEWED`。
- NICE 推荐页实际返回 HTTP 403，标记 `UNAVAILABLE`，未绕过访问限制。
- WHO 出版介绍页只作为来源信息；未宣称核对整本 mhGAP 第三版。
- 本次没有任何人被程序登记为已完成来源/适用性复核。

`artifacts/research/gold-readiness-20261023-v1/`

- 新建 20 个 legacy Train 分层开发病例、每例 14 节点，共 280 节点。
- `pilot-manifest.json` 保存风险轨迹、生态、表达方式的覆盖。
- 专家库：280 个 UNLABELED、无身份核验、无真实专家记录、Gold 为 0。
- 独立指南库：Gold 0、Silver 0、Unresolved 280。
- 指南创建事件含目录、来源正文、手册正文/摘要和数据摘要；当前只有创建事件。
- `private/guideline-v1/schemas/`：四种新增输入契约。
- `private/guideline-v1/first-node-preview.json`：仅第一个当前节点；
  `ai-preview-not-a-human` 只是导出预览的占位身份，没有提交任何映射。
- 专家标注访问凭据只在私有文件中，未输出到对话或本记录。

该包是流程准备，不是独立盲测，也不是已完成人类标注的数据集。

## 实际执行的验证

### 工程回归

```powershell
.research-venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/research/gold-readiness-tests-20261023-v1.xml
```

结果：**560 passed，2 failed，562 项，总耗时 347.45 秒**。不能宣称全量通过。
XML：`artifacts/research/gold-readiness-tests-20261023-v1.xml`。
其中新增指南模块 11 项全部通过。

两个失败已在未改动的既有测试中单独复现：

1. `test_benchmark_catalog.py::test_catalog_lists_only_valid_non_symlink_runs`：
   Windows 创建符号链接时 `WinError 1314`，当前进程没有所需权限。
2. `test_benchmark_log_import.py::test_json_import_creates_observable_only_expert_draft`：
   内容检查均已执行到权限断言；Windows 返回 `0666`，旧测试要求 POSIX `0600`。

没有跳过、删除或改写这两个测试来取得全绿；没有修改相关生产模块。
这也提示：`private` 目录名、应用层路径限制或 `chmod(0600)` 不等于 Windows 账户隔离。
正式分发给不同人员前须由管理员另行设置 NTFS ACL、独立账户和受控浏览器环境，
并在相应平台重验。不宣称已完成 OS 级保密或完整生产权限系统。

相关三模块测试在初次实现阶段运行 69 项通过，随后新增两项防元数据滥用测试，
已包含在上述全量结果中。最终以全量 XML 为准，不把重复测试加算成独立测试数。
新增模块及 CLI 的 Ruff 检查通过；三个新增生产文件的 Mypy 检查通过。

### 评分与来源隔离

新增测试覆盖：

- 取得原文不等于规则已获项目批准；未取得/不匹配的原文不能启用规则。
- 当前观察摘要、可见精确引用、未来引用、条件未知、无来源字段及跳节点校验。
- 作者不能自审；去除首尾空白后仍检查身份；AI 作者不能靠纠错改成“人类”。
- 未经复核的映射和 AI 草稿不生成 Silver；追加纠错使旧复核失效。
- 原映射和复核不可静默修改；哈希链可发现直接改动。
- Silver 状态不能被原 Gold 加载器接受。
- 改变 Silver 参考可改变其局部指标，不改变专家不可用状态或模拟恢复指标。
- 输出失败仍进入适用局部指标分母；字段缺失不补出风险等级、优选或时间窗。

### 真实本地端到端

实际执行 `references fetch-sources → references prepare → run-local →
references report`，并导出第一节点预览及初始专家状态。

- 4 个本地规则基线、280 节点/基线，共 1,120 个节点输出。
- 模型调用 0，没有调用付费模型或云服务，没有上传病例。
- `private/local-rules-v1/report.json`：原 v2 本地规则报告。
- `private/guideline-v1/report-001.json`：新增分栏报告。
- 四个策略的模拟恢复结果与原规则报告逐项一致。
- 专家指标均为 `UNAVAILABLE_NO_APPLICABLE_GOLD`。
- Silver 指标均为 `UNAVAILABLE_NO_REVIEWED_SILVER`，metrics 为 null。
- 冻结源码摘要与运行时当前源码匹配；missing_results 为 0。
- 实际调用 Gold freeze，按缺少三角色真实身份/资质记录拒绝；
  断言目标 Gold 文件不存在，专家审计头未变化。

所有角色/标签测试夹具仅存在于 pytest 临时目录，不混入上述正式任务包。

## 尚未完成

- 真实项目整理者的条款与临床到 AI 适用性复核；当前候选规则并未启用。
- 逐节点真实整理与不同人员复核。没有第二个人时必须继续保持待确认，
  不用虚构账号模拟“独立复核”。
- NICE 原文的合法可用快照与逐条核对。
- 两名真实专家与第三裁决者的资质核验、培训、盲标、一致性门禁与 Gold。
- 伦理、授权、隐私、OS 级访问控制和机构审批。
- 独立管理员收集的盲测材料、正式研究样本量规划、真实模型冻结实验。
- 本轮没有新增浏览器视觉验收：沿用旧页面，未修改前端。

本轮验收结论：**Gold 准备与指南 Silver 的可信流程已实现，并在本机验证主要路径；
真实 Gold 和人工确认的 Silver 尚未取得。**
