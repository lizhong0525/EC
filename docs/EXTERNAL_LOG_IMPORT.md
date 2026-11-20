# 外部合成事件日志导入契约

> 契约版本：`1.0`  
> 导入器版本：`external-log-import-v1`  
> 输出状态：仅供专家重标的草稿，不是 Benchmark 金标准

## 1. 用途与边界

`eco benchmark import-log` 把外部**合成**对话模拟器产生的可观察事件转换为
EcoCompanion 的脱敏、稳定 JSONL 草稿。导入器采用独立的中立契约，不依赖任何
上游项目包，也不包含针对某个上游源码结构的复制实现。

导入只映射：

- 合成人物的年龄组、语言、已披露背景和人格标签；
- 模拟时间；
- 当轮可见的社会生态事件；
- 合成学生当轮文本。

导入器不会推断或填充：

- 抑郁风险轨迹、风险等级或诊断；
- 隐藏世界状态；
- 表达风格或社会生态金标签；
- 证据码、可接受动作、优选动作或禁止动作；
- 转介等级、目标或干预窗口。

输出固定为 `PENDING_EXPERT_RELABEL`，且 `benchmark_eligible=false`。只有完成项目
标注手册规定的双专家独立标注、分歧裁决和质量检查后，才能由单独的受控流程转换
为正式病例。

## 2. 输入格式

### 2.1 单条记录

JSONL 的每个非空行必须符合 `ExternalEventLogRecordV1`：

```json
{
  "schema_version": "1.0",
  "record_type": "external_visible_event",
  "synthetic": true,
  "source_system": "example-simulator",
  "source_episode_id": "episode-001",
  "source_record_id": "turn-001",
  "sequence_index": 0,
  "simulated_time": "day-01-evening",
  "person": {
    "source_person_id": "synthetic-student-001",
    "age_group": "12-15",
    "language": "zh-CN",
    "disclosed_background": [
      "最近学习任务变多"
    ],
    "persona_tags": [
      "谨慎表达"
    ]
  },
  "visible_events": [
    "阶段测验临近"
  ],
  "text": "这周有点累。"
}
```

标识符只能是短的 ASCII 不透明 ID，不得放入姓名、手机号、邮箱、证件号、API
密钥或其他凭据。`synthetic` 必须为 `true`；V1 明确拒绝真人日志。

### 2.2 JSON 文档

`.json` 输入可以是：

- 一条 `ExternalEventLogRecordV1`；
- 一个记录数组；
- 一个 `ExternalEventLogDocumentV1` 封装：

```json
{
  "schema_version": "1.0",
  "document_type": "ecocompanion_external_visible_event_log",
  "records": [
    {
      "schema_version": "1.0",
      "record_type": "external_visible_event",
      "synthetic": true,
      "source_system": "example-simulator",
      "source_episode_id": "episode-001",
      "source_record_id": "turn-001",
      "sequence_index": 0,
      "simulated_time": "day-01-evening",
      "person": {
        "source_person_id": "synthetic-student-001",
        "age_group": "12-15",
        "language": "zh-CN"
      },
      "visible_events": [],
      "text": "今天想安静一会儿。"
    }
  ]
}
```

正式 JSON Schema 位于：

- `artifacts/schemas/ExternalEventLogRecordV1.schema.json`
- `artifacts/schemas/ExternalEventLogDocumentV1.schema.json`
- `artifacts/schemas/ImportedCaseDraftV1.schema.json`

## 3. 记录身份、顺序和冲突

一条源记录的逻辑身份是：

```text
(source_system, source_episode_id, source_record_id)
```

- 相同身份且内容完全相同：视为重复记录，安全去重；
- 相同身份但内容不同：视为来源冲突，导入失败；
- 同一 episode 出现两个不同记录但 `sequence_index` 相同：顺序有歧义，导入失败；
- 同一 episode 的人物信息不一致：导入失败；
- 输入记录顺序不影响输出；导入器按 episode 和 `sequence_index` 确定性排序；
- `sequence_index` 可以有间隔，输出草稿会重新编号为从 0 开始的连续 `step_id`。

JSON 对象中的重复字段名、未知字段、错误版本、空文本、过长输入和非 UTF-8 输入
都会快速失败。

## 4. 脱敏和可追溯性

导入器会在所有可见自由文本中确定性替换常见：

- API 密钥、Bearer Token 和 Authorization 值；
- 邮箱；
- 中国大陆手机号；
- 中国居民身份证号。

原始 `source_person_id`、`source_episode_id` 和 `source_record_id` 不进入输出；
输出只保存域分离 SHA-256 派生的稳定引用。`source_content_sha256` 基于脱敏后的
规范内容计算，用来发现内容变化，不用于恢复原始标识符。

自动正则脱敏不能证明任意自由文本都不存在间接身份信息。生成源仍不得写入真实
姓名、学校、地址、账号或可识别经历，正式进入标注流程前还必须由研究者复核。

## 5. 命令行

```bash
eco benchmark import-log synthetic_events.jsonl \
  --output artifacts/imports/expert_relabel_drafts.jsonl
```

首次成功会输出类似：

```text
status=created drafts=1 records=14/14 duplicates=0 redactions=0 output=...
```

对完全相同的逻辑输入再次运行时，输出字节保持不变并返回
`status=unchanged`。如果目标文件存在但内容不同，命令默认拒绝覆盖；确认需要替换
研究草稿时必须显式使用 `--overwrite`。

## 6. 输出契约

每个源 episode 产生一条 `ImportedCaseDraftV1`，包含：

- 脱敏可见人物信息；
- 脱敏的逐步事件和话语；
- 稳定的草稿、episode、事件和观察引用；
- 脱敏内容摘要；
- 固定的待专家标注字段清单。

输出结构没有 `HiddenState`、风险标签、动作标签或干预窗口值，因此不能被
`CaseSpecV1` 加载器误当作正式病例。导入结果与原始日志应分别保管；未经专家复核
的草稿不得加入 Train、Dev、Public Test 或 Hidden Test。
