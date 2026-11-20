# Windows 平台测试限制与处理

日期：2026-10-23
相关测试：`backend/tests/test_benchmark_catalog.py`、`backend/tests/test_benchmark_log_import.py`

## 1. 背景

此前全量测试在 Windows 上稳定出现 2 项失败，均来自平台能力差异，不是产品逻辑缺陷：

| 测试 | 失败原因 | 真实含义 |
| --- | --- | --- |
| `test_catalog_lists_only_valid_non_symlink_runs` | `WinError 1314`：当前账户无创建符号链接的权限（未开启开发者模式） | 无法在默认 Windows 账户下构造符号链接用例 |
| `test_json_import_creates_observable_only_expert_draft` | `stat.S_IMODE` 返回 `0o666`，断言 `0o600` 失败 | NTFS 上 `chmod` 只切换只读位，不表达 ACL |

关键点：**跳过或改写断言不能改变安全属性本身**。Windows 上的文件权限断言本来
就不是隐私保证；真正的隔离要靠独立账户与 NTFS ACL，这一点在
`docs/GOLD_READINESS_RESULTS_20261023.md` 与 `docs/TRUST_AUDIT_V2.md` 中已写明。

## 2. 本轮处理

### 2.1 拆分符号链接用例

原测试把“目录过滤”和“符号链接排除”混在一项里，符号链接创建失败会连带丢失
目录过滤的覆盖。现拆分为：

- `test_catalog_lists_only_valid_runs`：始终运行，覆盖合法/非法目录与 ignored 列表；
- `test_catalog_ignores_symlinked_runs`：仅在宿主允许创建符号链接时运行；
  不可用时 `pytest.skip` 并输出具体 `OSError`，不是静默通过。

符号链接排除是对“目录逃逸”的防御，属于安全属性；Windows 默认账户无法构造该
输入，因此该属性只能在 POSIX/CI 上验证。本机结论不能外推为“符号链接防御已验证”。

### 2.2 权限断言按平台分支，并补充意图测试

- POSIX：保留 `S_IMODE(...) == 0o600` 严格断言。
- Windows：不再用 `stat` 模式断言隐私属性，改为断言产物存在且内容正确；
  代码注释说明 NTFS 语义。
- 新增 `test_import_requests_restrictive_mode`：用 `monkeypatch` 拦截
  `os.chmod`，证明导入器确实请求 `0o600`（跨平台），把“意图”与“平台是否
  强制执行”分开。

这样 POSIX 上仍验证结果，Windows 上验证请求；不会把 Windows 的 `0o666`
说成安全通过。

## 3. 覆盖矩阵

| 属性 | Windows 本机 | POSIX |
| --- | --- | --- |
| 目录过滤与 ignored 列表 | 已验证 | 已验证 |
| 符号链接排除 | 未验证（无权限创建） | 已验证 |
| 导入产物 `0600` 结果 | 不适用（NTFS 不表达） | 已验证 |
| 导入器请求 `0600` | 已验证（monkeypatch） | 已验证 |

## 4. 复现

```powershell
$py = ".research-venv\Scripts\python.exe"
& $py -m pytest backend/tests/test_benchmark_catalog.py backend/tests/test_benchmark_log_import.py -q
& $py -m pytest -q --junitxml=artifacts/research/windows-full-20261023.xml
```

全量结果（2026-10-23 本机 Windows）：

```
578 passed, 1 skipped in 173.89s (0:02:53)
```

- 失败 0；唯一跳过即上表的符号链接能力用例。
- 与上一轮 562 项（560 通过 / 2 失败）相比，新增 16 项测试且两项失败已按平台语义处理，
  不是通过删除断言取得全绿。
- 原始输出：`artifacts/research/windows-full-20261023.log` 与
  `artifacts/research/windows-full-20261023.xml`。

## 5. 非声明

- 本机通过不等于 POSIX 安全属性通过；符号链接防御仍需在 Linux/CI 上跑一遍。
- `private` 目录名、`chmod(0600)`、应用层路径限制都不是 Windows 账户隔离。
- 分发正式任务包前需管理员配置 NTFS ACL、独立账户与受控浏览器环境。
