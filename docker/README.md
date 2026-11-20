# 本地 Docker 演示

该 Compose 配置用于本地研究演示，不是面向真实未成年人或公网生产环境的部署方案。

## 启动

1. 从 `.env.example` 创建被 Git 忽略的 `.env.local`。
2. 只在本机填写 `LITELLM_API_KEY`、已经通过资格测试的模型 ID，以及该模型快照对应的
   `ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION`、`ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION` 和
   `ECO_MODEL_PRICING_VERSION`。启用 provider-backed 生成式对话时，三项价格元数据是
   启动前置条件，用于成本账本和预算审计；不能留空。
3. 启动：

```bash
ECO_DOCKER_UID="$(id -u)" \
ECO_DOCKER_GID="$(id -g)" \
docker compose --env-file .env.local up --build
```

默认地址：

- Web：`http://127.0.0.1:4173`
- Backend：`http://127.0.0.1:8000`
- Web 健康检查：`http://127.0.0.1:4173/healthz`
- Backend 健康检查：`http://127.0.0.1:8000/healthz`

容器默认只绑定回环地址。不要在没有认证、TLS、访问控制、审计和人工接管的情况下
改为公网监听。

## 不配置 API 密钥时

健康检查和本地确定性功能不应访问外部模型。需要生成式对话或模型发现的请求会因
缺少密钥而快速失败。Compose 不包含任何默认密钥，镜像也不会复制 `.env.local`。

如果配置了 API 密钥但未配置完整的模型价格元数据，provider-backed Web 对话会在启动时
快速失败；请先通过 `eco models qualify` 生成并冻结模型快照，再把快照中的价格版本和
单价填入 `.env.local`。价格只用于成本核算，不代表中转服务当前实际计费承诺。

## 数据

宿主机 `./artifacts` 绑定到 backend 的 `/app/artifacts`。Linux 上通过
`ECO_DOCKER_UID` 和 `ECO_DOCKER_GID` 让容器写入文件保持为当前用户所有。

运行缓存、本地数据库、模型快照和 Trace 默认受 `.gitignore` 保护。准备发布包时，
仍需先做脱敏、Schema 验证、校验和及人工审核。

## 健康检查含义

- backend `/healthz` 只确认 API 进程和本地依赖可响应；
- web `/healthz` 只确认 Nginx 可响应；
- 健康状态不代表模型中转可用，也不代表任何临床安全结论；
- 检查过程不得调用 `/v1/models` 或对话 API，避免无意消耗额度。

查看状态：

```bash
docker compose ps
docker compose logs --no-log-prefix backend web
```

停止并保留 `artifacts`：

```bash
docker compose down
```

## 发布前容器验收

在安装 Docker Compose 的干净机器上执行：

1. `docker compose config` 成功；
2. backend 和 web 镜像可从空缓存构建；
3. 不设置 API 密钥时两个健康检查仍能通过；
4. 设置测试密钥后，Web 可创建会话并完成 SSE 流式回复；
5. 刷新页面后会话、普通记忆和 Trace 行为符合设计；
6. `artifacts` 中不存在密钥、Authorization 头或自由形式推理；
7. `docker history` 和导出的镜像层中不存在 `.env.local`；
8. 从首次执行到两个服务 healthy 的时间不超过两分钟；
9. 中断并重启后，SQLite 与可恢复运行状态保持一致；
10. 429、5xx 和模型超时不会阻断本地高风险安全降级。

当前开发机没有 Docker CLI，因此以上真实镜像构建、Nginx 配置加载和冷启动时限仍是
待在 Docker 主机完成的验收边界。
