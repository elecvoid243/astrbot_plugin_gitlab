# AstrBot GitLab 项目机器人

把 GitLab 项目接入 AstrBot 的「机器人」面板。项目成员在 Issue 或 MR 讨论中 `@机器人` 提问，机器人可以查阅本次提交的源码、项目历史 Issue、MR 和 Git 提交，在原讨论中回复。

## 安装与绑定

1. 在运行 AstrBot 的主机或容器中准备 Git 2.31 或以上版本，并确认它能通过 HTTP(S) 拉取目标项目。插件使用 Python 自带的 SQLite FTS5 和 AstrBot 的 `aiohttp`。本项目针对 `AstrBot_for_spzx` 4.31.4 的平台适配器接口。
2. 在 AstrBot 的插件面板上传 `astrbot_plugin_gitlab.zip`，或将源码目录放在 `data/plugins/astrbot_plugin_gitlab/` 后重启 AstrBot。
3. 在「模型提供商」中添加 OpenAI 兼容的聊天模型，API Base 填 AstrBot 能访问的 vLLM 地址（例如 `http://vllm:8000/v1`），模型名使用 vLLM `/v1/models` 返回的 ID。模型需要能按提示返回 JSON；插件自行执行检索，无需启用 vLLM 原生工具调用。
4. 在「机器人」中新建 **GitLab**。填写实例根地址、项目数字 ID、机器人 API Token、独立的 Webhook Secret token，选择上一步的模型。API Token 可使用绑定项目的 Project Access Token，角色 Reporter、scope `api`，或使用具有相应项目权限的专用机器人账号。`@` 提及时使用该 Token 对应的 GitLab 用户名。每个机器人实例绑定一个项目。
5. 保持统一 Webhook 开启，保存并启用机器人。从机器人面板或启动日志取得回调路径，在 GitLab 项目 **Settings → Webhooks** 中配置完整的 `https://<AstrBot 的可达域名>/api/v1/webhooks/platforms/<回调标识>`。在 GitLab 的 **Secret token** 填入与机器人配置相同的值。订阅 **Work item events**（旧版本显示 Issue events）、**Merge request events** 和 **Comments**。使用该路径即可接入，反向代理应将 POST 请求和 `X-Gitlab-Token` 请求头转交给 AstrBot。[GitLab Webhook 事件](https://docs.gitlab.com/user/project/integrations/webhook_events/)
6. 用有仓库读取权限的项目成员在一个测试 Issue 下发送 `@机器人用户名 /help`，验证回调、身份判断和回帖。再发送 `/related 一段错误信息` 验证模型和历史检索。

## 首版交互约定

在 AstrBot「机器人」面板中，每个绑定项目分别配置 **Issue 自动回复** 和 **MR 自动评审**，两项默认开启。关闭任一项后，仍可在讨论中手动调用相应能力。

自动事件和手动请求都按触发者身份判断。具备绑定项目仓库读取权限的项目成员可以触发机器人，通过群组继承成员资格的用户也适用。外部贡献者创建 Issue、提交 MR 或发表评论时，机器人保持安静；符合条件的成员仍可在同一讨论中通过 `@机器人` 或项目专用指令发起分析。

| 事件 | 默认行为 |
| --- | --- |
| 新建 Issue | 分析一次。能定位时引用源码，信息不足时询问最关键的复现条件。 |
| 新建待评审 MR，或草稿 MR 转为待评审 | 评审当前提交，发布一条摘要。草稿期间可用 `/review` 手动发起评审。 |
| 待评审 MR 推送新提交 | 评审新版本，更新原摘要。只修改标题、标签或描述时跳过。 |
| Issue 或 MR 中出现评论 | 收到 `@机器人` 或项目专用指令时按下表处理。普通评论不触发回复。 |

## 指令与分析依据

| 输入 | 结果 |
| --- | --- |
| `@机器人用户名 为什么这里会返回空值？` 或 `/ask 问题` | 结合当前 Issue/MR、所在讨论及检索结果回答。 |
| `/related` 或 `/related timeout` | 以当前标题或关键词检索本项目，比较已关闭和仍开放的历史 Issue，列出疑似重复、有关联或关系待核实的候选及理由。 |
| `/review` | 按当前 MR 的提交评审，可在草稿期间手动触发。 |
| `/help` | 在原讨论显示项目指令。 |

自动评审和手动 `/review` 共用一条 MR 摘要。首次默认发布为普通评论，开启 `publish_review_as_thread` 后，新摘要会发布为可解决讨论。已有摘要继续在原位置更新。更新后默认发布一条带摘要链接的评论，可关闭 `final_update_message`。开启 `reply_to_trigger_comment` 后，手动评审的更新链接会发送至命令所在讨论。如果无法定位讨论或回复失败，链接会作为普通评论发布。

新 Issue 会自动执行关联检索。模型可继续通过关键词、`#IID`、`!IID`、提交号、文件路径或代码标识符补查资料。重复判断作为评论供成员核实，关闭 Issue 或更改关联关系由项目成员决定。历史 Issue 候选排除当前 Issue 与机密 Issue，线程上下文排除内部备注。回复遵循所在讨论的可见范围。

Issue 分析使用默认分支的具体提交，MR 使用其 head 和 GitLab 返回的 diff 基线。索引覆盖最多 256 KiB 的普通文本文件，可按路径读取行段、搜索定义及调用位置。索引按文本和符号匹配检索，动态调用关系需要结合代码核实。历史搜索有候选数和推理轮次上限，未找到候选表示本次未确认重复。

自动评审和手动 `/review` 从本地 Git 按固定基线和 head 读取文件补丁，按所选模型的上下文容量安排一批评审。预算包含实际请求的提示词、源码依据和预留的模型输出。默认先用模型提供商配置的上下文容量，其次用 AstrBot 已知的模型容量，未知时按 8192 token 估算。紧预算下优先为相关调用方的精简源码依据留出空间。放不下的文件补丁整份跳过，摘要列出这些文件。模型提供商在重试时缩减原始输入，且 AstrBot 报告请求内容已改变时，本次评审报告失败。比较基线或 head 发生变化时，旧结果不会发布。

## 运行与排查

- `data/plugin_data/astrbot_plugin_gitlab/` 按实例地址、项目 ID、机器人实例 ID 隔离，包含裸 Git 仓库、源码索引和待处理事件。容器部署需持久挂载 AstrBot 的 `data`。首次分析需要拉取分支历史并建立索引。
- Webhook 写入本地队列后返回 HTTP 200。每个机器人串行处理事件，重启后会继续待处理工作。失败最多尝试五次；修复配置后可在 GitLab 重新投递失败事件或新发指令。
- HTTP 403 通常表示回调密钥不匹配。若返回 200 但没有评论，检查项目 ID、触发者的直接或继承成员资格、自动开关及 AstrBot 日志。自定义角色的读码权限若无法从成员 API 确认，机器人保守跳过。常规私有项目优先给需要使用机器人的成员标准 Reporter 或更高角色。[GitLab 成员 API](https://docs.gitlab.com/api/project_members/)
- 当前使用 GitLab 的 **Secret token** 回调方式。一个项目只启用一个处理同类事件的机器人实例，以免两个实例分别发表评论。
