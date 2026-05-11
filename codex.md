# Codex CLI 会话迁移与继续对话指南

本指南演示如何将本地 Codex CLI 对话会话导出并在另一台电脑继续对话。

---

## 前提条件

1. 两台电脑都已安装 **Codex CLI** 并能正常登录。
2. 可通过网络、云盘或 U 盘传输文件。

---

## 1️⃣ 在电脑 A 导出会话

假设你的会话名为 `my_chat`：

```bash
# 查看所有会话
codex chat --list-sessions

# 导出指定会话到 JSON 文件
codex chat --export my_chat --file ~/my_chat_backup.json

# 导入会话
codex chat --import /home/user/my_chat_backup.json

# 查看导入后的会话列表
codex chat --list-sessions

# 使用原来的会话继续聊天
codex chat --session my_chat

