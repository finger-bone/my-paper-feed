# Paper Feed

每周自动抓取 arXiv 和 HuggingFace 上的最新论文，使用 DeepSeek API 筛选与大模型推理/训练加速相关的论文，并生成 RSS 订阅源。

## 功能

- 📥 从 arXiv (cs.CL, cs.CV, cs.LG, cs.AI, cs.MM) 抓取最新论文
- 📥 从 HuggingFace Daily Papers 抓取每日论文
- 🤖 使用 DeepSeek API 自动判断论文相关性并生成中文摘要
- 📰 生成 RSS 2.0 和 Atom 格式订阅源
- ⏰ 每周自动运行 (GitHub Actions)

## 使用方法

### 1. 部署到 GitHub

1. 在 GitHub 上创建一个新仓库
2. 将本仓库代码推送到 GitHub：
   ```bash
   git init
   git add .
   git commit -m "init: paper feed"
   git branch -M main
   git remote add origin https://github.com/<你的用户名>/<仓库名>.git
   git push -u origin main
   ```

### 2. 配置 DeepSeek API Key

在 GitHub 仓库的 **Settings → Secrets and variables → Actions** 中添加以下 Secret：

| Secret Name | 必需 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API Key ([申请地址](https://platform.deepseek.com/api_keys)) |

可选配置（通过 Secret 或修改 `.github/workflows/weekly-paper.yml`）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名称（如 `deepseek-v4-flash`） |
| `MIN_RELEVANCE_SCORE` | `5` | 最低相关性分数 (0-10) |
| `LOOKBACK_DAYS` | `7` | 向前查找的天数 |

### 3. 启用 GitHub Pages（可选）

如果想让 RSS 订阅源可通过网页访问：

1. 仓库 **Settings → Pages**
2. Source 选择 **Deploy from branch**
3. Branch 选择 `main`，目录选 `/ (root)`
4. Save
5. 你的 RSS 订阅地址为：`https://<你的用户名>.github.io/<仓库名>/feed.xml`

### 4. 订阅 RSS

将 RSS 链接添加到你的 RSS 阅读器中即可。

## 本地运行

```bash
# 安装依赖
bun install

# 设置环境变量
export DEEPSEEK_API_KEY="your-api-key"

# 运行
bun start
```

## 技术栈

- **Runtime**: Bun
- **语言**: TypeScript
- **依赖**:
  - `openai` — DeepSeek API 调用
  - `feed` — RSS/Atom 生成
  - `fast-xml-parser` — arXiv XML 解析
- **CI/CD**: GitHub Actions (每周定时运行)
