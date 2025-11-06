# OpenAI API Key 设置指南

## 步骤1: 获取 OpenAI API Key

1. 访问 [OpenAI 官网](https://platform.openai.com/)
2. 登录或注册账户
3. 进入 [API Keys 页面](https://platform.openai.com/api-keys)
4. 点击 "Create new secret key"
5. 复制生成的 API Key (格式类似: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

## 步骤2: 设置 API Key

### 方法1: 使用 .env 文件（推荐）

1. 打开项目根目录下的 `.env` 文件
2. 将 `your_openai_api_key_here` 替换为你的实际 API Key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
3. 保存文件

### 方法2: 设置系统环境变量

#### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

#### Windows (命令提示符):
```cmd
set OPENAI_API_KEY=sk-your-actual-api-key-here
```

#### Linux/macOS:
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

## 高级配置（可选）

可以通过以下环境变量进一步配置 LLM 行为：

```env
# 自定义用于不同任务的模型与温度
LLM_TASK_MODEL=gpt-4o-mini
LLM_RESEARCH_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2

# 使用 OpenAI 兼容的服务端点（例如代理或第三方兼容服务）
# 默认为官方 https://api.openai.com/v1
OPENAI_BASE_URL=https://api.openai.com/v1

# 离线模式：不调用外部 LLM，改用启发式回退
LLM_OFFLINE=0
```

注意：开启 `LLM_OFFLINE=1` 后，判定与研究问题建议将使用本地启发式回退，方便在无网络/无 Key 时体验流程。

## 步骤3: 安装依赖

确保安装了 python-dotenv 包：

```bash
pip install python-dotenv
```

或者安装所有依赖：

```bash
pip install -r requirements.txt
```

## 步骤4: 验证设置

运行以下命令验证 API Key 是否正确设置：

```python
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 检查 API Key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✅ OPENAI_API_KEY 已设置")
    print(f"Key 前缀: {api_key[:10]}...")
else:
    print("❌ OPENAI_API_KEY 未设置")
```

## 步骤5: 使用智能判定功能

设置完成后，在 Streamlit 应用中：
1. 上传数据文件
2. 输入你的问题（可选）
3. 点击 "智能判定（OpenAI）" 按钮
4. 系统会调用 OpenAI API 分析数据并推荐任务类型和算法

## 注意事项

⚠️ **安全提醒**:
- API Key 是敏感信息，请勿分享给他人
- 不要将 API Key 提交到公开的 Git 仓库
- `.env` 文件已在 `.gitignore` 中，不会被提交

💰 **费用提醒**:
- OpenAI API 按使用量收费
- 建议设置使用限额避免意外费用
- 查看 [OpenAI 定价](https://openai.com/pricing) 了解详细费用

🔧 **故障排除**:
- 如果遇到 "OPENAI_API_KEY not set" 错误，请检查环境变量设置
- 如果 API 调用失败，请检查网络连接和 API Key 有效性
- 确保账户有足够的 API 额度