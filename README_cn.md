# AutoML-LLM 数据分析平台

🚀 **结合自动化机器学习（AutoML）与大语言模型（LLM）的智能数据分析平台**

English version: README_en.md

## 📋 项目简介

AutoML-LLM 是一个创新的数据分析平台，融合了自动化机器学习和大语言模型技术。该平台通过智能化的数据处理、特征工程、模型选择与评估，大幅提升数据分析效率，并利用 LLM 实现自然语言交互与自动化决策建议。

### 🎯 核心特性

- **🤖 智能任务检测**：基于 LLM 的自动任务类型识别
- **📊 自动化数据处理**：智能数据清洗、特征工程和预处理
- **🔍 模型自动选择**：根据数据特征自动选择最适合的机器学习模型
- **📈 可视化界面**：基于 Streamlit 的友好用户界面
- **🧠 研究问题建议**：AI 驱动的数据分析研究问题生成
- **📋 详细报告生成**：自动化模型评估和结果解释

### 🏗️ 项目结构

```
AutoML Data Analysis/
├── 📁 automl-llm/          # 主要应用代码
│   ├── 📁 app/             # Streamlit UI 和 LLM 代理
│   │   ├── ui_streamlit.py # 主要用户界面
│   │   └── llm_agent.py    # LLM 智能代理
│   ├── 📁 core/            # 核心数据处理模块
│   │   ├── ingest.py       # 数据摄取和预处理
│   │   ├── cleandata.py    # 数据清洗
│   │   ├── models.py       # 机器学习模型定义
│   │   └── train.py        # 模型训练和评估
│   └── 📁 artifacts/       # 模型和结果存储
├── 📁 tests/               # 完整测试套件
├── 📁 demos/               # 演示脚本和示例
├── 📁 docs/                # 详细文档和技术报告
├── 📁 examples/            # 示例数据集
├── 📁 prompts/             # LLM 提示模板
├── 📄 usage_example.py     # 使用示例脚本
├── 📄 requirements.txt     # Python 依赖
└── 📄 README.md            # 项目说明（本文件）
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Windows/Linux/macOS
- 2GB+ 可用内存

### 1️⃣ 创建虚拟环境

```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate  # Windows
# 或 source venv/bin/activate  # Linux/macOS
```

### 2️⃣ 安装依赖

```powershell
pip install -r requirements.txt
```

### 3️⃣ 配置 OpenAI/兼容 API（可选）

如需使用 LLM 功能，请配置 OpenAI API 密钥：

```powershell
# 创建 .env 文件并添加你的 API 密钥
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

更多配置详情请参考：[OpenAI 配置说明](docs/OPENAI_SETUP.md)

### 4️⃣ 启动应用

```powershell
# 启动 Streamlit Web 界面
streamlit run automl-llm/app/ui_streamlit.py

# 或运行使用示例
python usage_example.py
```

## ⚙️ 环境变量与离线模式

无需联网也能跑！未配置密钥或设置离线模式时，系统会使用“启发式回退”，依然可完成任务判定与研究问题建议（只是不调用线上模型）。

- OPENAI_API_KEY：OpenAI API 密钥（可选，启用联网 LLM 时需要）
- OPENAI_BASE_URL：OpenAI 兼容服务地址（可选，例如企业代理或第三方兼容端点）
- LLM_OFFLINE：设置为 1 可强制离线模式，不调用外部 LLM（默认 0）
- LLM_TASK_MODEL：任务判定所用模型名（默认 gpt-4o-mini）
- LLM_RESEARCH_MODEL：研究问题建议所用模型名（默认 gpt-4o-mini）
- LLM_TEMPERATURE：采样温度（默认 0.2）

在 Windows PowerShell 中临时设置环境变量：

```powershell
$env:LLM_OFFLINE = "1"        # 当会话有效，适合本地演示
$env:OPENAI_API_KEY = "sk-..." # 如需启用联网能力
```

长期保存（新开终端也有效）：

```powershell
setx LLM_OFFLINE "1"
setx OPENAI_API_KEY "sk-..."
```

## 💻 使用指南

### Web 界面使用

1. **数据上传**：在 Web 界面上传 CSV 格式的数据文件
2. **自动分析**：系统自动进行数据预处理和任务类型检测
3. **模型训练**：选择合适的机器学习模型进行自动训练
4. **结果查看**：查看模型性能评估和可视化结果
5. **研究建议**：获取 AI 生成的数据分析研究问题建议

### 编程接口使用

参考 `usage_example.py` 了解如何通过代码调用核心功能：

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'automl-llm'))

from core.ingest import read_table
from core.cleandata import clean_data
from core.train import run_all

# 读取数据
df = read_table("examples/your_data.csv")

# 自动数据清洗和预处理
cleaned_data = clean_data(df, task_type="auto", target_col="your_target")

# 自动模型训练和评估
results = run_all(X_train, y_train, X_test, y_test, 
                  task_type="classification", 
                  picked_models=["random_forest", "xgboost"])
```

## 🧪 运行测试

项目包含完整的测试套件，确保功能正常：

```powershell
# 运行完整系统测试
python tests/test_complete_system.py

# 运行特定功能测试
python tests/test_openai_setup.py
python tests/test_research_questions.py
python tests/test_ui_attributeerror_fix.py
```

提示：与 LLM 调用相关的测试在未设置 OPENAI_API_KEY 时会自动走“离线回退”，依旧可以观察到结构化输出。

## 📚 功能特性详解

### 🔍 智能任务检测
- 基于数据特征自动识别分类/回归任务
- 支持多种数据类型：数值型、分类型、时间序列

### 🛠️ 自动化数据处理
- **数据清洗**：缺失值处理、异常值检测
- **特征工程**：自动特征选择、编码转换
- **数据分割**：智能训练/测试集划分

### 🤖 机器学习模型
支持多种主流算法：
- **分类**：Random Forest, XGBoost, SVM, Logistic Regression
- **回归**：Linear Regression, Ridge, Lasso, Gradient Boosting
- **集成学习**：自动模型选择和超参数优化

### 🎨 可视化分析
- 数据分布可视化
- 模型性能对比图表
- 特征重要性分析
- 预测结果展示

### 🧷 Streamlit 界面能力（CSV 合并/分析）
- 支持一次上传多个 CSV 文件
- 自动计算“公共列”并支持两种合并模式：
    - 纵向堆叠（仅保留公共列，可选添加来源列 _source_file）
    - 横向匹配（以公共列作为键，类似多表 join，非键列可自动加前缀防冲突）
- 合并结果会保存到 `examples/` 目录，默认文件名如 `merged_common.csv`、`merged_horizontal.csv`
- 合并完成后可直接作为训练数据源继续分析

## 📖 技术文档

- [文件夹组织说明](docs/FOLDER_ORGANIZATION_README.md)
- [AttributeError 修复报告](docs/ATTRIBUTEERROR_FIX_REPORT.md)
- [UI 改进报告](docs/UI_IMPROVEMENT_REPORT.md)
- [研究问题功能报告](docs/RESEARCH_QUESTIONS_FEATURE_REPORT.md)
- [Pylance 修复报告](docs/PYLANCE_FIX_REPORT.md)

## 🎯 适用场景

- **🎓 学术研究**：快速进行数据分析和假设验证
- **💼 企业应用**：业务数据智能分析和预测
- **📚 教学培训**：机器学习和数据科学教学工具
- **🚀 原型开发**：快速构建数据驱动的应用原型

## 🤝 贡献指南

欢迎提交 Issues 和 Pull Requests！

1. Fork 本项目
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 开启 Pull Request

## 📝 更新日志

### 最新版本特性
- ✅ 修复了 AttributeError 相关问题
- ✅ 改进了 Streamlit UI 用户体验
- ✅ 增加了研究问题自动生成功能
- ✅ 优化了文件夹结构和代码组织
- ✅ 完善了测试覆盖和错误处理

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: [xdn1356@126.com]
- 🐛 Issues: [GitHub Issues](https://github.com/Frank-Xu03/AutoML-Data-Analysis/issues)
- 📖 Wiki: [项目文档](https://github.com/Frank-Xu03/AutoML-Data-Analysis/wiki)

---

**AutoML-LLM** - 让数据分析更智能、更高效 🚀

## 📦 示例数据与产物

- 示例数据位于 `examples/`：`movies.csv`、`ratings.csv`、`tags.csv`、`links.csv` 等
- 训练产物默认写入 `artifacts/`，例如 `leaderboard.csv`

## ❓常见问题（FAQ）

- 没有 OpenAI Key 能用吗？
    - 可以。设置 `LLM_OFFLINE=1` 或不设置密钥，系统会使用内置启发式回退给出可用计划与建议。
- UI 中 examples 文件会被清空吗？
    - 每个会话首次进入 UI 时会清理 `examples/`（以避免旧文件干扰），随后上传的/合并的文件会写回该目录。
- Windows 下命令怎么写？
    - 本 README 所有命令均适配 PowerShell。若使用 CMD，请适当替换环境变量写法。
