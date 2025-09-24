# 文件夹整理说明

## 📁 新的文件结构

项目现已重新组织为更清晰的目录结构：

```
AutoML Data Analysis/
├── 📁 automl-llm/          # 主要应用代码
│   ├── 📁 app/             # Streamlit UI 和 LLM 代理
│   ├── 📁 core/            # 核心数据处理模块
│   └── 📁 artifacts/       # 模型和结果存储
├── 📁 tests/               # 所有测试文件
├── 📁 demos/               # 演示脚本
├── 📁 docs/                # 文档和报告
├── 📁 examples/            # 示例数据
├── 📁 prompts/             # AI 提示模板
├── 📁 artifacts/           # 全局输出
├── 📄 README.md            # 项目说明
├── 📄 requirements.txt     # 依赖列表
├── 📄 usage_example.py     # 使用示例
└── 📄 .env                 # 环境变量配置
```

## 📝 各文件夹说明

### 🧪 tests/ - 测试文件
- `test_complete_system.py` - 完整系统集成测试
- `test_openai_setup.py` - OpenAI API 配置测试
- `test_research_questions.py` - 研究问题建议功能测试
- `test_ui_attributeerror_fix.py` - UI AttributeError 修复测试
- `test_fix.py` - 基础修复测试
- 其他测试文件...

### 🎯 demos/ - 演示脚本
- `demo_attributeerror_fix.py` - AttributeError 修复演示
- `demo_complete_workflow.py` - 完整工作流演示
- `demo_readable_ui.py` - 可读化UI演示

### 📚 docs/ - 文档和报告
- `ATTRIBUTEERROR_FIX_REPORT.md` - AttributeError 修复报告
- `RESEARCH_QUESTIONS_FEATURE_REPORT.md` - 研究问题功能报告
- `UI_IMPROVEMENT_REPORT.md` - UI 改进报告
- `PYLANCE_FIX_REPORT.md` - Pylance 修复报告
- `FIX_REPORT.md` - 基础修复报告
- `OPENAI_SETUP.md` - OpenAI 配置说明

## 🔧 路径更新

所有文件中的相对路径引用已更新：

### 测试文件中的路径
```python
# 之前：
sys.path.append(os.path.join(os.path.dirname(__file__), 'automl-llm'))

# 现在：
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'automl-llm'))
```

### 环境文件路径
```python
# 之前：
project_root = os.path.dirname(__file__)

# 现在：
project_root = os.path.dirname(os.path.dirname(__file__))
```

## 🚀 使用方法

### 运行主应用
```bash
streamlit run automl-llm/app/ui_streamlit.py
```

### 运行测试
```bash
# 从项目根目录运行
python tests/test_complete_system.py
python tests/test_openai_setup.py
```

### 运行演示
```bash
# AttributeError 修复演示
streamlit run demos/demo_attributeerror_fix.py

# 完整工作流演示
python demos/demo_complete_workflow.py
```

## ✅ 验证整理结果

1. **所有测试文件** 已移动到 `tests/` 目录
2. **所有演示文件** 已移动到 `demos/` 目录  
3. **所有文档报告** 已移动到 `docs/` 目录
4. **路径引用** 已全部更新并验证
5. **文件功能** 保持完整，无损坏

## 📋 整理前后对比

### 整理前（根目录混乱）
```
├── test_complete_system.py
├── test_openai_setup.py
├── demo_attributeerror_fix.py
├── ATTRIBUTEERROR_FIX_REPORT.md
├── UI_IMPROVEMENT_REPORT.md
├── ... (20+ 个文件混杂)
```

### 整理后（结构清晰）
```
├── tests/      (7个测试文件)
├── demos/      (3个演示文件)  
├── docs/       (6个文档报告)
├── README.md   (项目说明)
└── 其他核心文件
```

---

## 🎉 整理完成！

项目文件夹现在结构清晰，易于维护和导航。所有功能保持完整，路径引用已正确更新。