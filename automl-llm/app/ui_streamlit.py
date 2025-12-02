import streamlit as st
import os
import shutil
import json
import sys
sys.path.append(os.path.dirname(__file__))
from llm_agent import detect_task
st.set_page_config(page_title="LLM-Augmented AutoML", layout="wide")
import pandas as pd

def TT(zh: str, en: str):
	return en if st.session_state.get("lang", "zh") == "en" else zh

# 语言选择（默认中文）
if "lang" not in st.session_state:
	st.session_state["lang"] = "zh"
_lang_choice = st.sidebar.selectbox(
	"Language / 语言",
	["中文", "English"],
	index=0 if st.session_state.get("lang", "zh") == "zh" else 1,
)
st.session_state["lang"] = "zh" if _lang_choice == "中文" else "en"

st.title(TT("LLM-Augmented AutoML (本地训练)", "LLM-Augmented AutoML (Local Training)"))
st.success(TT("环境初始化成功。接下来将实现数据上传、判定与报告。", "Environment initialized. Next: upload, detection, and reporting."))

# 会话首次运行时清空 examples 目录（避免每次 rerun 都清空新上传文件）
try:
	if not st.session_state.get("__examples_cleared__", False):
		project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
		examples_dir = os.path.join(project_root, 'examples')
		if os.path.exists(examples_dir):
			for name in os.listdir(examples_dir):
				p = os.path.join(examples_dir, name)
				try:
					if os.path.isfile(p) or os.path.islink(p):
						os.remove(p)
					elif os.path.isdir(p):
						shutil.rmtree(p)
				except Exception:
					# 单个文件失败不影响整体
					pass
		st.session_state["__examples_cleared__"] = True
		st.caption(TT("已按需求清空 examples 文件夹（本会话仅一次）。", "Cleared examples folder as requested (once per session)."))
except Exception as __clear_err:
	st.warning(TT(f"清空 examples 文件夹失败：{__clear_err}", f"Failed to clear examples folder: {__clear_err}"))

"""多文件上传区"""
uploaded_files = st.file_uploader(TT("上传一个或多个 CSV 文件", "Upload one or more CSV files"), type=["csv"], accept_multiple_files=True)

active_df = None
df_source_name = None
loaded_dfs = {}

# 统一创建用于上方选择与预览的容器，避免分支未定义
select_container = st.container()
preview_container = st.container()

if uploaded_files:
	examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
	os.makedirs(examples_dir, exist_ok=True)
	# 仅加载与保存文件，不在循环中渲染公共列相关组件，避免重复 key
	for uf in uploaded_files:
		try:
			df_tmp = pd.read_csv(uf)
			loaded_dfs[uf.name] = df_tmp
			save_path = os.path.join(examples_dir, uf.name)
			with open(save_path, 'wb') as f:
				f.write(uf.getbuffer())
		except Exception as e:
			st.error(TT(f"读取文件 {uf.name} 失败: {e}", f"Failed to read file {uf.name}: {e}"))

	# 计算多文件公共列（一次性）
	if len(loaded_dfs) >= 2:
		list_of_colsets = [set(df.columns) for df in loaded_dfs.values() if hasattr(df, 'columns')]
		if list_of_colsets:
			common_columns = set.intersection(*list_of_colsets) if len(list_of_colsets) > 1 else list_of_colsets[0]
		else:
			common_columns = set()
	else:
		common_columns = set()

	# 顶部容器（单实例）
	select_container = st.container()
	preview_container = st.container()

	with st.expander(TT("📌 多文件公共列 (所有文件都包含)", "📌 Common columns across all files"), expanded=True):
		if common_columns:
			st.write(TT(f"共 {len(common_columns)} 个公共列：", f"Total {len(common_columns)} common column(s):"))
			st.code("\n".join(sorted(common_columns)))
		else:
			st.warning(TT("未找到所有文件都共同拥有的列。", "No columns are common to all files."))

		# 可选展示每个文件缺失公共列情况（单实例控件，避免重复 key）
		show_diff = st.checkbox(TT("显示各文件缺失公共列情况", "Show missing common columns per file"), value=False, key="show_diff_missing_cols")
		if show_diff and common_columns:
			for fname, df_tmp in loaded_dfs.items():
				missing_in_file = common_columns - set(df_tmp.columns)
				if missing_in_file:
					st.error(TT(f"{fname} 缺失 {len(missing_in_file)} 个公共列：{', '.join(sorted(missing_in_file))}", f"{fname} missing {len(missing_in_file)} common column(s): {', '.join(sorted(missing_in_file))}"))
				else:
					st.success(TT(f"{fname} 包含全部公共列 ✔", f"{fname} includes all common columns ✔"))

		# ---------------- 合并功能（横向 / 纵向） ----------------
		st.markdown("---")
		st.markdown(TT("### 🔗 合并工具", "### 🔗 Merge Tool"))
		merge_mode = st.radio(
			TT("选择合并方式", "Choose merge mode"),
			[TT("纵向堆叠（仅公共列）", "Vertical stack (common columns only)"), TT("横向匹配（公共列作为键，合并其余列）", "Horizontal join (use common columns as keys)")],
			index=0,
			help=TT("横向匹配=类似多表 join；纵向堆叠=append 行。", "Horizontal join ~ multi-table join; vertical stack ~ append rows.")
		)

		if merge_mode.startswith(TT("纵向", "Vertical")):
			st.caption(TT("仅保留公共列并按行堆叠（之前的行为）。", "Keep only common columns and stack rows (previous behavior)."))
			add_source_col = st.checkbox(TT("添加来源文件列 (_source_file)", "Add source file column (_source_file)"), value=True, key="add_source_file")
			merge_btn = st.button(TT("⚙️ 执行纵向合并", "⚙️ Run vertical merge"), key="merge_vertical")
			if merge_btn:
				if not common_columns:
					st.error(TT("无法合并：没有公共列。", "Cannot merge: no common columns."))
				else:
					try:
						merged_parts = []
						for fname, df_part in loaded_dfs.items():
							subset = df_part[list(common_columns)].copy()
							if add_source_col:
								subset["_source_file"] = fname
							merged_parts.append(subset)
						merged_df = pd.concat(merged_parts, ignore_index=True)
						base_name = "merged_common.csv"
						final_name = base_name
						idx = 1
						while final_name in loaded_dfs:
							idx += 1
							final_name = f"merged_common_{idx}.csv"
						try:
							examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
							os.makedirs(examples_dir, exist_ok=True)
							merged_path = os.path.join(examples_dir, final_name)
							merged_df.to_csv(merged_path, index=False, encoding="utf-8")
						except Exception as fs_err:
							st.warning(TT(f"合并文件保存失败，但内存依旧可用：{fs_err}", f"Failed to save merged file, but in-memory data is available: {fs_err}"))
						loaded_dfs[final_name] = merged_df
						st.session_state["preferred_file_name"] = final_name
						st.session_state["merged_common_df"] = merged_df
						st.success(TT(f"纵向合并成功：{final_name}，形状 {merged_df.shape}", f"Vertical merge success: {final_name}, shape {merged_df.shape}"))
						csv_bytes = merged_df.to_csv(index=False).encode('utf-8')
						st.download_button(TT("⬇️ 下载结果", "⬇️ Download result"), data=csv_bytes, file_name=final_name, mime="text/csv")
						st.info(TT("在上方文件选择框中可选择该合并文件继续分析。", "Select this merged file above to continue analysis."))
						st.session_state["train_df"] = merged_df
						st.session_state["train_source_name"] = final_name
						st.success(TT("训练将默认使用该合并结果。", "Training will default to this merged result."))
					except Exception as merge_err:
						st.error(TT(f"合并失败：{merge_err}", f"Merge failed: {merge_err}"))

		else:  # 横向匹配
			st.caption(TT("使用公共列作为键做多表 join，保留每个文件的其余列。", "Use common columns as keys to join tables; keep other columns."))
			if not common_columns:
				st.error(TT("无法进行横向匹配：没有公共列。", "Cannot do horizontal join: no common columns."))
			else:
				key_cols = sorted(common_columns)
				st.info(TT(f"键列：{', '.join(key_cols)}", f"Key columns: {', '.join(key_cols)}"))
				join_type = st.selectbox(TT("Join 类型", "Join type"), ["outer", "inner", "left"], index=0, help=TT("outer=保留所有键; inner=仅公共键; left=以第一个文件为主表", "outer=keep all keys; inner=only common keys; left=use first file as left table"))
				prefix_cols = st.checkbox(TT("为非键列加文件名前缀以防冲突", "Prefix non-key columns with filename to avoid conflicts"), value=True, key="prefix_non_key")
				drop_dup = st.checkbox(TT("如果某文件键列有重复行，仅保留第一条", "If duplicate keys exist in a file, keep the first"), value=True, key="drop_dup_keys")
				btn_hmerge = st.button(TT("⚙️ 执行横向匹配合并", "⚙️ Run horizontal join"), key="merge_horizontal")
				if btn_hmerge:
					try:
						merged_df = None
						for idx_file, (fname, df_part) in enumerate(loaded_dfs.items()):
							work_df = df_part.copy()
							missing_keys = [k for k in key_cols if k not in work_df.columns]
							if missing_keys:
								st.error(TT(f"文件 {fname} 缺失键列 {missing_keys}，跳过。", f"File {fname} missing key columns {missing_keys}, skipping."))
								continue
							if drop_dup and work_df.duplicated(subset=key_cols).any():
								dup_count = work_df.duplicated(subset=key_cols).sum()
								st.warning(TT(f"{fname} 键列存在 {dup_count} 个重复，将保留第一条。", f"{fname} has {dup_count} duplicate key(s); keeping the first."))
								work_df = work_df.drop_duplicates(subset=key_cols, keep='first')
							non_key_cols = [c for c in work_df.columns if c not in key_cols]
							if prefix_cols:
								base_prefix = os.path.splitext(os.path.basename(fname))[0]
								rename_map = {c: f"{base_prefix}__{c}" for c in non_key_cols}
								work_df = work_df.rename(columns=rename_map)
							cols_to_use = key_cols + [c for c in work_df.columns if c not in key_cols]
							if merged_df is None:
								merged_df = work_df[cols_to_use]
							else:
								merged_df = pd.merge(merged_df, work_df[cols_to_use], on=key_cols, how=join_type)
						if merged_df is None:
							st.error(TT("未能生成合并结果（可能所有文件都被跳过）", "No merged result produced (perhaps all files were skipped)"))
						else:
							base_name = "merged_horizontal.csv"
							final_name = base_name
							ix = 1
							while final_name in loaded_dfs:
								ix += 1
								final_name = f"merged_horizontal_{ix}.csv"
							try:
								examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
								os.makedirs(examples_dir, exist_ok=True)
								merged_path = os.path.join(examples_dir, final_name)
								merged_df.to_csv(merged_path, index=False, encoding='utf-8')
							except Exception as fs_err:
								st.warning(TT(f"合并文件保存失败，但内存仍可使用：{fs_err}", f"Failed to save merged file, but in-memory data is available: {fs_err}"))
							loaded_dfs[final_name] = merged_df
							st.session_state["preferred_file_name"] = final_name
							st.success(TT(f"横向匹配合并成功：{final_name}，形状 {merged_df.shape}", f"Horizontal join success: {final_name}, shape {merged_df.shape}"))
							csv_bytes = merged_df.to_csv(index=False).encode('utf-8')
							st.download_button(TT("⬇️ 下载结果", "⬇️ Download result"), data=csv_bytes, file_name=final_name, mime="text/csv")
							st.info(TT("在上方文件选择框中可选择该横向合并文件继续分析。", "Select this horizontally merged file above to continue analysis."))
							st.session_state["train_df"] = merged_df
							st.session_state["train_source_name"] = final_name
							st.success(TT("训练将默认使用该合并结果。", "Training will default to this merged result."))
					except Exception as e:
						st.error(TT(f"横向合并失败：{e}", f"Horizontal merge failed: {e}"))
else:
	st.info(TT("上传 2 个及以上文件后，将在此显示它们的公共列。", "Upload 2 or more files to see common columns here."))

# 无论是否上传成功，只要有已加载的数据集就提供选择器
if loaded_dfs:
	with select_container:
		file_names = list(loaded_dfs.keys())
		preferred = st.session_state.get("preferred_file_name")
		if preferred in file_names:
			default_idx = file_names.index(preferred)
		else:
			default_idx = 0 if file_names else 0
		pick_name = st.selectbox(TT("选择一个文件进行预览与分析", "Choose a file for preview and analysis"), file_names, index=default_idx, key="file_picker_top")
		active_df = loaded_dfs.get(pick_name)
		df_source_name = pick_name

if active_df is not None:
	df = active_df  # 保持后续代码变量名不变

	# 将“当前活动数据集 + 数据预览 + 数据概览”上移到合并工具上方的容器中
	with preview_container:
		st.info(TT(f"当前活动数据集: {df_source_name}; 形状: {df.shape}", f"Active dataset: {df_source_name}; shape: {df.shape}"))
		st.write(TT("数据预览：", "Data preview:"))
		st.dataframe(df.head())

		#（已移除多文件主列匹配功能）
		with st.expander(TT("🔎 数据概览", "🔎 Data Overview"), expanded=False):
			st.write(TT("数据描述：", "Describe:"))
			st.write(df.describe(include='all').transpose())
			st.write(TT("缺失值统计：", "Missing values:"))
			st.write(df.isnull().sum())

	# ----------- 判定按钮与结果展示区块 -------------
	# 优先使用最近一次合并结果作为 LLM 分析数据源
	analysis_df = st.session_state.get("train_df", df)
	analysis_df_name = st.session_state.get("train_source_name", df_source_name)
	st.caption(TT(f"AI 分析数据源：{analysis_df_name}", f"AI analysis source: {analysis_df_name}"))

	# 构建简易 profile；后续可替换为 ingest.profile
	prof = {
		"columns": [
			{"name": c, "dtype": str(analysis_df[c].dtype), "missing": int(analysis_df[c].isnull().sum()), "unique": int(analysis_df[c].nunique())}
			for c in analysis_df.columns
		]
	}															

	user_question = st.text_area(TT("你的问题（可选）", "Your question (optional)"), placeholder=TT("例如：我们能否预测乘客是否生还？或 预测价格/分群等。", "e.g., Can we predict survival? price? clustering?"))

	col1, col2, col3 = st.columns([1,1,1])
	with col1:
		if st.button(TT("🔍 发现研究问题", "🔍 Discover research questions")):
			with st.spinner(TT("AI 正在分析数据，寻找有价值的研究问题与清洗建议...", "AI is analyzing data to suggest research questions and cleaning tips...")):
				from llm_agent import suggest_research_questions, suggest_cleaning_suggestions
				research_suggestions = suggest_research_questions(prof)
				# 同步生成 清洗建议
				clean_suggest = suggest_cleaning_suggestions(prof, user_question or "")
			st.session_state["research_suggestions"] = research_suggestions
			st.session_state["cleaning_suggest"] = clean_suggest
			st.success(TT("问题发现完成！", "Discovery completed!"))
	
	with col2:
		if st.button(TT("🤖 智能判定任务", "🤖 Smart task detection")):
			with st.spinner(TT("调用 OpenAI 判定任务类型，并生成清洗建议...", "Calling OpenAI to detect task type and generate cleaning suggestions...")):
				from llm_agent import suggest_cleaning_suggestions
				plan = detect_task(prof, user_question or "")
				# 同步生成 清洗建议
				clean_suggest = suggest_cleaning_suggestions(prof, user_question or "")
			st.session_state["plan"] = plan
			st.session_state["cleaning_suggest"] = clean_suggest
			st.success(TT("任务判定完成！", "Task detection completed!"))
	
	with col3:
		if "plan" in st.session_state:
			st.download_button(TT("📄 下载判定结果", "📄 Download detection result"), data=json.dumps(st.session_state["plan"], ensure_ascii=False, indent=2),
							   file_name="task_plan.json", mime="application/json")

	# 新增：目标列与特征保留/删除建议
	col_a, col_b = st.columns([1,2])
	with col_a:
		if st.button(TT("🎯 目标与特征建议", "🎯 Target & feature suggestions")):
			with st.spinner(TT("AI 正在分析目标列与应保留/删除的列...", "AI is analyzing target column and keep/drop features...")):
				from llm_agent import suggest_target_and_features
				feat_suggest = suggest_target_and_features(prof, user_question or "")
			st.session_state["feature_suggest"] = feat_suggest
			st.success(TT("列建议已生成！", "Feature suggestions generated!"))
	with col_b:
		if st.session_state.get("feature_suggest"):
			st.caption(TT("你可以将建议直接应用到后续训练的数据列中。", "You can directly apply suggestions to the training columns."))

	# 显示研究问题建议
	if "research_suggestions" in st.session_state:
		st.subheader(TT("🔍 AI 数据洞察：可研究的问题", "🔍 AI Data Insights: Researchable Questions"))
		
		suggestions = st.session_state["research_suggestions"]
		
		def display_research_suggestions(suggestions):
			"""可读化显示研究问题建议"""
			
			# 检查 suggestions 是否为字典
			if not isinstance(suggestions, dict):
				st.error(TT("❌ 研究建议数据格式错误", "❌ Invalid format for research suggestions"))
				st.json(suggestions)
				return
			
			# 研究问题
			questions = suggestions.get("research_questions", [])
			if questions:
				st.markdown(TT("### 💡 **推荐研究问题**", "### 💡 Recommended Research Questions"))
				
				for i, q in enumerate(questions):
					with st.expander(TT(f"📋 问题 {i+1}: {q.get('question', '未知问题')}", f"📋 Question {i+1}: {q.get('question', 'Unknown')}"), expanded=i==0):
						col1, col2 = st.columns(2)
						with col1:
							st.markdown(TT(f"**类型**: {q.get('type', '未知')}", f"**Type**: {q.get('type', 'Unknown')}"))
							st.markdown(TT(f"**难度**: {q.get('difficulty', '未知')}", f"**Difficulty**: {q.get('difficulty', 'Unknown')}"))
						with col2:
							if q.get('target_column'):
								st.markdown(TT(f"**目标列**: `{q.get('target_column')}`", f"**Target**: `{q.get('target_column')}`"))
							methods = q.get('required_methods', [])
							if methods:
								st.markdown(TT(f"**推荐方法**: {', '.join(methods)}", f"**Recommended methods**: {', '.join(methods)}"))
						
						st.markdown(TT("**商业价值**:", "**Business value**:"))
						st.info(q.get('business_value', TT('未提供', 'Not provided')))
			
			# 应用场景
			scenarios = suggestions.get("application_scenarios", [])
			if scenarios:
				st.markdown(TT("### 🎯 **应用场景**", "### 🎯 Application Scenarios"))
				for i, scenario in enumerate(scenarios):
					st.markdown(f"**{i+1}.** {scenario}")
			
			# 关键洞察潜力
			insights = suggestions.get("key_insights_potential", [])
			if insights:
				st.markdown(TT("### 🔮 **可能发现的洞察**", "### 🔮 Potential Insights"))
				for insight in insights:
					st.markdown(f"• {insight}")
			
			# 数据集优势
			strengths = suggestions.get("dataset_strengths", [])
			if strengths:
				st.markdown(TT("### ✨ **数据集优势**", "### ✨ Dataset Strengths"))
				for strength in strengths:
					st.markdown(f"✅ {strength}")
			
			# 限制和注意事项
			limitations = suggestions.get("limitations", [])
			if limitations:
				st.markdown(TT("### ⚠️ **注意事项**", "### ⚠️ Caveats"))
				for limitation in limitations:
					st.warning(f"⚠️ {limitation}")

			# 建议
			recommendations = suggestions.get("recommendations", {})
			if recommendations:
				st.markdown(TT("### 🎯 **行动建议**", "### 🎯 Recommendations"))
				# 检查 recommendations 是否为字典
				if isinstance(recommendations, dict):
					priority = recommendations.get("priority_questions", [])
					if priority:
						st.markdown(TT("**🔥 优先研究问题**:", "**🔥 Priority questions**:"))
						for p in priority:
							st.markdown(f"• {p}")
					
					next_steps = recommendations.get("next_steps", [])
					if next_steps:
						st.markdown(TT("**📋 建议步骤**:", "**📋 Suggested steps**:"))
						for step in next_steps:
							st.markdown(f"• {step}")
					
					additional_data = recommendations.get("additional_data", [])
					if additional_data:
						st.markdown(TT("**📊 可能需要的额外数据**:", "**📊 Additional data needed**:"))
						for data in additional_data:
							st.markdown(f"• {data}")
				else:
					# 如果 recommendations 是字符串，直接显示
					st.markdown(TT(f"**💡 建议**: {recommendations}", f"**💡 Suggestion**: {recommendations}"))
		
		display_research_suggestions(suggestions)

		# 研究问题分析结论按钮与展示
		st.markdown(TT("### 🧠 研究问题分析结论", "### 🧠 Research Question Analysis"))
		col_rqa1, col_rqa2 = st.columns([1,2])
		with col_rqa1:
			if st.button(TT("分析研究问题结论", "Analyze research conclusions"), key="btn_analyze_research"):
				from llm_agent import analyze_research_questions
				with st.spinner(TT("AI 正在综合研究问题并生成结论...", "AI synthesizing research questions...")):
					res_analysis = analyze_research_questions(suggestions, st.session_state.get("profile_for_report"))
				st.session_state["__research_analysis__"] = res_analysis
				st.success(TT("研究问题分析完成", "Research analysis complete"))
		with col_rqa2:
			if st.session_state.get("__research_analysis__"):
				ra = st.session_state["__research_analysis__"]
				st.markdown(ra.get("markdown","(no analysis)"))
				st.download_button(
					TT("⬇️ 下载研究分析 Markdown", "⬇️ Download research analysis"),
					data=ra.get("markdown",""),
					file_name="research_analysis.md",
					mime="text/markdown"
				)
		
		# 原始 JSON (可选)
		with st.expander(TT("🔍 查看详细分析结果 (JSON)", "🔍 View detailed analysis (JSON)"), expanded=False):
			st.json(suggestions)
		
		st.success(TT("💡 基于以上分析，你可以选择感兴趣的问题进行深入研究！", "💡 Based on the above, pick questions to dive deeper!"))

	if "plan" in st.session_state:
		st.subheader(TT("🤖 AI 智能判定结果", "🤖 AI Task Detection Result"))
		
		plan = st.session_state["plan"]
		
		# 可读化显示判定结果
		def display_readable_plan(plan):
			# 任务类型
			task_type = plan.get("task_type", TT("未知", "unknown"))
			task_type_title_cn = {"classification": "分类任务", "regression": "回归任务", "clustering": "聚类任务"}.get(task_type, task_type)
			st.markdown(TT(f"### 📊 **任务类型**: {task_type_title_cn}", f"### 📊 **Task type**: {task_type}"))
			if task_type == "classification":
				st.info(TT("🎯 这是一个分类任务，目标是预测离散的类别标签", "🎯 Classification: predict discrete class labels"))
			elif task_type == "regression":
				st.info(TT("📈 这是一个回归任务，目标是预测连续的数值", "📈 Regression: predict continuous values"))
			elif task_type == "clustering":
				st.info(TT("🔍 这是一个聚类任务，目标是发现数据中的隐藏模式", "🔍 Clustering: discover hidden patterns"))
			
			# 目标候选列
			targets = plan.get("target_candidates", [])
			if targets:
				st.markdown(TT("### 🎯 **推荐目标列**", "### 🎯 Suggested target columns"))
				for i, target in enumerate(targets):
					st.markdown(f"**{i+1}.** `{target}`")
			else:
				st.warning(TT("⚠️ 未找到明确的目标列，请手动选择", "⚠️ No clear target column found; please select manually"))
			
			# 推荐算法
			algorithms = plan.get("algorithms", [])
			if algorithms:
				st.markdown(TT("### 🤖 **推荐算法**", "### 🤖 Recommended algorithms"))
				algo_names = {
					"xgboost": TT("XGBoost (极端梯度提升)", "XGBoost (Extreme Gradient Boosting)"),
					"ridge": TT("Ridge 回归 (岭回归)", "Ridge Regression"),
					"knn": TT("K-近邻算法", "K-Nearest Neighbors"),
					"random_forest": TT("随机森林", "Random Forest"),
					"linear_regression": TT("线性回归", "Linear Regression"),
					"logistic_regression": TT("逻辑回归", "Logistic Regression"),
					"svm": TT("支持向量机", "Support Vector Machine"),
					"mlp": TT("多层感知机", "MLP (Multilayer Perceptron)")
				}
				
				cols = st.columns(min(len(algorithms), 3))
				for i, algo in enumerate(algorithms):
					with cols[i % 3]:
						algo_display = algo_names.get(algo, algo.replace('_', ' ').title())
						st.markdown(f"**{i+1}.** {algo_display}")
			
			# 评估指标
			metrics = plan.get("metrics", [])
			if metrics:
				st.markdown(TT("### 📏 **评估指标**", "### 📏 Metrics"))
				metric_names = {
					"rmse": TT("RMSE (均方根误差)", "RMSE (root mean squared error)"),
					"mae": TT("MAE (平均绝对误差)", "MAE (mean absolute error)"), 
					"r2": TT("R² (决定系数)", "R² (coefficient of determination)"),
					"accuracy": TT("准确率", "Accuracy"),
					"f1": TT("F1 分数", "F1 score"),
					"precision": TT("精确率", "Precision"),
					"recall": TT("召回率", "Recall"),
					"auc": TT("AUC (曲线下面积)", "AUC (area under curve)")
				}
				
				metric_cols = st.columns(min(len(metrics), 3))
				for i, metric in enumerate(metrics):
					with metric_cols[i % 3]:
						metric_display = metric_names.get(metric, metric.upper())
						st.markdown(f"**•** {metric_display}")
			
			# 交叉验证设置
			cv_info = plan.get("cv", {})
			if cv_info:
				st.markdown(TT("### ✅ **交叉验证设置**", "### ✅ Cross-validation settings"))
				folds = cv_info.get("folds", 5)
				stratified = cv_info.get("stratified", False)
				
				col1, col2 = st.columns(2)
				with col1:
					st.metric(TT("交叉验证折数", "CV folds"), f"{folds}")
				with col2:
					stratify_text = TT("是", "Yes") if stratified else TT("否", "No")
					st.metric(TT("分层采样", "Stratified"), stratify_text)
			
			# 类别不平衡信息
			imbalance = plan.get("imbalance", {})
			if imbalance and imbalance.get("is_imbalanced"):
				st.markdown(TT("### ⚖️ **数据不平衡警告**", "### ⚖️ Imbalance warning"))
				ratio = imbalance.get("ratio")
				if ratio:
					st.warning(TT(f"检测到数据不平衡，主要类别占比: {ratio:.1%}", f"Detected imbalance; majority class ratio: {ratio:.1%}"))
					st.caption(TT("建议考虑使用类别权重平衡或采样技术", "Consider class weights or sampling techniques"))
		
		# 显示可读化结果
		display_readable_plan(plan)
		
		# 可选显示原始 JSON
		with st.expander(TT("🔍 查看详细 JSON 结果", "🔍 View detailed JSON"), expanded=False):
			st.json(plan)
		
		st.caption(TT("💡 以上结果将自动应用到训练设置中，你也可以手动调整参数。", "💡 These results will auto-apply to training; you can adjust manually."))

	# 已移除“目标列与特征选择建议”功能与应用入口

	# 显示 数据清洗建议（合并到以上两个流程后展示）
	if st.session_state.get("cleaning_suggest"):
		st.subheader(TT("🧹 数据清洗建议", "🧹 Cleaning Suggestions"))
		cs = st.session_state["cleaning_suggest"]
		# Drop
		drops = cs.get("drop_columns", [])
		with st.expander(TT(f"🗑️ 建议删除列（{len(drops)}）", f"🗑️ Suggested drop columns ({len(drops)})"), expanded=False):
			if drops:
				for d in drops:
					st.write(f"- {d.get('name')}: {d.get('reason','')}")
			else:
				st.write(TT("无", "None"))
		# Imputations
		imps = cs.get("imputations", [])
		with st.expander(TT(f"🧩 缺失值填充建议（{len(imps)}）", f"🧩 Imputation suggestions ({len(imps)})"), expanded=False):
			if imps:
				for d in imps:
					st.write(f"- {d.get('name')}: {d.get('strategy')}")
			else:
				st.write(TT("无", "None"))
		# Type casts and parse dates
		casts = cs.get("type_casts", [])
		pdates = cs.get("parse_dates", [])
		with st.expander(TT(f"🧭 类型转换建议（{len(casts)}）/ 日期解析（{len(pdates)}）", f"🧭 Type casts ({len(casts)}) / Parse dates ({len(pdates)})"), expanded=False):
			if casts:
				for d in casts:
					st.write(f"- {d.get('name')} -> {d.get('to_dtype')}: {d.get('reason','')}")
			else:
				st.write(TT("类型转换：无", "Type casts: None"))
			if pdates:
				st.write(TT("日期解析：", "Parse dates:"))
				st.code("\n".join(pdates))
			else:
				st.write(TT("日期解析：无", "Parse dates: None"))
		# Scaling
		scaling = cs.get("scaling", {}) or {}
		with st.expander(TT("📐 缩放建议", "📐 Scaling"), expanded=False):
			st.write(TT(f"建议缩放: {'是' if scaling.get('apply') else '否'}", f"Scale recommended: {'Yes' if scaling.get('apply') else 'No'}"))
			sc_cols = scaling.get("columns", [])
			if sc_cols:
				st.code("\n".join(sc_cols))
			else:
				st.write(TT("列：无", "Columns: None"))
		# Outliers
		outliers = cs.get("outliers", {}) or {}
		with st.expander(TT("📉 异常值处理建议", "📉 Outlier handling"), expanded=False):
			st.write(TT(f"建议处理: {'是' if outliers.get('apply') else '否'}; 方法: {outliers.get('method','iqr_clip')}", f"Apply handling: {'Yes' if outliers.get('apply') else 'No'}; Method: {outliers.get('method','iqr_clip')}"))
			out_cols = outliers.get("columns", [])
			if out_cols:
				st.code("\n".join(out_cols))
			else:
				st.write(TT("列：无", "Columns: None"))
		# Text processing
		txts = cs.get("text_processing", [])
		with st.expander(TT(f"📝 文本处理建议（{len(txts)}）", f"📝 Text processing suggestions ({len(txts)})"), expanded=False):
			if txts:
				for d in txts:
					st.write(f"- {d.get('name')}: {d.get('suggestion')}")
			else:
				st.write(TT("无", "None"))
		# Leakage
		leaks = cs.get("leakage_risk", [])
		with st.expander(TT(f"⚠️ 可能的泄露风险列（{len(leaks)}）", f"⚠️ Potential leakage columns ({len(leaks)})"), expanded=False):
			if leaks:
				st.code("\n".join(leaks))
			else:
				st.write(TT("无", "None"))
		st.caption(cs.get("notes") or "")

		# ============ 一键与手动清洗操作 ============
		st.markdown("---")
		st.markdown(TT("### ⚙️ 应用清洗操作", "### ⚙️ Apply cleaning operations"))

		# 当前用于训练/分析的数据
		work_df = st.session_state.get("train_df", analysis_df)
		work_name = st.session_state.get("train_source_name", analysis_df_name)
		st.caption(TT(f"清洗目标数据集：{work_name} 形状：{work_df.shape}", f"Dataset to clean: {work_name} Shape: {work_df.shape}"))

		# 一键应用 GPT 建议
		def _iqr_clip_inline(df, cols, whisker: float = 1.5):
			import numpy as np
			df = df.copy()
			for c in cols:
				if c in df.columns:
					try:
						s = pd.to_numeric(df[c], errors='coerce')
						q1, q3 = s.quantile(0.25), s.quantile(0.75)
						iqr = q3 - q1
						low, high = q1 - whisker * iqr, q3 + whisker * iqr
						df[c] = s.clip(lower=low, upper=high)
					except Exception:
						pass
			return df

		def _apply_type_casts(df, casts):
			df = df.copy()
			for item in casts or []:
				name = item.get('name')
				to_dtype = (item.get('to_dtype') or '').lower()
				if name not in df.columns:
					continue
				try:
					if to_dtype in ('float','float64','double','number'):
						df[name] = pd.to_numeric(df[name], errors='coerce')
					elif to_dtype in ('int','int64','long'):
						df[name] = pd.to_numeric(df[name], errors='coerce').astype('Int64')
					elif to_dtype in ('bool','boolean'):
						df[name] = df[name].astype('boolean')
					elif to_dtype in ('category','categorical'):
						df[name] = df[name].astype('category')
					elif to_dtype in ('string','str','object'):
						df[name] = df[name].astype('string')
					# else: leave as is
				except Exception:
					pass
			return df

		def _apply_imputations(df, imputations):
			df = df.copy()
			for item in imputations or []:
				name = item.get('name')
				strategy = (item.get('strategy') or 'most_frequent').lower()
				if name not in df.columns:
					continue
				try:
					if strategy == 'median':
						val = pd.to_numeric(df[name], errors='coerce').median()
						df[name] = pd.to_numeric(df[name], errors='coerce').fillna(val)
					elif strategy == 'mean':
						val = pd.to_numeric(df[name], errors='coerce').mean()
						df[name] = pd.to_numeric(df[name], errors='coerce').fillna(val)
					else:  # most_frequent
						val = df[name].mode(dropna=True)
						val = val.iloc[0] if len(val) else None
						if val is not None:
							df[name] = df[name].fillna(val)
				except Exception:
					pass
			return df

		col_btn1, col_btn2 = st.columns([1,2])
		with col_btn1:
			if st.button(TT("⚡ 一键应用 GPT 清洗建议", "⚡ Apply GPT cleaning suggestions")):
				try:
					new_df = work_df.copy()
					# Drop
					to_drop = [d.get('name') for d in (cs.get('drop_columns') or []) if d.get('name') in new_df.columns]
					if to_drop:
						new_df = new_df.drop(columns=to_drop, errors='ignore')
					# Parse dates
					for c in (cs.get('parse_dates') or []):
						if c in new_df.columns:
							try:
								new_df[c] = pd.to_datetime(new_df[c], errors='coerce')
							except Exception:
								pass
					# Type casts
					new_df = _apply_type_casts(new_df, cs.get('type_casts'))
					# Imputations
					new_df = _apply_imputations(new_df, cs.get('imputations'))
					# Outliers
					out_cols = []
					out_meta = cs.get('outliers') or {}
					if isinstance(out_meta, dict) and out_meta.get('apply'):
						out_cols = [c for c in (out_meta.get('columns') or []) if c in new_df.columns]
					if out_cols:
						new_df = _iqr_clip_inline(new_df, out_cols)

					st.session_state["train_df"] = new_df
					st.session_state["train_source_name"] = TT(f"{work_name}（已按GPT建议清洗）", f"{work_name} (cleaned by GPT suggestions)")
					st.success(TT(f"已应用 GPT 清洗建议，当前形状：{new_df.shape}", f"Applied GPT cleaning; current shape: {new_df.shape}"))
				except Exception as e:
					st.error(TT(f"应用失败：{e}", f"Apply failed: {e}"))

		with col_btn2:
			st.caption(TT("或手动选择以下清洗操作：", "Or manually select operations below:"))
			# 手动选择
			all_cols = list(work_df.columns)
			default_drop = [d.get('name') for d in (cs.get('drop_columns') or []) if d.get('name') in all_cols]
			pick_drop = st.multiselect(TT("要删除的列", "Columns to drop"), options=all_cols, default=default_drop)

			default_dates = [c for c in (cs.get('parse_dates') or []) if c in all_cols]
			pick_dates = st.multiselect(TT("要解析为日期的列", "Columns to parse as dates"), options=all_cols, default=default_dates)

			out_meta = cs.get('outliers') or {}
			default_out = [c for c in (out_meta.get('columns') or []) if c in all_cols]
			pick_outliers = st.multiselect(TT("IQR 裁剪的数值列", "Numeric columns for IQR clipping"), options=all_cols, default=default_out)

			apply_imp = st.checkbox(TT("按建议填充缺失值（数值: 中位/均值；类别: 众数）", "Impute missing values as suggested (numeric: median/mean; categorical: mode)"), value=True, key="apply_imputations_suggested")
			apply_casts = st.checkbox(TT("按建议进行类型转换", "Apply suggested type casts"), value=True, key="apply_type_casts_suggested")

			if st.button(TT("🛠️ 应用选中清洗操作", "🛠️ Apply selected operations")):
				try:
					new_df = work_df.copy()
					if pick_drop:
						new_df = new_df.drop(columns=pick_drop, errors='ignore')
					for c in pick_dates:
						if c in new_df.columns:
							try:
								new_df[c] = pd.to_datetime(new_df[c], errors='coerce')
							except Exception:
								pass
					if apply_casts:
						new_df = _apply_type_casts(new_df, cs.get('type_casts'))
					if apply_imp:
						new_df = _apply_imputations(new_df, cs.get('imputations'))
					if pick_outliers:
						new_df = _iqr_clip_inline(new_df, pick_outliers)
					st.session_state["train_df"] = new_df
					st.session_state["train_source_name"] = TT(f"{work_name}（已按手动清洗）", f"{work_name} (cleaned manually)")
					st.success(TT(f"已应用手动清洗，当前形状：{new_df.shape}", f"Applied manual cleaning; current shape: {new_df.shape}"))
				except Exception as e:
						st.error(TT(f"应用失败：{e}", f"Apply failed: {e}"))

		# ============ 清洗后数据可视化 ============
		st.markdown("---")
		st.markdown(TT("### 📊 数据可视化（清洗后）", "### 📊 Visualization (after cleaning)"))
		viz_df = st.session_state.get("train_df", work_df)
		viz_name = st.session_state.get("train_source_name", work_name)
		st.caption(TT(f"基于清洗后的数据：{viz_name} ；形状：{viz_df.shape}", f"Using cleaned data: {viz_name} ; shape: {viz_df.shape}"))

		enable_viz = st.checkbox(TT("启用可视化", "Enable visualization"), value=True, key="enable_viz_after_clean")
		if enable_viz and viz_df is not None and len(viz_df.columns) > 0:
			try:
				import altair as alt
				_has_altair = True
			except Exception:
				alt = None
				_has_altair = False
			col_left, col_right = st.columns([1,2])
			with col_left:
				picked_col = st.selectbox(TT("选择要可视化的列", "Column to visualize"), options=list(viz_df.columns), key="viz_col_select")
				if picked_col is not None:
					series = viz_df[picked_col]
					is_num = pd.api.types.is_numeric_dtype(series)
					is_dt = pd.api.types.is_datetime64_any_dtype(series)
					if is_num:
						bins = st.slider(TT("直方图分箱数", "Histogram bins"), min_value=5, max_value=100, value=30, step=5, key="viz_bins")
					elif is_dt:
						freq = st.selectbox(TT("时间聚合粒度", "Time aggregation"), ["D","W","M"], index=0, help=TT("按天/周/月统计计数", "Group by Day/Week/Month"), key="viz_dt_freq")
					else:
						topk = st.slider(TT("类别Top N", "Top-N categories"), min_value=5, max_value=100, value=20, step=5, key="viz_topk")

			with col_right:
				if 'picked_col' in locals() and picked_col is not None:
					series = viz_df[picked_col]
					is_num = pd.api.types.is_numeric_dtype(series)
					is_dt = pd.api.types.is_datetime64_any_dtype(series)
					if is_num:
						# 数值：直方图 + 箱线图（无 Altair 时退化为柱状图）
						base_df = pd.DataFrame({picked_col: pd.to_numeric(series, errors='coerce')})
						if _has_altair:
							hist = alt.Chart(base_df).mark_bar().encode(
								alt.X(f"{picked_col}:Q", bin=alt.Bin(maxbins=bins)),
								y='count()'
							).properties(height=220)
							box = alt.Chart(base_df).mark_boxplot().encode(x=alt.X(f"{picked_col}:Q")).properties(height=120)
							st.altair_chart(hist & box, use_container_width=True)
						else:
							# 计算直方图数据并用原生 bar_chart 展示
							try:
								import numpy as np
								counts, bin_edges = np.histogram(base_df[picked_col].dropna(), bins=bins)
								centers = (bin_edges[:-1] + bin_edges[1:]) / 2
								df_hist = pd.DataFrame({"bin": centers, "count": counts})
								st.bar_chart(df_hist.set_index("bin")["count"])
							except Exception:
								st.line_chart(base_df[picked_col])
					elif is_dt:
						# 时间：按粒度计数
						df_dt = pd.DataFrame({picked_col: pd.to_datetime(series, errors='coerce')}).dropna()
						if not df_dt.empty:
							df_dt['__bucket__'] = df_dt[picked_col].dt.to_period(freq).dt.start_time
							cnt = df_dt.groupby('__bucket__').size().reset_index(name='count')
							if _has_altair:
								chart = alt.Chart(cnt).mark_bar().encode(x='__bucket__:T', y='count:Q').properties(height=260)
								st.altair_chart(chart, use_container_width=True)
							else:
								cnt = cnt.set_index('__bucket__')
								st.bar_chart(cnt['count'])
						else:
							st.info(TT("所选列无法解析为有效时间格式。", "Selected column cannot be parsed as valid datetime."))
					else:
						# 类别：TopK 频次条形图
						vc = series.astype('string').value_counts().reset_index()
						vc.columns = [picked_col, 'count']
						vc = vc.head(topk)
						if _has_altair:
							chart = alt.Chart(vc).mark_bar().encode(
								y=alt.Y(f"{picked_col}:N", sort='-x'),
								x=alt.X('count:Q')
							).properties(height=max(200, 16*len(vc)))
							st.altair_chart(chart, use_container_width=True)
						else:
							st.bar_chart(vc.set_index(picked_col)['count'])
		else:
			st.info(TT("无可视化数据可用或未选择列。", "No data available for visualization or no column selected."))

	# ------------------ 训练设置与训练流程（最小接入） ------------------
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from core import cleandata, train as train_core

	st.subheader(TT("🛠️ 训练设置（本地）", "🛠️ Training Settings (local)"))

	# 训练数据来源选择：默认使用最新合并结果（若存在），否则使用当前活动数据集
	options = []
	if "train_df" in st.session_state and "train_source_name" in st.session_state:
		options.append(TT(f"最新合并结果（{st.session_state['train_source_name']}）", f"Latest merged result ({st.session_state['train_source_name']})"))
	options.append(TT(f"当前活动数据集（{df_source_name}）", f"Active dataset ({df_source_name})"))
	default_idx = 0 if options and options[0].startswith("最新合并结果") else 0
	selected_source = st.radio(TT("训练数据来源", "Training data source"), options, index=default_idx, horizontal=True)
	if (selected_source.startswith("最新合并结果") or selected_source.startswith("Latest merged")) and "train_df" in st.session_state:
		train_df = st.session_state["train_df"]
		train_source_name = st.session_state.get("train_source_name", TT("最新合并结果", "Latest merged result"))
	else:
		train_df = df
		train_source_name = df_source_name
	st.info(TT(f"训练数据集：{train_source_name}；形状：{train_df.shape}", f"Training dataset: {train_source_name}; shape: {train_df.shape}"))
	
	# 智能应用 AI 判定结果
	plan = st.session_state.get("plan", {})
	ai_targets = plan.get("target_candidates", [])
	ai_task_type = plan.get("task_type", "")
	ai_algorithms = plan.get("algorithms", [])
	ai_cv = plan.get("cv", {})
	
	# 目标列选择 - 优先使用 AI 推荐
	available_columns = [c for c in train_df.columns if c != ""]
	if ai_targets and ai_targets[0] in available_columns:
		default_target_index = available_columns.index(ai_targets[0])
		st.success(TT(f"🤖 AI 推荐目标列: {ai_targets[0]}", f"🤖 AI suggested target: {ai_targets[0]}"))
	else:
		default_target_index = 0
	
	target = st.selectbox(TT("目标列", "Target column"), options=available_columns, index=default_target_index)
	
	# 自动推荐任务类型
	if target:
		target_series = train_df[target]
		target_series = target_series.dropna()  # 去除缺失值
		
		# 检查是否为数值类型
		is_numeric = pd.api.types.is_numeric_dtype(target_series)
		unique_count = target_series.nunique()
		total_count = len(target_series)
		
		# 推荐逻辑
		if is_numeric and unique_count > 20 and unique_count / total_count > 0.05:
			recommended_task = "regression"
			reason = TT(f"数值类型，{unique_count} 个唯一值", f"Numeric with {unique_count} unique values")
		else:
			recommended_task = "classification" 
			if not is_numeric:
				reason = TT(f"非数值类型，{unique_count} 个类别", f"Non-numeric with {unique_count} categories")
			else:
				reason = TT(f"数值类型但只有 {unique_count} 个唯一值，可能是分类", f"Numeric with only {unique_count} unique values; likely classification")
		
		st.info(TT(f"🤖 推荐任务类型: **{recommended_task}** ({reason})", f"🤖 Suggested task: **{recommended_task}** ({reason})"))
		
		# 显示目标变量的基本统计
		col1, col2 = st.columns(2)
		with col1:
			st.metric(TT("唯一值数量", "Unique values"), unique_count)
		with col2:
			st.metric(TT("样本数量", "Samples"), total_count)
		
		if unique_count <= 10:
			st.write(TT("目标变量的值分布:", "Target value distribution:"))
			value_counts = target_series.value_counts().head(10)
			st.bar_chart(value_counts)
	
	# 任务类型选择 - 智能应用 AI 推荐
	task_options = ["classification", "regression"]
	if ai_task_type and ai_task_type in task_options:
		default_task_index = task_options.index(ai_task_type)
		st.success(TT(f"🤖 AI 推荐任务类型: {ai_task_type}", f"🤖 AI suggested task: {ai_task_type}"))
	else:
		default_task_index = 0
	
	task_type = st.selectbox(TT("任务类型", "Task type"), task_options, index=default_task_index)
	
	# 算法选择 - 智能应用 AI 推荐
	available_algos = ["logreg","rf","xgb","knn","mlp"] if task_type=="classification" else ["linreg","ridge","rf","xgb","knn","mlp"]
	
	# 映射 AI 推荐的算法名称到本地算法名称
	algo_mapping = {
		"xgboost": "xgb",
		"random_forest": "rf", 
		"ridge": "ridge",
		"knn": "knn",
		"logistic_regression": "logreg",
		"linear_regression": "linreg",
		"mlp": "mlp"
	}
	
	ai_algos_mapped = []
	if ai_algorithms:
		for ai_algo in ai_algorithms:
			mapped_algo = algo_mapping.get(ai_algo, ai_algo)
			if mapped_algo in available_algos:
				ai_algos_mapped.append(mapped_algo)
		
		if ai_algos_mapped:
			st.success(TT(f"🤖 AI 推荐算法: {', '.join(ai_algos_mapped)}", f"🤖 AI suggested algorithms: {', '.join(ai_algos_mapped)}"))
			default_algos = ai_algos_mapped
		else:
			default_algos = ["rf","xgb"]
	else:
		default_algos = ["rf","xgb"]
	
	picked = st.multiselect(
		TT("候选算法", "Candidate algorithms"),
		available_algos,
		default=default_algos
	)
	budget = st.slider(TT("每模型搜索次数 (n_iter)", "Search iterations per model (n_iter)"), 10, 80, 30)
	
	# 交叉验证折数 - 智能应用 AI 推荐
	default_folds = ai_cv.get("folds", 5) if ai_cv else 5
	if ai_cv and "folds" in ai_cv:
		st.success(TT(f"🤖 AI 推荐 CV 折数: {default_folds}", f"🤖 AI suggested CV folds: {default_folds}"))
	
	folds = st.slider(TT("CV 折数", "CV folds"), 3, 10, default_folds)


	# 评估行数限制设置
	with st.expander(TT("⚙️ 评估数据量设置", "⚙️ Evaluation data limit"), expanded=False):
		col_a, col_b = st.columns([1,2])
		with col_a:
			use_eval_limit = st.checkbox(TT("限制评估行数", "Limit evaluation rows"), value=True, key="use_eval_limit_rows", help=TT("仅在评估指标/预测时使用测试集前 N 行，适合快速迭代。", "Use first N rows of test set only for evaluation/prediction; faster iteration."))
		with col_b:
			if use_eval_limit:
				custom_eval_rows = st.number_input(TT("评估最大行数 N", "Max evaluation rows N"), min_value=50, max_value=20000, value=500, step=50, help=TT("超过该行数时仅截取前 N 行；不影响模型训练。", "If larger, only take first N rows; training unaffected."))
			else:
				custom_eval_rows = None

	if st.button(TT("开始训练", "Start training")):
		X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(train_df, target, task_type)
		leaderboard, artifacts = train_core.run_all(
			X_train, y_train, X_test, y_test,
			task_type=task_type,
			picked_models=picked,
			preprocessor=pre,
			n_iter=budget,
			cv_folds=folds,
			artifacts_dir="artifacts"
		)
		st.success(TT("训练完成！", "Training completed!"))
		st.dataframe(leaderboard)
		# 记录本次训练上下文，供后续一致性检查使用
		st.session_state["__trained_target__"] = target
		st.session_state["__trained_algos__"] = picked
		# 根据用户设置限制评估阶段使用的测试集行数
		if custom_eval_rows and custom_eval_rows > 0 and len(X_test) > custom_eval_rows:
			n_eval = int(min(custom_eval_rows, len(X_test)))
			if hasattr(X_test, 'head'):
				try:
					X_test_eval = X_test.head(n_eval)
				except Exception:
					X_test_eval = X_test[:n_eval]
			else:
				X_test_eval = X_test[:n_eval]
			if hasattr(y_test, 'iloc'):
				y_test_eval = y_test.iloc[:n_eval]
			else:
				y_test_eval = y_test[:n_eval]
			st.info(TT(f"评估行数限制启用：使用测试集前 {n_eval} 行（原始 {len(X_test)} 行）。", f"Eval row limit enabled: using first {n_eval} rows (original {len(X_test)})"))
		else:
			X_test_eval = X_test
			y_test_eval = y_test
			st.info(TT(f"评估行数限制未启用，使用全部测试集 {len(X_test)} 行。", f"Eval row limit disabled: using all {len(X_test)} test rows."))
		st.session_state["__eval_pack__"] = (task_type, X_test_eval, y_test_eval, artifacts)

		# 存储排行榜用于报告生成
		st.session_state["leaderboard_df"] = leaderboard

		# 展示排行榜关键指标与模型下载
		st.markdown(TT("### 🏁 训练结果与模型导出", "### 🏁 Training Results & Model Export"))
		# 关键指标摘要
		try:
			if task_type == "classification":
				metric_cols = [c for c in ["acc","f1_macro","roc_auc"] if c in leaderboard.columns]
			else:
				metric_cols = [c for c in ["rmse","mae","r2"] if c in leaderboard.columns]
			if metric_cols:
				st.caption(TT(f"展示关键评估指标：{', '.join(metric_cols)}", f"Showing key metrics: {', '.join(metric_cols)}"))
				st.dataframe(leaderboard[["model","cv_score(primary)",*metric_cols,"fit_s","predict_s","params"]])
		except Exception:
			st.dataframe(leaderboard)

		# 提供各模型文件下载
		with st.expander(TT("⬇️ 下载最佳模型文件", "⬇️ Download best model files"), expanded=True):
			for mname, info in (artifacts or {}).items():
				mpath = info.get("model_path")
				st.write(TT(f"模型：{mname}", f"Model: {mname}"))
				if mpath and os.path.exists(mpath):
					try:
						with open(mpath, "rb") as fh:
							st.download_button(TT(f"下载 {os.path.basename(mpath)}", f"Download {os.path.basename(mpath)}"), data=fh.read(), file_name=os.path.basename(mpath))
					except Exception as e:
						st.warning(TT(f"无法提供下载：{e}", f"Cannot provide download: {e}"))
				else:
					st.info(TT("模型文件尚未生成或路径不可用。", "Model file not generated or path unavailable."))


		# （分析按钮移至全局区块，保证重新运行后仍可使用）

	# ------------------ 报告生成（OpenAI） ------------------
	# 全局显示训练结果分析区块（若已有 leaderboard），避免按钮只在训练当次出现
	leaderboard_existing = st.session_state.get("leaderboard_df")
	if leaderboard_existing is not None:
		st.markdown(TT("### 🔎 训练结果分析 (LLM)", "### 🔎 Training Result Analysis (LLM)"))
		# 安全获取 artifacts 与 task_type
		_eval_pack = st.session_state.get("__eval_pack__") or (None, None, None, {})
		_artifacts = _eval_pack[3] if isinstance(_eval_pack, (list, tuple)) and len(_eval_pack) == 4 else {}
		_task_type_for_analysis = (st.session_state.get("plan") or {}).get("task_type") or _eval_pack[0] or "classification"
		_plan_obj = st.session_state.get("plan")
		col_an1, col_an2 = st.columns([1,2])
		with col_an1:
			if st.button(TT("🧠 分析训练结果", "🧠 Analyze training results"), key="btn_analyze_training_global"):
				from llm_agent import analyze_training_results
				with st.spinner(TT("AI 正在分析训练排行榜...", "AI analyzing leaderboard...")):
					analysis = analyze_training_results(
						leaderboard_existing,
						_artifacts,
						_task_type_for_analysis,
						_plan_obj,
						lang=st.session_state.get("lang", "zh")
					)
				st.session_state["__training_analysis__"] = analysis
				st.success(TT("分析完成", "Analysis complete"))
		with col_an2:
			if st.session_state.get("__training_analysis__"):
				an = st.session_state["__training_analysis__"]
				st.markdown(an.get("markdown","(no analysis)"))
				st.download_button(
					TT("⬇️ 下载训练分析 Markdown", "⬇️ Download training analysis"),
					data=an.get("markdown",""),
					file_name="training_analysis.md",
					mime="text/markdown"
				)

		# —— 新增：与推荐研究问题一致性检查 ——
		st.markdown(TT("### ✅ 与研究问题的一致性检查", "### ✅ Alignment with Research Questions"))
		rs_suggest = st.session_state.get("research_suggestions")
		col_chk1, col_chk2 = st.columns([1,2])
		with col_chk1:
			if st.button(TT("对齐检查", "Run alignment check"), key="btn_alignment_check"):
				from llm_agent import check_research_alignment
				with st.spinner(TT("正在对比训练结果与推荐研究问题...", "Comparing training against research questions...")):
					align = check_research_alignment(
						leaderboard_existing,
						_artifacts,
						_task_type_for_analysis,
						rs_suggest,
						trained_target=st.session_state.get("__trained_target__"),
						picked_models=st.session_state.get("__trained_algos__"),
						lang=st.session_state.get("lang","zh"),
					)
				st.session_state["__alignment_report__"] = align
				st.success(TT("对齐检查完成", "Alignment check complete"))
		with col_chk2:
			if st.session_state.get("__alignment_report__"):
				rep = st.session_state["__alignment_report__"]
				st.markdown(rep.get("markdown", "(no alignment result)"))
				st.download_button(
					TT("⬇️ 下载对齐报告 Markdown", "⬇️ Download alignment report"),
					data=rep.get("markdown", ""),
					file_name="alignment_with_research_questions.md",
					mime="text/markdown"
				)

		# —— 新增：基于训练结果“回答”研究问题 ——
		st.markdown(TT("### 💬 回答研究问题", "### 💬 Answer Research Questions"))
		col_ans1, col_ans2 = st.columns([1,2])
		with col_ans1:
			if st.button(TT("生成回答", "Generate answers"), key="btn_answer_questions"):
				from llm_agent import answer_research_questions
				with st.spinner(TT("正在汇总最佳模型指标并作答...", "Summarizing best model metrics to answer...")):
					ans = answer_research_questions(
						research_suggestions=rs_suggest or {},
						profile=st.session_state.get("profile_for_report"),
						leaderboard=leaderboard_existing,
						artifacts=_artifacts,
						task_type=_task_type_for_analysis,
						trained_target=st.session_state.get("__trained_target__"),
						lang=st.session_state.get("lang","zh"),
					)
				st.session_state["__rq_answers__"] = ans
				st.success(TT("研究问题回答已生成", "Research question answers generated"))
		with col_ans2:
			if st.session_state.get("__rq_answers__"):
				ans = st.session_state["__rq_answers__"]
				st.markdown(ans.get("markdown", "(no answers)"))
				st.download_button(
					TT("⬇️ 下载回答 Markdown", "⬇️ Download answers"),
					data=ans.get("markdown", ""),
					file_name="research_questions_answers.md",
					mime="text/markdown"
				)
	else:
		st.info(TT("尚未训练，训练结果分析按钮将在训练完成后出现。", "No training yet; analysis button will appear after training."))

	st.markdown("---")
	st.subheader(TT("📄 生成总结报告（OpenAI）", "📄 Generate Summary Report (OpenAI)"))

	# 组装报告上下文
	bundle = {
		"meta": {
			"dataset_name": st.session_state.get("train_source_name", df_source_name),
		},
		"profile": {
			"columns": [
				{"name": c, "dtype": str((st.session_state.get("train_df", df))[c].dtype)}
				for c in (st.session_state.get("train_df", df)).columns
			]
		},
		"research_suggestions": st.session_state.get("research_suggestions"),
		"plan": st.session_state.get("plan"),
		"cleaning_suggest": st.session_state.get("cleaning_suggest"),
		"research_suggestions": st.session_state.get("research_suggestions"),
		"leaderboard": st.session_state.get("leaderboard_df"),
		"artifacts": st.session_state.get("__eval_pack__", (None, None, None, {}))[3],
	}

	col_r1, col_r2 = st.columns([1,2])
	with col_r1:
		if st.button(TT("🧠 使用 OpenAI 生成报告", "🧠 Generate report via OpenAI")):
			with st.spinner(TT("正在生成报告…", "Generating report…")):
				from llm_agent import write_report
				report_md = write_report(bundle, lang=st.session_state.get("lang","zh"))
				st.session_state["__final_report_md__"] = report_md
			st.success(TT("报告已生成！", "Report generated!"))

	with col_r2:
		if st.session_state.get("__final_report_md__"):
			st.markdown(st.session_state["__final_report_md__"])
			st.download_button(
				TT("⬇️ 下载报告 Markdown", "⬇️ Download report (Markdown)"),
				data=st.session_state["__final_report_md__"],
				file_name="automl_report.md",
				mime="text/markdown"
			)
