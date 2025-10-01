import streamlit as st
import os
import json
import sys
sys.path.append(os.path.dirname(__file__))
from llm_agent import detect_task
st.set_page_config(page_title="LLM-Augmented AutoML", layout="wide")
import pandas as pd

st.title("LLM-Augmented AutoML (Local Training)")
st.success("环境初始化成功。接下来将实现数据上传、判定与报告。")

"""多文件上传区"""
uploaded_files = st.file_uploader("上传一个或多个 CSV 文件", type=["csv"], accept_multiple_files=True)

active_df = None
df_source_name = None
loaded_dfs = {}

if uploaded_files:
	examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
	os.makedirs(examples_dir, exist_ok=True)
	for uf in uploaded_files:
		try:
			df_tmp = pd.read_csv(uf)
			loaded_dfs[uf.name] = df_tmp
			# 保存文件
			save_path = os.path.join(examples_dir, uf.name)
			with open(save_path, 'wb') as f:
				f.write(uf.getbuffer())
		except Exception as e:
			st.error(f"读取文件 {uf.name} 失败: {e}")

	st.success(f"成功载入 {len(loaded_dfs)} 个文件。")

	# -------- 新增：显示多个文件共同拥有的列名（公共列） --------
	if len(loaded_dfs) >= 2:
		# 计算所有数据集的列集合交集
		list_of_colsets = [set(df.columns) for df in loaded_dfs.values() if hasattr(df, 'columns')]
		if list_of_colsets:  # 防御性检查
			common_columns = set.intersection(*list_of_colsets) if len(list_of_colsets) > 1 else list_of_colsets[0]
		else:
			common_columns = set()

		with st.expander("📌 多文件公共列 (所有文件都包含)", expanded=True):
			if common_columns:
				st.write(f"共 {len(common_columns)} 个公共列：")
				# 排序便于浏览
				st.code("\n".join(sorted(common_columns)))
			else:
				st.warning("未找到所有文件都共同拥有的列。")

			# 提示：可选展示每个文件缺失的列（帮助用户理解差异）
			show_diff = st.checkbox("显示各文件缺失公共列情况", value=False)
			if show_diff and common_columns:
				for fname, df_tmp in loaded_dfs.items():
					missing_in_file = common_columns - set(df_tmp.columns)
					if missing_in_file:
						st.error(f"{fname} 缺失 {len(missing_in_file)} 个公共列：{', '.join(sorted(missing_in_file))}")
					else:
						st.success(f"{fname} 包含全部公共列 ✔")

			# ---------------- 合并功能（新增：横向匹配模式） ----------------
			st.markdown("---")
			st.markdown("### 🔗 合并工具")
			merge_mode = st.radio(
				"选择合并方式",
				["纵向堆叠（仅公共列）", "横向匹配（公共列作为键，合并其余列）"],
				index=0,
				help="横向匹配=类似多表 join；纵向堆叠=append 行。"
			)

			if merge_mode.startswith("纵向"):
				st.caption("仅保留公共列并按行堆叠（之前的行为）。")
				add_source_col = st.checkbox("添加来源文件列 (_source_file)", value=True, key="add_source_file")
				merge_btn = st.button("⚙️ 执行纵向合并", key="merge_vertical")
				if merge_btn:
					if not common_columns:
						st.error("无法合并：没有公共列。")
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
								st.warning(f"合并文件保存失败，但内存依旧可用：{fs_err}")
							loaded_dfs[final_name] = merged_df
							st.session_state["merged_common_df"] = merged_df
							st.success(f"纵向合并成功：{final_name}，形状 {merged_df.shape}")
							csv_bytes = merged_df.to_csv(index=False).encode('utf-8')
							st.download_button("⬇️ 下载结果", data=csv_bytes, file_name=final_name, mime="text/csv")
							st.info("在下方文件选择框中可选择该合并文件继续分析。")
						except Exception as merge_err:
							st.error(f"合并失败：{merge_err}")

			else:  # 横向匹配
				st.caption("使用公共列作为键做多表 join，保留每个文件的其余列。")
				if not common_columns:
					st.error("无法进行横向匹配：没有公共列。")
				else:
					key_cols = sorted(common_columns)
					st.info(f"键列：{', '.join(key_cols)}")
					join_type = st.selectbox("Join 类型", ["outer", "inner", "left"], index=0, help="outer=保留所有键; inner=仅公共键; left=以第一个文件为主表")
					prefix_cols = st.checkbox("为非键列加文件名前缀以防冲突", value=True, key="prefix_non_key")
					drop_dup = st.checkbox("如果某文件键列有重复行，仅保留第一条", value=True, key="drop_dup_keys")
					btn_hmerge = st.button("⚙️ 执行横向匹配合并", key="merge_horizontal")
					if btn_hmerge:
						try:
							merged_df = None
							for idx_file, (fname, df_part) in enumerate(loaded_dfs.items()):
								work_df = df_part.copy()
								missing_keys = [k for k in key_cols if k not in work_df.columns]
								if missing_keys:
									st.error(f"文件 {fname} 缺失键列 {missing_keys}，跳过。")
									continue
								# 处理重复键
								if drop_dup and work_df.duplicated(subset=key_cols).any():
									dup_count = work_df.duplicated(subset=key_cols).sum()
									st.warning(f"{fname} 键列存在 {dup_count} 个重复，将保留第一条。")
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
								st.error("未能生成合并结果（可能所有文件都被跳过）")
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
									st.warning(f"合并文件保存失败，但内存仍可使用：{fs_err}")
								loaded_dfs[final_name] = merged_df
								st.success(f"横向匹配合并成功：{final_name}，形状 {merged_df.shape}")
								csv_bytes = merged_df.to_csv(index=False).encode('utf-8')
								st.download_button("⬇️ 下载结果", data=csv_bytes, file_name=final_name, mime="text/csv")
								st.info("在下方文件选择框中可选择该横向合并文件继续分析。")
						except Exception as e:
							st.error(f"横向合并失败：{e}")
	else:
		st.info("上传 2 个及以上文件后，将在此显示它们的公共列。")

	file_names = list(loaded_dfs.keys())
	pick_name = st.selectbox("选择一个文件进行预览与分析", file_names)
	active_df = loaded_dfs.get(pick_name)
	df_source_name = pick_name

if active_df is not None:
	df = active_df  # 保持后续代码变量名不变
	st.info(f"当前活动数据集: {df_source_name}; 形状: {df.shape}")
	st.write("数据预览：")
	st.dataframe(df.head())

	#（已移除多文件主列匹配功能）
	with st.expander("�🔎 数据概览", expanded=False):
		st.write("数据描述：")
		st.write(df.describe(include='all').transpose())
		st.write("缺失值统计：")
		st.write(df.isnull().sum())

	# ----------- 判定按钮与结果展示区块 -------------
	# 构建简易 profile；后续可替换为 ingest.profile
	prof = {
		"columns": [
			{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isnull().sum()), "unique": int(df[c].nunique())}
			for c in df.columns
		]
	}

	user_question = st.text_area("你的问题（可选）", placeholder="例如：我们能否预测乘客是否生还？或 预测价格/分群等。")

	col1, col2, col3 = st.columns([1,1,1])
	with col1:
		if st.button("🔍 发现研究问题"):
			with st.spinner("AI 正在分析数据，寻找有价值的研究问题..."):
				from llm_agent import suggest_research_questions
				research_suggestions = suggest_research_questions(prof)
			st.session_state["research_suggestions"] = research_suggestions
			st.success("问题发现完成！")
	
	with col2:
		if st.button("🤖 智能判定任务"):
			with st.spinner("调用 OpenAI 判定任务类型与方案..."):
				plan = detect_task(prof, user_question or "")
			st.session_state["plan"] = plan
			st.success("任务判定完成！")
	
	with col3:
		if "plan" in st.session_state:
			st.download_button("📄 下载判定结果", data=json.dumps(st.session_state["plan"], ensure_ascii=False, indent=2),
							   file_name="task_plan.json", mime="application/json")

	# 显示研究问题建议
	if "research_suggestions" in st.session_state:
		st.subheader("🔍 AI 数据洞察：可研究的问题")
		
		suggestions = st.session_state["research_suggestions"]
		
		def display_research_suggestions(suggestions):
			"""可读化显示研究问题建议"""
			
			# 检查 suggestions 是否为字典
			if not isinstance(suggestions, dict):
				st.error("❌ 研究建议数据格式错误")
				st.json(suggestions)
				return
			
			# 研究问题
			questions = suggestions.get("research_questions", [])
			if questions:
				st.markdown("### 💡 **推荐研究问题**")
				
				for i, q in enumerate(questions):
					with st.expander(f"📋 问题 {i+1}: {q.get('question', '未知问题')}", expanded=i==0):
						col1, col2 = st.columns(2)
						with col1:
							st.markdown(f"**类型**: {q.get('type', '未知')}")
							st.markdown(f"**难度**: {q.get('difficulty', '未知')}")
						with col2:
							if q.get('target_column'):
								st.markdown(f"**目标列**: `{q.get('target_column')}`")
							methods = q.get('required_methods', [])
							if methods:
								st.markdown(f"**推荐方法**: {', '.join(methods)}")
						
						st.markdown("**商业价值**:")
						st.info(q.get('business_value', '未提供'))
			
			# 应用场景
			scenarios = suggestions.get("application_scenarios", [])
			if scenarios:
				st.markdown("### 🎯 **应用场景**")
				for i, scenario in enumerate(scenarios):
					st.markdown(f"**{i+1}.** {scenario}")
			
			# 关键洞察潜力
			insights = suggestions.get("key_insights_potential", [])
			if insights:
				st.markdown("### 🔮 **可能发现的洞察**")
				for insight in insights:
					st.markdown(f"• {insight}")
			
			# 数据集优势
			strengths = suggestions.get("dataset_strengths", [])
			if strengths:
				st.markdown("### ✨ **数据集优势**")
				for strength in strengths:
					st.markdown(f"✅ {strength}")
			
			# 限制和注意事项
			limitations = suggestions.get("limitations", [])
			if limitations:
				st.markdown("### ⚠️ **注意事项**")
				for limitation in limitations:
					st.warning(f"⚠️ {limitation}")
			
			# 建议
			recommendations = suggestions.get("recommendations", {})
			if recommendations:
				st.markdown("### 🎯 **行动建议**")
				
				# 检查 recommendations 是否为字典
				if isinstance(recommendations, dict):
					priority = recommendations.get("priority_questions", [])
					if priority:
						st.markdown("**🔥 优先研究问题**:")
						for p in priority:
							st.markdown(f"• {p}")
					
					next_steps = recommendations.get("next_steps", [])
					if next_steps:
						st.markdown("**📋 建议步骤**:")
						for step in next_steps:
							st.markdown(f"• {step}")
					
					additional_data = recommendations.get("additional_data", [])
					if additional_data:
						st.markdown("**📊 可能需要的额外数据**:")
						for data in additional_data:
							st.markdown(f"• {data}")
				else:
					# 如果 recommendations 是字符串，直接显示
					st.markdown(f"**💡 建议**: {recommendations}")
		
		display_research_suggestions(suggestions)
		
		# 原始 JSON (可选)
		with st.expander("🔍 查看详细分析结果 (JSON)", expanded=False):
			st.json(suggestions)
		
		st.success("💡 基于以上分析，你可以选择感兴趣的问题进行深入研究！")

	if "plan" in st.session_state:
		st.subheader("🤖 AI 智能判定结果")
		
		plan = st.session_state["plan"]
		
		# 可读化显示判定结果
		def display_readable_plan(plan):
			# 任务类型
			task_type = plan.get("task_type", "未知")
			task_type_cn = {"classification": "分类任务", "regression": "回归任务", "clustering": "聚类任务"}.get(task_type, task_type)
			
			st.markdown(f"### 📊 **任务类型**: {task_type_cn}")
			if task_type == "classification":
				st.info("🎯 这是一个分类任务，目标是预测离散的类别标签")
			elif task_type == "regression":
				st.info("📈 这是一个回归任务，目标是预测连续的数值")
			elif task_type == "clustering":
				st.info("🔍 这是一个聚类任务，目标是发现数据中的隐藏模式")
			
			# 目标候选列
			targets = plan.get("target_candidates", [])
			if targets:
				st.markdown("### 🎯 **推荐目标列**")
				for i, target in enumerate(targets):
					st.markdown(f"**{i+1}.** `{target}`")
			else:
				st.warning("⚠️ 未找到明确的目标列，请手动选择")
			
			# 推荐算法
			algorithms = plan.get("algorithms", [])
			if algorithms:
				st.markdown("### 🤖 **推荐算法**")
				algo_names = {
					"xgboost": "XGBoost (极端梯度提升)",
					"ridge": "Ridge 回归 (岭回归)",
					"knn": "K-近邻算法",
					"random_forest": "随机森林",
					"linear_regression": "线性回归",
					"logistic_regression": "逻辑回归",
					"svm": "支持向量机",
					"mlp": "多层感知机"
				}
				
				cols = st.columns(min(len(algorithms), 3))
				for i, algo in enumerate(algorithms):
					with cols[i % 3]:
						algo_display = algo_names.get(algo, algo.replace('_', ' ').title())
						st.markdown(f"**{i+1}.** {algo_display}")
			
			# 评估指标
			metrics = plan.get("metrics", [])
			if metrics:
				st.markdown("### 📏 **评估指标**")
				metric_names = {
					"rmse": "RMSE (均方根误差)",
					"mae": "MAE (平均绝对误差)", 
					"r2": "R² (决定系数)",
					"accuracy": "准确率",
					"f1": "F1 分数",
					"precision": "精确率",
					"recall": "召回率",
					"auc": "AUC (曲线下面积)"
				}
				
				metric_cols = st.columns(min(len(metrics), 3))
				for i, metric in enumerate(metrics):
					with metric_cols[i % 3]:
						metric_display = metric_names.get(metric, metric.upper())
						st.markdown(f"**•** {metric_display}")
			
			# 交叉验证设置
			cv_info = plan.get("cv", {})
			if cv_info:
				st.markdown("### ✅ **交叉验证设置**")
				folds = cv_info.get("folds", 5)
				stratified = cv_info.get("stratified", False)
				
				col1, col2 = st.columns(2)
				with col1:
					st.metric("交叉验证折数", f"{folds} 折")
				with col2:
					stratify_text = "是" if stratified else "否"
					st.metric("分层采样", stratify_text)
			
			# 类别不平衡信息
			imbalance = plan.get("imbalance", {})
			if imbalance and imbalance.get("is_imbalanced"):
				st.markdown("### ⚖️ **数据不平衡警告**")
				ratio = imbalance.get("ratio")
				if ratio:
					st.warning(f"检测到数据不平衡，主要类别占比: {ratio:.1%}")
					st.caption("建议考虑使用类别权重平衡或采样技术")
		
		# 显示可读化结果
		display_readable_plan(plan)
		
		# 可选显示原始 JSON
		with st.expander("🔍 查看详细 JSON 结果", expanded=False):
			st.json(plan)
		
		st.caption("💡 以上结果将自动应用到训练设置中，你也可以手动调整参数。")

	# ------------------ 训练设置与训练流程（最小接入） ------------------
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from core import cleandata, train as train_core

	st.subheader("🛠️ 训练设置（本地）")
	
	# 智能应用 AI 判定结果
	plan = st.session_state.get("plan", {})
	ai_targets = plan.get("target_candidates", [])
	ai_task_type = plan.get("task_type", "")
	ai_algorithms = plan.get("algorithms", [])
	ai_cv = plan.get("cv", {})
	
	# 目标列选择 - 优先使用 AI 推荐
	available_columns = [c for c in df.columns if c != ""]
	if ai_targets and ai_targets[0] in available_columns:
		default_target_index = available_columns.index(ai_targets[0])
		st.success(f"🤖 AI 推荐目标列: {ai_targets[0]}")
	else:
		default_target_index = 0
	
	target = st.selectbox("目标列", options=available_columns, index=default_target_index)
	
	# 自动推荐任务类型
	if target:
		target_series = df[target]
		target_series = target_series.dropna()  # 去除缺失值
		
		# 检查是否为数值类型
		is_numeric = pd.api.types.is_numeric_dtype(target_series)
		unique_count = target_series.nunique()
		total_count = len(target_series)
		
		# 推荐逻辑
		if is_numeric and unique_count > 20 and unique_count / total_count > 0.05:
			recommended_task = "regression"
			reason = f"数值类型，{unique_count} 个唯一值"
		else:
			recommended_task = "classification" 
			if not is_numeric:
				reason = f"非数值类型，{unique_count} 个类别"
			else:
				reason = f"数值类型但只有 {unique_count} 个唯一值，可能是分类"
		
		st.info(f"🤖 推荐任务类型: **{recommended_task}** ({reason})")
		
		# 显示目标变量的基本统计
		col1, col2 = st.columns(2)
		with col1:
			st.metric("唯一值数量", unique_count)
		with col2:
			st.metric("样本数量", total_count)
		
		if unique_count <= 10:
			st.write("目标变量的值分布:")
			value_counts = target_series.value_counts().head(10)
			st.bar_chart(value_counts)
	
	# 任务类型选择 - 智能应用 AI 推荐
	task_options = ["classification", "regression"]
	if ai_task_type and ai_task_type in task_options:
		default_task_index = task_options.index(ai_task_type)
		st.success(f"🤖 AI 推荐任务类型: {ai_task_type}")
	else:
		default_task_index = 0
	
	task_type = st.selectbox("任务类型", task_options, index=default_task_index)
	
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
			st.success(f"🤖 AI 推荐算法: {', '.join(ai_algos_mapped)}")
			default_algos = ai_algos_mapped
		else:
			default_algos = ["rf","xgb"]
	else:
		default_algos = ["rf","xgb"]
	
	picked = st.multiselect(
		"候选算法",
		available_algos,
		default=default_algos
	)
	budget = st.slider("每模型搜索次数 (n_iter)", 10, 80, 30)
	
	# 交叉验证折数 - 智能应用 AI 推荐
	default_folds = ai_cv.get("folds", 5) if ai_cv else 5
	if ai_cv and "folds" in ai_cv:
		st.success(f"🤖 AI 推荐 CV 折数: {default_folds}")
	
	folds = st.slider("CV 折数", 3, 10, default_folds)


	# 评估行数限制设置
	with st.expander("⚙️ 评估数据量设置", expanded=False):
		col_a, col_b = st.columns([1,2])
		with col_a:
			use_eval_limit = st.checkbox("限制评估行数", value=True, help="仅在评估指标/预测时使用测试集前 N 行，适合快速迭代。")
		with col_b:
			if use_eval_limit:
				custom_eval_rows = st.number_input("评估最大行数 N", min_value=50, max_value=20000, value=500, step=50, help="超过该行数时仅截取前 N 行；不影响模型训练。")
			else:
				custom_eval_rows = None

	if st.button("开始训练"):
		X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(df, target, task_type)
		leaderboard, artifacts = train_core.run_all(
			X_train, y_train, X_test, y_test,
			task_type=task_type,
			picked_models=picked,
			preprocessor=pre,
			n_iter=budget,
			cv_folds=folds,
			artifacts_dir="artifacts"
		)
		st.success("训练完成！")
		st.dataframe(leaderboard)
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
			st.info(f"评估行数限制启用：使用测试集前 {n_eval} 行（原始 {len(X_test)} 行）。")
		else:
			X_test_eval = X_test
			y_test_eval = y_test
			st.info(f"评估行数限制未启用，使用全部测试集 {len(X_test)} 行。")
		st.session_state["__eval_pack__"] = (task_type, X_test_eval, y_test_eval, artifacts)
