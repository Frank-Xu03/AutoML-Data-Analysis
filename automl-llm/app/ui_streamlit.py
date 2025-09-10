

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

# 数据上传
uploaded_file = st.file_uploader("上传数据文件（CSV）", type=["csv"])

if uploaded_file is not None:
	df = pd.read_csv(uploaded_file)
	# 保存到 examples 目录
	save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples', uploaded_file.name)
	# 自动创建目录
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	with open(save_path, "wb") as f:
		f.write(uploaded_file.getbuffer())
	st.success(f"文件已保存到 examples/{uploaded_file.name}")
	st.write("数据预览：")
	st.dataframe(df.head())
	st.write("数据描述：")
	st.write(df.describe())
	st.write("缺失值统计：")
	st.write(df.isnull().sum())

	# ----------- 判定按钮与结果展示区块 -------------
	# 这里假设 prof 是数据 profile，实际应由 ingest/profile 生成
	# 你可以用 df.describe().to_dict() 或自定义 profile
	prof = {
		"columns": [
			{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isnull().sum()), "unique": int(df[c].nunique())}
			for c in df.columns
		]
	}

	user_question = st.text_area("你的问题（可选）", placeholder="例如：我们能否预测乘客是否生还？或 预测价格/分群等。")

	col1, col2 = st.columns([1,1])
	with col1:
		if st.button("智能判定（OpenAI）"):
			with st.spinner("调用 OpenAI 判定任务类型与方案..."):
				plan = detect_task(prof, user_question or "")
			st.session_state["plan"] = plan
			st.success("判定完成")
	with col2:
		if "plan" in st.session_state:
			st.download_button("下载判定 JSON", data=json.dumps(st.session_state["plan"], ensure_ascii=False, indent=2),
							   file_name="task_plan.json", mime="application/json")

	if "plan" in st.session_state:
		st.subheader("判定结果")
		st.json(st.session_state["plan"])
		st.caption("上面结果将指导后续：目标列选择、候选算法、评估指标与交叉验证设置。")
