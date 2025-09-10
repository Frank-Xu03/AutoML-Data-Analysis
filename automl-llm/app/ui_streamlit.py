import streamlit as st
import os
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
