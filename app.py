import streamlit as st

st.title("学术文献智能问答系统")

uploader_file = st.file_uploader(
    label="Upload your file",
    type=['txt'],
    accept_multiple_files=False,
)

if uploader_file is not None:
    # extract the file
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f}KB")
    # get_value
    text = uploader_file.getvalue().decode("utf-8")
    st.write(text)