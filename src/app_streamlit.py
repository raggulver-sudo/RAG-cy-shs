import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config
from src.questions_processing import QuestionsProcessor
import json
import base64
import os

# 你可以让 root_path 固定，也可以让用户输入
root_path = Path("data/stock_data")

@st.cache_resource
def get_pipeline():
    return Pipeline(root_path, run_config=max_config)

pipeline = get_pipeline()

# PDF报告目录
PDF_REPORTS_DIR = Path("data/stock_data/1_pdf_reports")
# Chunked报告目录
CHUNKED_REPORTS_DIR = Path("data/stock_data/databases/chunked_reports")


def convert_page_numbers_in_text(text: str) -> str:
    """将文本中的页码从0-based转换为1-based"""
    import re
    # 匹配"页码 X"或"第X页"或"page X"等格式的页码
    # 使用正则表达式匹配数字，然后加1
    def replace_page_number(match):
        page_num = int(match.group(1))
        return match.group(0).replace(str(page_num), str(page_num + 1))
    
    # 匹配"页码 数字"格式
    text = re.sub(r'页码\s*(\d+)', replace_page_number, text)
    # 匹配"第数字页"格式
    text = re.sub(r'第(\d+)页', replace_page_number, text)
    # 匹配"page 数字"格式（不区分大小写）
    text = re.sub(r'page\s*(\d+)', replace_page_number, text, flags=re.IGNORECASE)
    # 匹配"Page 数字"格式
    text = re.sub(r'Page\s*(\d+)', replace_page_number, text)
    
    return text


def find_pdf_path(file_name: str) -> Path:
    """根据文件名查找对应的PDF文件路径"""
    # 移除.json后缀，替换为.pdf
    if file_name.endswith('.json'):
        file_name = file_name[:-5] + '.pdf'
    
    # 移除文件名中的多余空格
    import re
    file_name = re.sub(r'\s+', '', file_name)
    
    pdf_files = list(PDF_REPORTS_DIR.glob("*.pdf"))
    
    # 首先尝试精确匹配
    for pdf_file in pdf_files:
        if pdf_file.name == file_name:
            return pdf_file
    
    # 如果精确匹配失败，尝试匹配stem（不含扩展名的部分）
    file_stem = Path(file_name).stem
    for pdf_file in pdf_files:
        pdf_stem = pdf_file.stem
        # 移除PDF文件名中的空格进行匹配
        pdf_stem_clean = re.sub(r'\s+', '', pdf_stem)
        if pdf_stem_clean == file_stem:
            return pdf_file
    
    return None


def get_chunked_json_path(file_name: str) -> Path:
    """根据文件名查找对应的chunked JSON文件路径"""
    # 移除文件名中的多余空格
    import re
    file_name = re.sub(r'\s+', '', file_name)
    
    json_files = list(CHUNKED_REPORTS_DIR.glob("*.json"))
    
    # 首先尝试精确匹配
    for json_file in json_files:
        if json_file.name == file_name:
            return json_file
    
    # 如果精确匹配失败，尝试匹配stem（不含扩展名的部分）
    file_stem = Path(file_name).stem
    for json_file in json_files:
        json_stem = json_file.stem
        # 移除JSON文件名中的空格进行匹配
        json_stem_clean = re.sub(r'\s+', '', json_stem)
        if json_stem_clean == file_stem:
            return json_file
    
    return None


# 全局缓存，用于存储已读取的JSON内容
@st.cache_resource
def get_json_cache():
    return {}

def get_chunked_json_content(file_name: str) -> dict:
    """获取chunked JSON文件内容，使用缓存"""
    cache = get_json_cache()
    if file_name in cache:
        return cache[file_name]
    
    json_path = get_chunked_json_path(file_name)
    if json_path:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cache[file_name] = data
            return data
        except Exception as e:
            print(f"[ERROR] 读取JSON文件失败 {json_path}: {e}")
            return None
    return None

def get_bboxes_for_page(json_path: Path, page_index: int, retrieval_results: list = None) -> list:
    """从chunked JSON文件中获取指定页面的bbox信息，返回该页面所有相关的bbox（chunk级别）"""
    try:
        # 使用文件名而不是路径来利用缓存
        file_name = json_path.name
        data = get_chunked_json_content(file_name)
        
        if not data:
            return []
        
        chunks = data.get("content", {}).get("chunks", [])
        
        # 收集所有bbox（page_index现在是bbox级别的page）
        all_bboxes = []
        
        # 如果有检索结果，我们优先找到匹配的chunk
        target_chunks = []
        if retrieval_results and isinstance(retrieval_results, list):
            relevant_results = [r for r in retrieval_results if r.get('page') == page_index]
            if relevant_results:
                # 找到这些结果对应的chunk
                for res in relevant_results:
                    res_text = res.get('text', '')
                    for chunk in chunks:
                        if chunk.get("page") == page_index:
                            chunk_text = chunk.get("text", "")
                            # 简单匹配
                            if res_text in chunk_text or chunk_text in res_text:
                                target_chunks.append(chunk)
        
        # 如果找到了目标chunk，只返回这些chunk的bbox
        if target_chunks:
            for chunk in target_chunks:
                chunk_bboxes = chunk.get("bboxes", [])
                for bbox_info in chunk_bboxes:
                    # 确保是当前页
                    if bbox_info.get("page", -1) == page_index:
                        all_bboxes.append(bbox_info)
            
            # 如果找到了bbox，直接返回（不再过滤，显示chunk的所有bbox以保证覆盖内容）
            if all_bboxes:
                print(f"[DEBUG] page={page_index}, 找到 {len(all_bboxes)} 个bbox (基于检索匹配的chunk)")
                return all_bboxes

        # 如果没有检索结果匹配，或者匹配的chunk没有bbox，则回退到返回该页所有bbox
        # (原有逻辑保留作为保底)
        for chunk in chunks:
            chunk_bboxes = chunk.get("bboxes", [])
            for bbox_info in chunk_bboxes:
                if bbox_info.get("page", -1) == page_index:
                    all_bboxes.append(bbox_info)
        
        print(f"[DEBUG] page={page_index}, 返回该页所有 {len(all_bboxes)} 个bbox")
        return all_bboxes

    except Exception as e:
        print(f"读取bbox信息时出错: {e}")
        return []


def display_pdf_with_highlights(pdf_path: Path, page_index: int, bboxes: list):
    """显示PDF页面并高亮指定区域"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        
        # 转换为0-based索引
        page_num = page_index - 1
        
        if page_num < 0 or page_num >= len(doc):
            st.error(f"页码 {page_index} 超出范围")
            doc.close()
            return
        
        page = doc[page_num]
        
        # 获取PDF页面的实际尺寸
        page_width = page.rect.width
        page_height = page.rect.height
        
        # 使用基于mineru页面尺寸的精确缩放因子
        # mineru基于页面尺寸: 961 x 996
        # 缩放因子: scale_x = PDF宽度 / 961, scale_y = PDF高度 / 996
        scale_x = page_width / 961
        scale_y = page_height / 996
        
        # 创建高亮标注，将bbox坐标缩放到PDF的实际尺寸
        for bbox_info in bboxes:
            bbox = bbox_info.get("bbox", [])
            if len(bbox) == 4:
                # 将bbox坐标缩放到PDF的实际尺寸
                scaled_bbox = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ]
                # 添加高亮标注（黄色半透明）
                highlight = page.add_highlight_annot(scaled_bbox)
                highlight.set_colors(stroke=(1, 1, 0))
                highlight.update()
        
        # 将页面渲染为图片，不使用缩放
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        
        # 使用列布局控制显示宽度
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image(img_data, caption=f"第 {page_index} 页", use_container_width=True)
        
        doc.close()
    except ImportError:
        st.error("需要安装PyMuPDF库来显示PDF高亮。请运行: pip install PyMuPDF")
    except Exception as e:
        st.error(f"显示PDF时出错: {e}")


def show_pdf_viewer(ref: dict):
    """显示PDF查看器，精确到bbox级别"""
    file_name = ref.get("file_name", "")
    page_index = ref.get("page_index", 1)
    bbox_coords = ref.get("bbox_coords", "")
    
    # 将页码从0-based转换为1-based进行显示
    if isinstance(page_index, int):
        page_display = page_index + 1
    else:
        page_display = page_index
    
    # 查找PDF文件路径
    pdf_path = find_pdf_path(file_name)
    
    # 添加调试信息（精确到bbox级别），使用折叠状态
    with st.expander("调试信息（bbox级别）", expanded=False):
        st.write(f"- 原始文件名: {file_name}")
        st.write(f"- 原始页码(bbox级别, 0-based): {page_index}")
        st.write(f"- 显示页码(bbox级别, 1-based): {page_display}")
        st.write(f"- Bbox坐标: {bbox_coords}")
        st.write(f"- PDF路径: {pdf_path}")
        
        # 从bbox_coords解析坐标信息
        if bbox_coords:
            try:
                coords = bbox_coords.split(',')
                if len(coords) == 4:
                    x1, y1, x2, y2 = map(float, coords)
                    st.write(f"- 坐标详情: 左上角({x1:.2f}, {y1:.2f}), 右下角({x2:.2f}, {y2:.2f})")
                    st.write(f"- 区域尺寸: 宽度 {x2-x1:.2f}, 高度 {y2-y1:.2f}")
            except ValueError:
                st.write(f"- 坐标解析失败: {bbox_coords}")
    
    if not pdf_path:
        st.error(f"未找到PDF文件: {file_name}")
        return
    
    # 查找chunked JSON文件路径
    json_path = get_chunked_json_path(file_name)
    
    # 获取高亮区域
    bboxes_to_highlight = []
    
    # 策略调整：用户要求精确追踪到最契合的bbox，因此优先使用ref中的bbox_coords
    # 只有当ref中没有bbox信息时，才尝试基于chunk匹配
    
    if bbox_coords:
        # 解析bbox坐标字符串 "x1,y1,x2,y2"
        try:
            coords = bbox_coords.split(',')
            if len(coords) == 4:
                x1, y1, x2, y2 = map(float, coords)
                bboxes_to_highlight = [{
                    "page": page_index,
                    "bbox": [x1, y1, x2, y2],
                    "type": "highlight",
                    "text": ""  # 暂时为空，不影响高亮
                }]
                st.toast("精确锁定最契合Bbox", icon="🎯")
        except ValueError:
            pass
            
    # 如果没有找到精确的bbox，尝试从chunk中获取（保底策略）
    if not bboxes_to_highlight and json_path:
        # 获取检索结果上下文，用于更精确的bbox匹配
        retrieval_results = st.session_state.get("retrieval_results", [])
        bboxes_to_highlight = get_bboxes_for_page(json_path, page_index, retrieval_results)
        
        if bboxes_to_highlight:
            st.toast(f"已加载 {len(bboxes_to_highlight)} 个高亮区域 (基于Chunk匹配)", icon="🔍")
    
    # 显示PDF和高亮，使用1-based页码，只高亮该来源的bbox
    st.subheader(f"PDF查看器 - 第{page_display}页（bbox级别）")
    display_pdf_with_highlights(pdf_path, page_display, bboxes_to_highlight)


st.set_page_config(page_title="知识库", layout="wide")

# 初始化 session state
if "answer_data" not in st.session_state:
    st.session_state.answer_data = None
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = None
if "current_pdf_viewer_idx" not in st.session_state:
    st.session_state.current_pdf_viewer_idx = None
if "active_source_btn_idx" not in st.session_state:
    st.session_state.active_source_btn_idx = None
if "font_size" not in st.session_state:
    st.session_state.font_size = 14

# 页面标题
st.markdown("""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>🚀 RAG 知识库</h2>
    <div style='color: #fff; font-size: 16px;'>minerU+faiss+qwen | 支持多公司年报问答 | 向量检索+LLM推理 | 页码追踪</div>
</div>
""", unsafe_allow_html=True)

# 自定义CSS样式
st.markdown("""
<style>
/* 全局字体设置 */
html, body, [class*="css"] {
    font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

/* 来源按钮样式 - 匹配type="primary"的按钮 */
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] {
    background-color: #fff3cd !important;
    color: #000 !important;
    border: none !important;
    transition: all 0.3s ease;
    padding: 10px !important;
    border-radius: 5px !important;
    font-weight: normal !important;
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover {
    background-color: rgba(0, 123, 255, 0.8) !important;
    color: #fff !important;
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:active {
    background-color: #007bff !important;
    color: #fff !important;
}

/* 优化文本输入框 */
.stTextArea textarea {
    border-radius: 8px !important;
    border: 1px solid #ddd !important;
    padding: 10px !important;
    font-size: 16px !important;
    transition: border-color 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #7b2ff2 !important;
    box-shadow: 0 0 5px rgba(123, 47, 242, 0.3) !important;
}

/* 优化侧边栏样式 */
section[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #eee;
}

/* 标题样式优化 */
h1, h2, h3 {
    color: #2c3e50;
    font-weight: 600;
}

/* 移除调试信息上面的横线 */
div[data-testid="stExpander"] > div > div > div > div {
    border-top: none !important;
}

/* 移除expander的边框 */
div[data-testid="stExpander"] {
    border: none !important;
    background-color: transparent !important;
}

/* 移除expander内部的分隔线 */
div[data-testid="stExpander"] > div > div {
    border: none !important;
}

/* 生成答案按钮样式增强 */
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%);
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(123, 47, 242, 0.4);
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:active {
    transform: translateY(0);
}
</style>
""", unsafe_allow_html=True)

# 左侧输入区
with st.sidebar:
    st.header("查询设置")
    
    # 预设问题
    st.markdown("**预设问题：**")
    preset_questions = [
        "中芯国际2024的营业收入是多少",
        "中芯国际2024的归属于上市公司股东的净利润是多少"
    ]
    
    def on_preset_change():
        if st.session_state.selected_preset and st.session_state.selected_preset != "请选择预设问题...":
            # 只需要更新 text_area 的 key 对应的值，Streamlit 会自动同步
            st.session_state.user_question_area = st.session_state.selected_preset

    selected_preset = st.selectbox(
        "选择预设问题", 
        ["请选择预设问题..."] + preset_questions, 
        label_visibility="collapsed",
        key="selected_preset",
        on_change=on_preset_change
    )
    
    # 增加间距
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 初始化问题内容（如果还未设置）
    if "user_question_area" not in st.session_state:
        st.session_state.user_question_area = "请简要总结公司2022年主营业务的主要内容。"
    
    # 仅单问题输入
    # 注意：当设置了 key 时，不要使用 value 参数，直接通过 session_state 初始化或更新
    user_question = st.text_area(
        "输入问题", 
        height=80,
        key="user_question_area"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    submit_btn = st.button("生成答案", use_container_width=True)
    
    # 移除了 A/B Testing & Optimization Section UI
    # 移除了界面设置区域

# 右侧主内容区
st.markdown("<h3 style='margin-top: 24px;'>检索结果</h3>", unsafe_allow_html=True)

if submit_btn and user_question.strip():
    # Update Pipeline Configuration dynamically
    # 默认配置（界面控件已移除）
    pipeline.processor.llm_reranking = False 
    pipeline.processor.enable_data_validation = True
    
    import time
    start_time = time.time()
    
    with st.spinner("正在生成答案，请稍候..."):
        try:
            answer = pipeline.answer_single_question(user_question, kind="string")
            # 兼容 answer 可能为 str 或 dict
            if isinstance(answer, str):
                try:
                    answer_dict = json.loads(answer)
                except Exception:
                    st.error("返回内容无法解析为结构化答案：" + str(answer))
                    answer_dict = {}
            else:
                answer_dict = answer
            # 直接从 answer_dict 获取内容
            content = answer_dict
                    
            step_by_step = content.get("step_by_step_analysis", "-")
            reasoning_summary = content.get("reasoning_summary", "-")
            relevant_pages = content.get("relevant_pages", [])
            references = content.get("references", [])
            final_answer = content.get("final_answer", "-")
            retrieval_results = content.get("retrieval_results", [])
            
            # 保存到 session state
            st.session_state.answer_data = {
                "step_by_step": step_by_step,
                "reasoning_summary": reasoning_summary,
                "relevant_pages": relevant_pages,
                "references": references,
                "final_answer": final_answer
            }
            # 保存检索结果，用于PDF高亮时找到最相关的chunk
            st.session_state.retrieval_results = retrieval_results
            
            # 打印调试
            print("[DEBUG] step_by_step_analysis:", step_by_step)
            print("[DEBUG] reasoning_summary:", reasoning_summary)
            print("[DEBUG] relevant_pages:", relevant_pages, "type:", type(relevant_pages))
            print("[DEBUG] references:", references)
            print("[DEBUG] final_answer:", final_answer)
        except Exception as e:
            st.error(f"生成答案时出错: {e}")
            st.session_state.answer_data = None

# 如果有答案数据，显示出来
if st.session_state.answer_data:
    step_by_step = st.session_state.answer_data["step_by_step"]
    reasoning_summary = st.session_state.answer_data["reasoning_summary"]
    relevant_pages = st.session_state.answer_data["relevant_pages"]
    references = st.session_state.answer_data["references"]
    final_answer = st.session_state.answer_data["final_answer"]
    
    # 确保 relevant_pages 是列表
    if not isinstance(relevant_pages, list):
        print(f"[DEBUG] relevant_pages 不是列表，尝试转换: {relevant_pages}")
        if isinstance(relevant_pages, str):
            try:
                relevant_pages = json.loads(relevant_pages)
            except:
                relevant_pages = []
        elif isinstance(relevant_pages, dict):
            relevant_pages = list(relevant_pages.values()) if relevant_pages else []
        else:
            relevant_pages = []
    
    st.markdown("**分步推理：**")
    # 将分步推理中的页码从0-based转换为1-based
    step_by_step_converted = convert_page_numbers_in_text(step_by_step)
    with st.expander("查看分步推理详情", expanded=False):
        st.info(step_by_step_converted)
    st.markdown("**推理摘要：**")
    # 将推理摘要中的页码从0-based转换为1-based
    reasoning_summary_converted = convert_page_numbers_in_text(reasoning_summary)
    st.success(reasoning_summary_converted)
    st.markdown("**最终答案：**")
    st.success(final_answer)
    st.markdown("**相关页面：** ")
    if relevant_pages:
        # 使用 references 列表展示每个页码对应的公司名称和文档名称
        if references and isinstance(references, list):
            # 根据bbox的page从小到大排列
            references_sorted = sorted(references, key=lambda x: x.get("page_index", float('inf')))
            for idx, ref in enumerate(references_sorted):
                page = ref.get("page_index", "N/A")
                company = ref.get("company_name", "未知公司")
                file = ref.get("file_name", "未知文件")
                bbox_coords = ref.get("bbox_coords", "")
                
                # 将页码从0-based转换为1-based进行显示
                if isinstance(page, int):
                    page_display = page + 1
                else:
                    page_display = page
                
                # 将.json替换为.pdf进行显示
                if file.endswith('.json'):
                    file_display = file[:-5] + '.pdf'
                else:
                    file_display = file
                
                # 判断当前按钮是否被激活
                is_active = st.session_state.active_source_btn_idx == idx
                
                # 创建橙黄色背景的列表项样式，使用 columns 布局
                current_font_size = st.session_state.get("font_size", 14)
                
                # 使用 columns 实现同一行布局
                col_content, col_button = st.columns([8, 2])
                
                with col_content:
                    # 显示内容
                    bbox_info = ""
                    if bbox_coords:
                        try:
                            coords = bbox_coords.split(',')
                            if len(coords) == 4:
                                x1, y1, x2, y2 = map(float, coords)
                                bbox_info = f"【bbox坐标:({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) 区域尺寸:{x2-x1:.0f}×{y2-y1:.0f}】"
                            else:
                                bbox_info = f"【bbox:{bbox_coords}】"
                        except ValueError:
                            bbox_info = f"【bbox:{bbox_coords}】"
                    
                    st.markdown(f"""
                    <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: {current_font_size}px;'>
                        {company} - {file_display} 【页码：{page_display}】
                        {bbox_info}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_button:
                    # 添加【来源】按钮
                    if st.button(f"来源", key=f"source_btn_{idx}", use_container_width=True, type="primary"):
                        # 如果当前按钮已经被激活，则关闭PDF查看器
                        if st.session_state.active_source_btn_idx == idx:
                            st.session_state.active_source_btn_idx = None
                            st.session_state.current_pdf_viewer_idx = None
                            st.session_state[f"show_pdf_{idx}"] = False
                        else:
                            # 关闭之前打开的PDF查看器
                            if st.session_state.current_pdf_viewer_idx is not None:
                                st.session_state[f"show_pdf_{st.session_state.current_pdf_viewer_idx}"] = False
                            # 打开当前PDF查看器
                            st.session_state[f"show_pdf_{idx}"] = True
                            st.session_state[f"current_ref_{idx}"] = ref
                            st.session_state.current_pdf_viewer_idx = idx
                            st.session_state.active_source_btn_idx = idx
                        st.rerun()
                
                # 只有当前点击的来源才显示PDF查看器
                if st.session_state.get(f"show_pdf_{idx}", False) and st.session_state.current_pdf_viewer_idx == idx:
                    st.markdown("---")
                    show_pdf_viewer(ref)
                    
                    # 添加关闭按钮
                    if st.button("关闭PDF查看器", key=f"close_pdf_{idx}"):
                        st.session_state[f"show_pdf_{idx}"] = False
                        st.session_state.current_pdf_viewer_idx = None
                        st.session_state.active_source_btn_idx = None
                        st.rerun()
        else:
            # 如果没有 references 信息，则只显示页码
            pages_str = ', '.join(map(str, relevant_pages))
            st.write(pages_str)
    else:
        st.write("无相关页面")
else:
    st.info("请在左侧输入问题并点击【生成答案】") 