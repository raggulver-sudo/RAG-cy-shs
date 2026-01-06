# Bug历史解决方案

## 1. Chunking方式修改

### 问题描述
原系统使用.md文件进行chunking，无法包含表格和图片内容。

### 解决方案
修改`text_splitter.py`中的`split_content_list_file`方法，改为使用`_content_list.json`文件进行chunking：
- 直接从JSON文件中读取内容列表
- 包含表格和图片内容
- 保留bbox坐标信息

### 相关文件
- `src/text_splitter.py`
- `src/pipeline.py`

---

## 2. Bbox信息保留

### 问题描述
Chunked文件中缺少bbox信息，无法进行PDF内容高亮显示。

### 解决方案
在chunking过程中收集bbox信息：
- 遍历content_list中的每个元素
- 提取bbox坐标（x1, y1, x2, y2）
- 将bbox信息添加到chunk的metadata中

### 相关文件
- `src/text_splitter.py`

---

## 3. Web界面添加"来源"按钮

### 问题描述
需要在web界面的引用文档旁边添加"来源"按钮，点击后打开对应的PDF文档并高亮显示相关内容。

### 解决方案
在`app_streamlit.py`中实现：
1. 添加PDF路径映射函数`find_pdf_path`
2. 实现bbox提取函数`get_bboxes_for_page`
3. 创建PDF显示函数`display_pdf_with_highlights`
4. 在引用区域添加"来源"按钮
5. 使用PyMuPDF库进行PDF渲染和高亮

### 相关文件
- `app_streamlit.py`

---

## 4. PDF路径映射

### 问题描述
文件名存在额外空格和扩展名不匹配（.json vs .pdf），导致无法找到对应的PDF文件。

### 解决方案
修改`find_pdf_path`函数：
- 规范化文件名（移除多余空格）
- 将.json扩展名替换为.pdf
- 在正确的目录中搜索PDF文件

### 相关文件
- `app_streamlit.py`

---

## 5. 文件名显示问题

### 问题描述
Web界面显示的文件名是.json扩展名，用户希望显示原始的.pdf文件名。

### 解决方案
修改显示逻辑：
- 将文件名从.json替换为.pdf
- 在相关页面显示原始PDF文件名

### 相关文件
- `app_streamlit.py`

---

## 6. PDF显示大小问题

### 问题描述
PDF显示区域过大，右侧内容被截断。

### 解决方案
调整PDF渲染参数：
- 减少PyMuPDF缩放因子（从2.0改为1.5）
- 实现三列布局（1:6:1）来居中显示PDF
- 使用`use_container_width=True`自适应宽度

### 相关文件
- `app_streamlit.py`

---

## 7. 页码转换问题（0-based到1-based）

### 问题描述
JSON文件使用0-based页码，但web界面应该显示1-based页码。

### 解决方案
实现页码转换函数：
- 在显示时将页码加1
- 在所有显示位置应用转换（分步推理、推理摘要、相关页面）

### 相关文件
- `app_streamlit.py`
- `src/questions_processing.py`

---

## 8. PDF查看器状态管理

### 问题描述
1. 未点击"来源"按钮时PDF查看器就打开了
2. 点击新来源时之前的查看器没有关闭

### 解决方案
使用Streamlit的session state管理：
- 添加`current_pdf_viewer_idx`状态变量
- 只在点击"来源"按钮时设置查看器索引
- 点击新来源时自动关闭之前的查看器

### 相关文件
- `app_streamlit.py`

---

## 9. 摘要页码验证问题

### 问题描述
摘要中提到多个页码，但相关页面只列出了一个页码。

### 解决方案
修改`_validate_page_references`方法：
- 保留LLM引用的所有页码
- 不再强制要求所有页码都有对应的chunk

### 相关文件
- `src/questions_processing.py`

---

## 10. Bbox选择算法问题

### 问题描述
初始的bbox选择算法不够精确，无法准确匹配查询结果相关的bbox。

### 解决方案
实现改进的文本匹配算法：
- 使用LCS（最长公共子序列）算法进行文本匹配
- 实现文本包含度评分
- 选择与查询文本最相关的bbox

### 相关文件
- `app_streamlit.py`

---

## 11. 性能优化问题

### 问题描述
点击"来源"按钮后处理时间长，笔记本电脑风扇噪音大。

### 解决方案
优化bbox匹配算法：
- 替换LCS为简单的字符串匹配
- 限制文本处理长度（前500字符）
- 实现文本规范化（去除空格和特殊字符）

### 相关文件
- `app_streamlit.py`

---

## 12. Bbox级别页面追踪

### 问题描述
系统使用chunk级别的页码，不够精确，需要追踪到bbox级别的页码。

### 解决方案
实现bbox级别页面追踪：
- 添加`_find_most_relevant_bbox_page`方法
- 从chunk的page进一步精确到最相关bbox的page
- 更新所有使用页码的地方（分步推理、推理摘要、相关页面）

### 相关文件
- `src/questions_processing.py`

---

## 13. Bbox坐标显示

### 问题描述
Web界面的相关页面没有显示bbox坐标信息。

### 解决方案
修改`_extract_references`方法：
- 在reference中添加bbox_coords字段
- 格式：`x1,y1,x2,y2`
- 在web界面显示bbox坐标（格式：`【页码：X】【bbox:x1,y1,x2,y2】`）

### 相关文件
- `src/questions_processing.py`
- `app_streamlit.py`

---

## 14. 调试信息精确度

### 问题描述
调试信息显示chunk page里的所有bbox，而不是只显示该来源对应的bbox。

### 解决方案
修改`show_pdf_viewer`函数：
- 只显示该来源对应的bbox信息
- 从reference数据中提取目标bbox
- 过滤掉不相关的bbox

### 相关文件
- `app_streamlit.py`

---

## 15. 相关页面排序

### 问题描述
相关页面列表没有按页码排序。

### 解决方案
在显示相关页面时添加排序逻辑：
- 根据bbox的page从小到大排列
- 使用sorted函数对references进行排序

### 相关文件
- `app_streamlit.py`

---

## 16. PDF高亮精确度

### 问题描述
PDF查看器高亮了多个bbox，而不是只高亮该来源的bbox。

### 解决方案
修改高亮逻辑：
- 只传递目标bbox到高亮函数
- 过滤掉不相关的bbox
- 确保只高亮一个bbox区域

### 相关文件
- `app_streamlit.py`

---

## 17. 坐标系统不匹配

### 问题描述
Bbox坐标和PDF高亮显示区域不一致。

### 解决方案
分析并实现正确的坐标转换：
- 发现mineru基于页面尺寸961 x 996生成bbox坐标
- 使用动态缩放因子：`scale_x = PDF宽度 / 961`, `scale_y = PDF高度 / 996`
- 将bbox坐标缩放到PDF的实际尺寸

### 相关文件
- `app_streamlit.py`

---

## 18. 坐标原点问题

### 问题描述
怀疑坐标原点不同导致高亮区域不准确。

### 解决方案
通过测试确认：
- mineru使用左上角为原点（与PyMuPDF相同）
- 不需要翻转y坐标
- 不需要额外的偏移量修正
- 只需要缩放转换

### 相关文件
- `app_streamlit.py`

---

## 19. 性能优化：关闭无效重排序

### 问题描述
项目运行特别慢，单次查询耗时超过300秒。经分析发现主要瓶颈在于 LLM Reranking：
1. `max_config` 中开启了重排序且样本量为30。
2. 针对 DashScope (Qwen) 的重排序实现存在缺陷，返回全0分，导致重排序无效。
3. 且在单问模式下并发请求数被限制为1，导致串行处理耗时极长。

### 解决方案
修改 `src/pipeline.py` 中的 `max_config`：
- 关闭重排序：`llm_reranking=False`
- 增加并发数：`parallel_requests` 从 4 增加到 10
- 更新 `RunConfig` 默认值以匹配优化后的策略

### 相关文件
- `src/pipeline.py`

---

## 20. 性能优化：QuestionsProcessor实例复用与缓存

### 问题描述
单次查询依然耗时较长。分析发现：
1. `answer_single_question`每次调用都重新初始化`QuestionsProcessor`，导致重复执行耗时的`glob`文件扫描操作。
2. `app_streamlit.py`在每次高亮PDF时都重新读取JSON文件，造成大量重复IO。

### 解决方案
1. 修改 `src/pipeline.py`：
   - 在 `Pipeline.__init__` 中初始化 `QuestionsProcessor` 并持久化。
   - `answer_single_question` 复用已存在的 processor 实例。
2. 修改 `src/app_streamlit.py`：
   - 使用 `@st.cache_resource` 实现 `get_json_cache`。
   - 实现 `get_chunked_json_content` 函数，优先从缓存读取JSON内容。

### 相关文件
- `src/pipeline.py`
- `src/app_streamlit.py`

---

## 21. Streamlit 预置问题交互优化

### 问题描述
预置问题选择后，输入框没有自动更新，且没有默认选中的视觉反馈。

### 解决方案
修改 `src/app_streamlit.py`：
- 在 `on_preset_change` 回调中同时更新 `user_question_content` 和 `user_question_area`（session_state key）。
- 确保下拉框选择事件能正确触发输入框的值更新。

### 相关文件
- `src/app_streamlit.py`

---

## 22. PDF 高亮 Bbox 精确度优化（精确追踪）

### 问题描述
Web界面右侧高亮的PDF区域与用户问题不够契合，需要精确追踪到Chunk下最相关的特定Bbox。

### 解决方案
1. 修改 `src/questions_processing.py`：
   - 实现 `simple_match_score` 启发式评分算法，结合关键词、数值类型、位置重叠等多维度特征。
   - 禁用极度耗时的 Bbox 级大模型语义评分。
   - 在 `_format_retrieval_results` 阶段计算并缓存最佳 Bbox。
2. 修改 `src/app_streamlit.py`：
   - `show_pdf_viewer` 优先使用 Reference 中携带的 `bbox_coords`（即最佳 Bbox）。
   - 仅在无精确 Bbox 时回退到显示 Chunk 所有 Bbox。

### 相关文件
- `src/questions_processing.py`
- `src/app_streamlit.py`

---

## 23. 性能优化：QuestionsProcessing Bbox计算优化

### 问题描述
`questions_processing.py` 查询依然耗时，主要原因在于：
1. Bbox 级的大模型语义评分（Semantic Scoring）导致每个 Bbox 都要调用一次 API。
2. Bbox 计算逻辑在不同步骤（检索结果格式化、引用提取、坐标查找）中被重复调用，导致重复计算和IO。

### 解决方案
修改 `src/questions_processing.py`：
1. **禁用大模型评分**：在 `simple_match_score` 中强制禁用语义相关性评分，消除了大量 API 调用开销。
2. **结果缓存**：在 `_format_retrieval_results` 中，将计算出的最佳 `bbox_page`、`bbox_coords` 和 `bbox_score` 直接缓存到 `retrieval_results` 对象中。
3. **缓存优先**：修改 `_find_bbox_coordinates_for_page`，优先从 `retrieval_results` 的缓存中获取 Bbox 信息，避免重复读取 JSON 文件和重新计算。

### 相关文件
- `src/questions_processing.py`

---

## 技术要点总结

### 坐标转换公式
```python
# 获取PDF页面的实际尺寸
page_width = page.rect.width
page_height = page.rect.height

# 使用基于mineru页面尺寸的精确缩放因子
scale_x = page_width / 961
scale_y = page_height / 996

# 将bbox坐标缩放到PDF的实际尺寸
scaled_bbox = [
    bbox[0] * scale_x,
    bbox[1] * scale_y,
    bbox[2] * scale_x,
    bbox[3] * scale_y
]
```

### 页码转换
```python
# 将0-based页码转换为1-based页码
page_display = page_index + 1
```

### 文件名规范化
```python
# 移除多余空格
normalized_name = file_name.replace(' ', '')
# 将.json替换为.pdf
pdf_name = normalized_name.replace('.json', '.pdf')
```

### 性能优化策略
```python
# 1. 实例复用
self.processor = QuestionsProcessor(...) # 在Pipeline初始化时创建

# 2. JSON资源缓存
@st.cache_resource
def get_json_cache():
    return {}

# 3. Bbox计算缓存
if 'bbox_coords' in result:
    return result['bbox_coords']
```

---

## 关键文件说明

### `app_streamlit.py`
Web界面主文件，包含：
- PDF路径映射
- Bbox提取和显示（集成缓存与精确匹配逻辑）
- PDF渲染和高亮
- 页码转换
- 查看器状态管理
- 预置问题交互逻辑

### `src/text_splitter.py`
文本分割器，包含：
- JSON文件chunking
- Bbox信息收集
- 表格和图片内容包含

### `src/pipeline.py`
管道配置，包含：
- Chunking方法选择
- 参数配置
- 性能优化配置（关闭无效Rerank）
- Processor实例复用逻辑

### `src/questions_processing.py`
问题处理，包含：
- Bbox级别页面追踪（集成启发式评分与缓存）
- 引用提取和验证
- 页码转换
- 性能优化（禁用昂贵评分，避免重复计算）

---

## 测试验证

### 功能测试
1. ✅ Chunking包含表格和图片
2. ✅ Bbox信息正确保留
3. ✅ "来源"按钮正常工作
4. ✅ PDF路径正确映射
5. ✅ 文件名正确显示
6. ✅ PDF显示大小合适
7. ✅ 页码正确转换
8. ✅ 查看器状态正确管理
9. ✅ 摘要页码正确验证
10. ✅ Bbox选择精确（基于多维度启发式评分）
11. ✅ 性能优化有效（关闭无效Rerank + JSON缓存 + 实例复用 + 禁用Bbox LLM评分 + 结果缓存）
12. ✅ Bbox级别页面追踪准确
13. ✅ Bbox坐标正确显示
14. ✅ 调试信息精确
15. ✅ 相关页面正确排序
16. ✅ PDF高亮精确（追踪到Chunk下最契合的Bbox）
17. ✅ 坐标转换准确
18. ✅ 坐标原点一致
19. ✅ 性能显著提升（查询耗时大幅降低，重复IO消除，API调用减少）
20. ✅ 预置问题交互顺畅（点击即更新输入框）

---

## 结论

经过系统性的问题分析和解决方案实施，成功实现了：
- 基于JSON的chunking方式，包含表格和图片
- Bbox坐标信息的保留和精确转换
- Web界面的PDF查看和高亮功能
- 精确的bbox级别页面追踪（基于高效的启发式算法）
- 良好的用户体验和性能（查询耗时显著降低，交互更加流畅）
- PDF高亮区域精准匹配用户问题
- 代码结构的优化（缓存机制、实例复用）

所有功能均已测试验证，工作正常。
