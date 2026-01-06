import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from src.data_validator import DataValidator
from src.performance_monitor import PerformanceMonitor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import time


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,  # 是否启用父文档检索
        llm_reranking: bool = False,              # 是否启用LLM重排
        llm_reranking_sample_size: int = 5,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "dashscope", # openai
        answering_model: str = "qwen-turbo-latest", # gpt-4o-2024-08-06
        full_context: bool = False,
        enable_data_validation: bool = True
    ):
        # 初始化问题处理器，配置检索、模型、并发等参数
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context
        self.enable_data_validation = enable_data_validation

        if self.subset_path:
            DataValidator.set_subset_path(self.subset_path)
        
        # Record startup time (approximate, as this class is initialized after pipeline setup)
        PerformanceMonitor.record_startup_time()

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()
        
        # 缓存JSON文件路径映射
        self.json_file_map = {p.name: p for p in self.documents_dir.glob("*.json")}
        # 缓存已读取的JSON内容
        self.json_content_cache = {}

    def _get_json_content(self, file_name: str) -> Optional[dict]:
        """获取JSON文件内容，使用缓存"""
        if file_name in self.json_content_cache:
            return self.json_content_cache[file_name]
            
        json_path = self.json_file_map.get(file_name)
        if not json_path:
            # 尝试重新扫描（可能是新生成的文件）
            self.json_file_map = {p.name: p for p in self.documents_dir.glob("*.json")}
            json_path = self.json_file_map.get(file_name)
            
        if json_path:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.json_content_cache[file_name] = data
                return data
            except Exception as e:
                print(f"[ERROR] 读取JSON文件失败 {json_path}: {e}")
                return None
        return None

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        # 加载问题文件，返回问题列表
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """将检索结果格式化为RAG上下文字符串，使用bbox级别的page"""
        if not retrieval_results:
            return ""
        
        print(f"\n{'='*80}")
        print(f"[DEBUG] _format_retrieval_results 开始执行")
        print(f"[DEBUG] 输入: retrieval_results数量={len(retrieval_results)}")
        
        context_parts = []
        for idx, result in enumerate(retrieval_results):
            print(f"\n[DEBUG] 处理第{idx+1}个检索结果:")
            print(f"[DEBUG]   file_name={result.get('file_name', '')}")
            print(f"[DEBUG]   chunk_page={result.get('page', 0)}")
            print(f"[DEBUG]   text预览='{result.get('text', '')[:100]}...'")
            
            # 使用bbox级别的page而不是chunk级别的page
            # 检查是否已经计算过（避免重复计算）
            if 'bbox_coords' in result and 'bbox_page' in result:
                bbox_page = result['bbox_page']
                bbox_coords = result['bbox_coords']
                bbox_score = result.get('bbox_score', 0.0)
                print(f"[DEBUG]   使用已缓存的bbox信息")
            else:
                bbox_result = self._find_most_relevant_bbox_page(result)
                bbox_page = bbox_result.get("page", result.get('page', 0))
                bbox_coords = bbox_result.get("bbox_coords", "")
                bbox_score = bbox_result.get("score", 0.0)
                
                # 将计算结果保存回result对象，供后续步骤使用
                result['bbox_page'] = bbox_page
                result['bbox_coords'] = bbox_coords
                result['bbox_score'] = bbox_score
                result['all_bboxes'] = bbox_result.get("all_bboxes", [])
            
            print(f"[DEBUG]   bbox_page={bbox_page} (bbox级别)")
            print(f"[DEBUG]   bbox_coords={bbox_coords}")
            print(f"[DEBUG]   bbox_score={bbox_score:.2f}")
            
            text = result['text']
            context_parts.append(f'Text retrieved from page {bbox_page}: \n"""\n{text}\n"""')
        
        rag_context = "\n\n---\n\n".join(context_parts)
        print(f"\n[DEBUG] _format_retrieval_results 完成")
        print(f"[DEBUG] 输出: RAG上下文字符串长度={len(rag_context)}")
        print(f"{'='*80}\n")
        
        return rag_context

    def _extract_references(self, pages_list: list, company_name: str, retrieval_results: list = None, question: str = None) -> list:
        # 根据公司名和页码列表，提取引用信息，包含bbox坐标
        # pages_list现在可能是包含page和bbox_coords的字典列表，或者是纯页码列表
        print(f"\n{'='*80}")
        print(f"[DEBUG] _extract_references 开始执行")
        print(f"[DEBUG] 输入:")
        print(f"[DEBUG]   pages_list={pages_list}")
        print(f"[DEBUG]   company_name={company_name}")
        print(f"[DEBUG]   retrieval_results数量={len(retrieval_results) if retrieval_results else 0}")
        print(f"[DEBUG]   question={question}")
        print(f"{'='*80}\n")
        
        if self.subset_path is None:
            raise ValueError("subset_path is required for new challenge pipeline when processing references.")
        # 优先尝试 utf-8，失败则尝试 gbk
        try:
            self.companies_df = pd.read_csv(self.subset_path, encoding='utf-8')
        except UnicodeDecodeError:
            print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
            self.companies_df = pd.read_csv(self.subset_path, encoding='gbk')

        # Find the company's SHA1 from the subset CSV
        matching_rows = self.companies_df[self.companies_df['company_name'] == company_name]
        if matching_rows.empty:
            company_sha1 = ""
        else:
            company_sha1 = matching_rows.iloc[0]['sha1']

        # 从检索结果中提取公司名和文件名
        ref_company_name = company_name
        ref_file_name = "未知文件"
        if retrieval_results:
            ref_company_name = retrieval_results[0].get('company_name', company_name)
            ref_file_name = retrieval_results[0].get('file_name', '未知文件')

        refs = []
        for idx, page_item in enumerate(pages_list):
            print(f"\n[DEBUG] 处理第{idx+1}个page_item: {page_item}")
            
            # 检查page_item是字典还是纯页码
            if isinstance(page_item, dict):
                page = page_item.get("page")
                bbox_coords = page_item.get("bbox_coords", "")
                print(f"[DEBUG]   page_item是字典，直接获取: page={page}, bbox_coords={bbox_coords}")
            else:
                page = page_item
                # 为每个page找到对应的bbox坐标
                bbox_coords = self._find_bbox_coordinates_for_page(page, retrieval_results, question)
                print(f"[DEBUG]   page_item是纯页码，调用_find_bbox_coordinates_for_page: page={page}, bbox_coords={bbox_coords}")
            
            ref_item = {
                "pdf_sha1": company_sha1,
                "page_index": page,
                "company_name": ref_company_name,
                "file_name": ref_file_name,
                "bbox_coords": bbox_coords
            }
            refs.append(ref_item)
            print(f"[DEBUG]   添加引用: {ref_item}")
        
        print(f"\n[DEBUG] _extract_references 完成")
        print(f"[DEBUG] 输出: refs数量={len(refs)}")
        for idx, ref in enumerate(refs):
            print(f"[DEBUG]   ref[{idx}]: page={ref['page_index']}, bbox_coords={ref['bbox_coords']}")
        print(f"{'='*80}\n")
        
        return refs

    def _find_bbox_coordinates_for_page(self, page: int, retrieval_results: list, question: str = None) -> str:
        """
        为指定页面找到对应的bbox坐标，返回最相关的bbox
        
        参数:
            page: bbox级别的page（0-based，来自chunked_reports的page）
            retrieval_results: 检索结果列表
            question: 原始查询问题，用于提取年份等信息
        
        返回:
            bbox坐标字符串，格式为"x1,y1,x2,y2"
        """
        # print(f"[DEBUG] _find_bbox_coordinates_for_page 被调用: page={page}")
        
        if not retrieval_results:
            return ""
        
        # 1. 优先从已计算的retrieval_results中查找
        for result in retrieval_results:
            # 检查bbox_page是否匹配（这是最准确的）
            if result.get('bbox_page') == page:
                bbox_coords = result.get('bbox_coords', "")
                if bbox_coords:
                    # print(f"[DEBUG] 从retrieval_results缓存中找到bbox: page={page}, bbox={bbox_coords}")
                    return bbox_coords
        
        # 2. 如果没找到，尝试匹配chunk_page
        for result in retrieval_results:
            result_page = result.get('page', 0)
            if result_page == page:
                # 这是一个chunk page，我们需要计算最相关的bbox
                # 检查是否已经计算过
                if 'bbox_coords' in result:
                    return result['bbox_coords']
                
                # 如果没有缓存，重新计算
                bbox_result = self._find_most_relevant_bbox_page(result, question)
                return bbox_result.get("bbox_coords", "")
                
        return ""

    def _find_most_relevant_bbox_page(self, retrieval_result: dict, question: str = None) -> dict:
        """
        从检索结果中找到最相关的bbox的page和坐标，并返回所有bbox的评分信息
        
        参数:
            retrieval_result: 单个检索结果，包含file_name, page, text等信息
            question: 原始查询问题，用于提取年份等信息
        
        返回:
            包含最相关bbox的page、bbox_coords、score以及所有bbox评分信息的字典
            格式为 {
                "page": int, 
                "bbox_coords": str,
                "score": float,
                "all_bboxes": [{"page": int, "bbox_coords": str, "score": float}, ...]
            }
        """
        file_name = retrieval_result.get('file_name', '')
        chunk_page = retrieval_result.get('page', 0)
        chunk_text = retrieval_result.get('text', '')
        
        print(f"[DEBUG] _find_most_relevant_bbox_page 开始执行: file_name={file_name}, chunk_page={chunk_page}")
        print(f"[DEBUG]   chunk_text长度={len(chunk_text)}, 内容预览='{chunk_text[:100]}...'")
        
        # 查找对应的chunked JSON文件
        data = self._get_json_content(file_name)
        
        if not data:
            print(f"[WARNING] 未找到文件或读取失败: {file_name}")
            return {"page": chunk_page, "bbox_coords": "", "score": 0.0, "all_bboxes": []}
        
        try:
            chunks = data.get("content", {}).get("chunks", [])
            print(f"[DEBUG] JSON文件中找到 {len(chunks)} 个chunks")
            
            # 找到对应的chunk
            target_chunk = None
        
            # 策略1：遍历所有chunk，检查chunk包含的bboxes中是否有目标页码
            # 这是一个更可靠的方法，因为chunks跨越多页
            for chunk in chunks:
                chunk_pages = set()
                if 'bboxes' in chunk:
                    for bbox in chunk['bboxes']:
                        chunk_pages.add(bbox.get('page'))
                
                if chunk_page in chunk_pages:
                    target_chunk = chunk
                    print(f"[DEBUG] 找到匹配的chunk (页码包含): target_page={chunk_page}, chunk_id={chunk.get('id')}, chunk_pages={sorted(list(chunk_pages))}")
                    break

            # 策略2：如果策略1失败，尝试通过页码和文本匹配
            if not target_chunk:
                for chunk in chunks:
                    # 检索结果和chunk数据都使用0-based页码，不需要转换
                    chunk_page_in_file = chunk.get("page")
                    if chunk_page_in_file == chunk_page:
                        # 检查chunk的文本是否匹配
                        chunk_text_in_file = chunk.get("text", "")
                        if chunk_text_in_file == chunk_text or chunk_text in chunk_text_in_file:
                            target_chunk = chunk
                            print(f"[DEBUG] 找到匹配的chunk (页码起始匹配): page={chunk_page}, chunk实际页码={chunk_page}, chunk_text_in_file长度={len(chunk_text_in_file)}")
                            break
            
            # 策略3：如果策略2失败，尝试仅通过文本匹配（处理跨页chunk的情况）
            if not target_chunk:
                print(f"[DEBUG] 未通过页码找到chunk，尝试全量文本匹配: chunk_page={chunk_page}")
                for chunk in chunks:
                    chunk_text_in_file = chunk.get("text", "")
                    # 只有当检索文本在chunk文本中时才匹配
                    if chunk_text_in_file == chunk_text or chunk_text in chunk_text_in_file:
                        target_chunk = chunk
                        print(f"[DEBUG] 找到匹配的chunk (文本匹配): retrieval_page={chunk_page}, chunk_start_page={chunk.get('page')}")
                        break
            
            if not target_chunk:
                print(f"[WARNING] 未找到匹配的chunk: chunk_page={chunk_page}")
                return {"page": chunk_page, "bbox_coords": "", "score": 0.0, "all_bboxes": []}
            
            # 获取该chunk的所有bbox
            bboxes = target_chunk.get("bboxes", [])
            print(f"[DEBUG] chunk包含 {len(bboxes)} 个bboxes")
            if not bboxes:
                print(f"[WARNING] chunk中没有bboxes: chunk_page={chunk_page}")
                return {"page": chunk_page, "bbox_coords": "", "score": 0.0, "all_bboxes": []}
            
            # 找到最相关的bbox
            def normalize_text(text):
                text = text.lower().replace(' ', '').replace('\n', '').replace('\t', '')
                return text
            
            def find_text_position(text, chunk_text):
                """
                在chunk text中查找text的位置
                
                参数:
                    text: 要查找的文本
                    chunk_text: chunk的完整文本
                
                返回:
                    (start_pos, end_pos) 找到的位置，未找到返回(-1, -1)
                """
                if not text or not chunk_text:
                    return -1, -1
                
                # 先尝试精确匹配
                pos = chunk_text.find(text)
                if pos != -1:
                    return pos, pos + len(text)
                
                # 尝试规范化后匹配
                normalized_text = normalize_text(text)
                normalized_chunk = normalize_text(chunk_text)
                pos = normalized_chunk.find(normalized_text)
                if pos != -1:
                    return pos, pos + len(normalized_text)
                
                return -1, -1
            
            def extract_numbers_and_amounts(text):
                """
                从文本中提取数字和金额
                
                返回:
                    numbers: 所有数字的列表（去重）
                    amounts: 所有金额的列表（去重，包含货币符号或单位）
                """
                # 提取所有数字（包括小数、百分比）
                numbers = re.findall(r'\d+\.?\d*%', text)
                numbers += re.findall(r'\d+\.?\d*', text)
                
                # 提取金额（包含货币符号或单位）
                amounts = re.findall(r'[\d,]+\.?\d*\s*(元|万元|亿元|美元|万|亿)', text)
                amounts += re.findall(r'¥[\d,]+\.?\d*', text)
                amounts += re.findall(r'\$[\d,]+\.?\d*', text)
                
                return list(set(numbers)), list(set(amounts))
            
            def extract_metric_value_pairs(text):
                """
                提取指标名称和对应的数值对
                
                参数:
                    text: 文本内容
                
                返回:
                    list: 指标名称和数值对的列表，格式为 [{"metric": "指标名", "value": "数值"}]
                """
                pairs = []
                
                # 定义财务指标关键词
                financial_metrics = [
                    '营业收入', '营收', '收入', '利润', '财务', '资产', '负债', 
                    '现金流', '净利润', '毛利率', '净利率', '每股收益', '市盈率', 
                    '市净率', '增长率', '同比', '环比', '总资产', '净资产', 
                    '营业成本', '营业利润', '投资收益', '财务费用', '管理费用',
                    '销售费用', '研发费用', '营业外收入', '营业外支出', '所得税',
                    '每股净资产', '每股收益', '净资产收益率', '资产负债率',
                    '流动比率', '速动比率', '存货周转率', '应收账款周转率',
                    '总资产周转率', '毛利率', '净利率', '营业利润率'
                ]
                
                # 提取金额（包含货币符号或单位）
                amount_pattern = r'([\d,]+\.?\d*)\s*(元|万元|亿元|美元|万|亿|%)'
                amounts = re.finditer(amount_pattern, text)
                
                for metric in financial_metrics:
                    # 查找指标名称附近是否有数值
                    metric_positions = []
                    for match in re.finditer(metric, text):
                        metric_positions.append(match.start())
                    
                    for metric_pos in metric_positions:
                        # 在指标名称附近查找数值（前后50个字符）
                        search_start = max(0, metric_pos - 50)
                        search_end = min(len(text), metric_pos + len(metric) + 50)
                        search_text = text[search_start:search_end]
                        
                        # 查找数值
                        value_matches = re.finditer(amount_pattern, search_text)
                        for value_match in value_matches:
                            value = value_match.group(1) + value_match.group(2)
                            pairs.append({
                                "metric": metric,
                                "value": value,
                                "position": metric_pos
                            })
                
                return pairs
            
            def calculate_metric_value_match_score(bbox_text, retrieved_text):
                """
                计算指标名称和数值的匹配分数
                
                参数:
                    bbox_text: bbox的文本内容
                    retrieved_text: 检索到的文本内容
                
                返回:
                    float: 匹配分数，分数越高表示指标和数值匹配度越高
                """
                bbox_pairs = extract_metric_value_pairs(bbox_text)
                retrieved_pairs = extract_metric_value_pairs(retrieved_text)
                
                if not bbox_pairs or not retrieved_pairs:
                    return 0
                
                match_score = 0
                
                # 检查是否有相同的指标名称和数值
                for bbox_pair in bbox_pairs:
                    for retrieved_pair in retrieved_pairs:
                        # 指标名称相同
                        if bbox_pair["metric"] == retrieved_pair["metric"]:
                            # 数值也相同
                            if bbox_pair["value"] == retrieved_pair["value"]:
                                match_score += 2000
                                print(f"[DEBUG] 指标和数值完全匹配: {bbox_pair['metric']} = {bbox_pair['value']}")
                            else:
                                # 指标名称相同但数值不同，给予较低分数
                                match_score += 200
                                print(f"[DEBUG] 指标名称匹配但数值不同: {bbox_pair['metric']} (bbox={bbox_pair['value']}, retrieved={retrieved_pair['value']})")
                
                return match_score
            
            def extract_years_from_text(text):
                """
                从文本中提取年份信息
                """
                # print(f"[DEBUG] extract_years_from_text 输入: '{text}'")
                all_years = set()
                
                if not text:
                    return all_years

                # Debug: Check for 2024A
                if '2024A' in text:
                    print(f"[DEBUG] extract_years_from_text 发现2024A: {text[:50]}")
                
                # 匹配各种年份格式：
                # 1. 2024年格式
                years_with_nian = re.findall(r'(\d{4})年', text)
                all_years.update(years_with_nian)
                
                # 2. 2024A, 2024E格式
                years_with_letter = re.findall(r'(\d{4})[AaEe]', text)
                all_years.update(years_with_letter)
                
                # 3. 独立的4位年份
                # 使用负向先行断言和后行断言，确保前后不是数字
                standalone_years = re.findall(r'(?<!\d)(\d{4})(?!\d)', text)
                all_years.update(standalone_years)
                
                # 4. 年份-年份格式（如2024-2025）
                range_pattern = r'(\d{4})-(\d{4})'
                ranges = re.findall(range_pattern, text)
                for start_year, end_year in ranges:
                    start = int(start_year)
                    end = int(end_year)
                    for year in range(start, end + 1):
                        all_years.add(str(year))
                
                if '2024A' in text:
                    print(f"[DEBUG] extract_years_from_text for 2024A result: {all_years}")

                return all_years
            
            def extract_query_keywords(question):
                """
                从查询问题中提取关键词
                
                参数:
                    question: 查询问题文本
                
                返回:
                    dict: 包含提取的关键词信息
                """
                print(f"[DEBUG] extract_query_keywords 输入问题: '{question}'")
                if not question:
                    return {
                        "years": set(),
                        "metrics": set(),
                        "numbers": set(),
                        "amounts": set(),
                        "query_type": "unknown"
                    }
                
                years = extract_years_from_text(question)
                print(f"[DEBUG] extract_query_keywords 提取年份: {years}")
                
                keywords = {
                    "years": years,
                    "metrics": set(),
                    "numbers": set(),
                    "amounts": set(),
                    "query_type": "unknown"
                }
                
                # 提取财务指标关键词
                financial_metrics = [
                    '营业收入', '营收', '收入', '利润', '财务', '资产', '负债', 
                    '现金流', '净利润', '毛利率', '净利率', '每股收益', '市盈率', 
                    '市净率', '增长率', '同比', '环比', '总资产', '净资产', 
                    '营业成本', '营业利润', '投资收益', '财务费用', '管理费用',
                    '销售费用', '研发费用', '营业外收入', '营业外支出', '所得税',
                    '每股净资产', '净资产收益率', '资产负债率', '流动比率', 
                    '速动比率', '存货周转率', '应收账款周转率', '总资产周转率'
                ]
                
                for metric in financial_metrics:
                    if metric in question:
                        keywords["metrics"].add(metric)
                
                # 提取数字和金额
                numbers, amounts = extract_numbers_and_amounts(question)
                keywords["numbers"].update(numbers)
                keywords["amounts"].update(amounts)
                
                # 判断查询类型：是询问具体数值还是增长率/比率
                growth_rate_keywords = ['增长率', '同比', '环比', '增长', '下降', '涨幅', '跌幅']
                ratio_keywords = ['毛利率', '净利率', '资产负债率', '流动比率', '速动比率', '净资产收益率', '市盈率', '市净率']
                value_query_patterns = ['是多少', '多少', '为多少', '是', '达到', '达到多少']
                
                # 检测是否询问增长率
                is_growth_query = any(keyword in question for keyword in growth_rate_keywords)
                
                # 检测是否询问比率
                is_ratio_query = any(keyword in question for keyword in ratio_keywords)
                
                # 检测是否询问具体数值
                is_value_query = any(pattern in question for pattern in value_query_patterns)
                
                # 判断查询类型
                if is_growth_query or is_ratio_query:
                    keywords["query_type"] = "percentage"
                elif is_value_query or (keywords["metrics"] and not is_growth_query and not is_ratio_query):
                    keywords["query_type"] = "value"
                else:
                    keywords["query_type"] = "unknown"
                
                return keywords

            def simple_match_score(bbox_text, retrieved_text, chunk_text, question_years=None, query_keywords=None, question=None, bbox_type=None):
                """
                计算bbox文本与检索文本的匹配分数
                
                参数:
                    bbox_text: bbox的文本内容
                    retrieved_text: 从向量数据库检索到的文本
                    chunk_text: chunk的完整文本
                    question_years: 查询中的年份信息，用于优先匹配
                    query_keywords: 查询中的关键词信息
                    question: 原始查询问题，用于大模型语义评分
                
                返回:
                    匹配分数，分数越高表示越相关
                """
                if not bbox_text or not retrieved_text:
                    return 0
                
                normalized_bbox = normalize_text(bbox_text)
                normalized_retrieved = normalize_text(retrieved_text)
                
                if not normalized_bbox or not normalized_retrieved:
                    return 0
                
                score = 0
                
                # 获取查询类型
                query_type = query_keywords.get("query_type", "unknown") if query_keywords else "unknown"
                print(f"[DEBUG] 查询类型: {query_type}")
                
                # 检测bbox中的数据类型（改进版：更准确地判断主要数据类型）
                # 统计百分比数据的数量
                percentage_keywords = ['增长率', '同比', '环比', '毛利率', '净利率', '资产负债率', '流动比率', '速动比率', '净资产收益率', '市盈率', '市净率', '%']
                percentage_count = sum(1 for keyword in percentage_keywords if keyword in bbox_text)
                
                # 统计金额数据的数量（排除"每股指标"等非金额场景）
                amount_keywords = ['万元', '亿元', '美元', '¥', '$']
                amount_count = sum(1 for keyword in amount_keywords if keyword in bbox_text)
                
                # 增强的金额数据检测：检查是否存在非年份的大数值
                # 1. 移除百分比数字
                text_no_pct = re.sub(r'\d+(\.\d+)?%', ' ', bbox_text)
                # 2. 提取剩余数字
                raw_nums = re.findall(r'\d+(?:\.\d+)?', text_no_pct)
                has_large_number = False
                for n in raw_nums:
                    try:
                        val = float(n)
                        # 过滤掉可能的年份 (1990-2030) 和小整数 (0-100，可能是序号或小数值)
                        # 关注大于100的数值，通常是金额
                        if val > 100 and not (1990 <= val <= 2030 and val.is_integer()):
                            has_large_number = True
                            # print(f"[DEBUG] 发现大数值: {val}，视为金额数据")
                            break
                    except:
                        pass
                
                if has_large_number:
                    amount_count += 1
                
                # 检查是否包含"元"字，但排除"每股指标"等场景
                if '元' in bbox_text:
                    # 检查"元"字是否出现在"每股指标"、"每股收益"、"每股净资产"、"每股经营现金流"、"每股股利"等上下文中
                    # 如果是，则不应该认为包含金额数据
                    per_share_keywords = ['每股指标', '每股收益', '每股净资产', '每股经营现金流', '每股股利', '收盘价']
                    if not any(keyword in bbox_text for keyword in per_share_keywords):
                        amount_count += 1
                
                # 根据数量判断主要数据类型
                has_percentage_data = percentage_count > 0
                has_amount_data = amount_count > 0

                # 预先检查是否有关键词匹配
                has_relevant_keywords = False
                if query_keywords:
                    query_metrics = query_keywords.get("metrics", set())
                    query_years = query_keywords.get("years", set())
                    if any(metric in bbox_text for metric in query_metrics) or \
                       any(year in bbox_text for year in query_years):
                        has_relevant_keywords = True
                
                # 1.5 表格类型加分 (降低盲目加分，仅当包含关键词时给予高分)
                if bbox_type == 'table':
                    if has_relevant_keywords:
                        score += 50000
                        print(f"[DEBUG] bbox类型为table且包含关键词，+50000分")
                    else:
                        score += 5000
                        print(f"[DEBUG] bbox类型为table但不含关键词，+5000分")
                
                # 如果同时包含两种类型，判断哪种类型占主导
                if has_percentage_data and has_amount_data:
                    # 如果检测到大数值（通常是金额），即使百分比数量多，也不应该认为是纯百分比类型
                    if has_large_number:
                        print(f"[DEBUG] bbox同时包含百分比和金额数据，且包含大数值，判定为混合类型（保留金额属性）")
                    elif percentage_count > amount_count:
                        has_amount_data = False  # 百分比占主导，且无大数值，不认为包含金额数据
                        print(f"[DEBUG] bbox同时包含百分比和金额数据，但百分比占主导（{percentage_count} vs {amount_count}），判定为百分比类型")
                    elif amount_count > percentage_count:
                        has_percentage_data = False  # 金额占主导，不认为包含百分比数据
                        print(f"[DEBUG] bbox同时包含百分比和金额数据，但金额占主导（{amount_count} vs {percentage_count}），判定为金额类型")
                    else:
                        print(f"[DEBUG] bbox同时包含百分比和金额数据，数量相当（{percentage_count} vs {amount_count}），判定为混合类型")
                
                print(f"[DEBUG] bbox数据类型: has_percentage_data={has_percentage_data}, has_amount_data={has_amount_data}")
                
                # 根据查询类型调整评分权重（这是最重要的评分因素）
                if query_type == "value":
                    # 如果查询具体数值，优先选择包含金额数据的bbox
                    if has_amount_data and not has_percentage_data:
                        if has_relevant_keywords:
                            score += 500000  # 大幅提升包含金额数据的bbox分数（从5000提升到500000）
                            print(f"[DEBUG] 查询类型为value，bbox包含金额数据且含关键词，+500000分")
                        else:
                            score += 50000   # 不含关键词但含金额，给予较低提升
                            print(f"[DEBUG] 查询类型为value，bbox包含金额数据但不含关键词，+50000分")
                    elif has_amount_data and has_percentage_data:
                        if has_large_number:
                            if has_relevant_keywords:
                                score += 500000  # 包含大数值的混合数据，视为高质量金额数据，给予最高分
                                print(f"[DEBUG] 查询类型为value，bbox混合且含大数值和关键词，+500000分")
                            else:
                                score += 50000
                                print(f"[DEBUG] 查询类型为value，bbox混合且含大数值但不含关键词，+50000分")
                        else:
                            score += 250000  # 同时包含金额和百分比数据，给予中等分数（从2500提升到250000）
                            print(f"[DEBUG] 查询类型为value，bbox同时包含金额和百分比数据，+250000分")
                    elif has_percentage_data and not has_amount_data:
                        score -= 200000  # 只包含百分比数据，大幅降低分数（从-2000提升到-200000）
                        print(f"[DEBUG] 查询类型为value，bbox只包含百分比数据，-200000分")
                
                elif query_type == "percentage":
                    # 如果查询增长率/比率，优先选择包含百分比数据的bbox
                    if has_percentage_data and not has_amount_data:
                        score += 500000  # 大幅提升包含百分比数据的bbox分数（从5000提升到500000）
                        print(f"[DEBUG] 查询类型为percentage，bbox包含百分比数据，+500000分")
                    elif has_percentage_data and has_amount_data:
                        score += 250000  # 同时包含百分比和金额数据，给予中等分数（从2500提升到250000）
                        print(f"[DEBUG] 查询类型为percentage，bbox同时包含百分比和金额数据，+250000分")
                    elif has_amount_data and not has_percentage_data:
                        score -= 200000  # 只包含金额数据，大幅降低分数（从-2000提升到-200000）
                        print(f"[DEBUG] 查询类型为percentage，bbox只包含金额数据，-200000分")
                
                # 0. 大模型语义相关性评分（最高优先级）
                # 性能优化：禁用bbox级别的大模型评分，因其严重拖慢检索速度（每个bbox一次API调用）
                if question and False:  # 强制禁用
                    semantic_score = self.calculate_semantic_relevance_score(question, bbox_text)
                    # 将语义分数（0-1）转换为评分权重（0-1000000）
                    # 这样语义相关性最高的bbox可以获得1000000分
                    semantic_weight = int(semantic_score * 1000000)
                    score += semantic_weight
                    print(f"[DEBUG] 大模型语义相关性评分: {semantic_score}, 转换为权重: {semantic_weight}")
                
                # 1. 查询关键词匹配（高优先级，但低于类型匹配和大模型语义评分）
                if query_keywords:
                    # 指标名称完全匹配（最高优先级，仅次于类型匹配）
                    query_metrics = query_keywords.get("metrics", set())
                    for metric in query_metrics:
                        # 检查bbox中是否包含完全一致的指标名称
                        if metric in bbox_text:
                            # 检查是否是独立的指标名称（不是其他指标的一部分）
                            # 例如："营业收入"应该匹配，但"营业收入增长率"不应该匹配"营业收入"
                            # 使用更精确的方法：检查指标名称前后是否有分隔符
                            # 将bbox_text按常见分隔符分割，然后检查是否有完全匹配的词
                            # 使用正则表达式匹配独立的指标名称
                            # 匹配指标名称前后是表格单元格边界、空格、换行符或标点符号的情况
                            pattern = r'(?:^|[<>\s，、。；：\n\r])' + re.escape(metric) + r'(?:$|[<>\s，、。；：\n\r])'
                            if re.search(pattern, bbox_text):
                                score += 100000  # 指标名称完全匹配，给予高分
                                print(f"[DEBUG] 查询指标完全匹配: {metric} 在bbox文本中找到，+100000分")
                            else:
                                score += 5000  # 指标名称部分匹配，给予较低分
                                print(f"[DEBUG] 查询指标部分匹配: {metric} 在bbox文本中找到，+5000分")
                            
                            # 检查是否是增长率指标但用户查询的是数值
                            if query_type == "value":
                                # 检查指标后是否紧跟"增长率"、"增速"、"同比"等词
                                # 考虑到HTML标签可能存在，放宽检查范围
                                indices = [m.start() for m in re.finditer(re.escape(metric), bbox_text)]
                                is_growth_metric = False
                                for idx in indices:
                                    # 检查接下来的20个字符
                                    snippet = bbox_text[idx+len(metric):idx+len(metric)+20]
                                    if '增长率' in snippet or '增速' in snippet or '同比' in snippet:
                                        is_growth_metric = True
                                        break
                                
                                if is_growth_metric:
                                    score -= 300000
                                    print(f"[DEBUG] 查询类型为value，但bbox包含{metric}增长率/增速，判定为不匹配，-300000分")
                    
                    # 年份匹配（大幅提升权重）
                    query_years = query_keywords.get("years", set())
                    bbox_years = extract_years_from_text(bbox_text)
                    print(f"[DEBUG] 年份检测: query_years={query_years}, bbox_years={bbox_years}, bbox_text片段='{bbox_text[:20]}...'")
                    for year in query_years:
                        if year in bbox_years:
                            score += 200000  # 从10000提升到200000（确保优先选择包含目标年份的bbox）
                            print(f"[DEBUG] 查询年份匹配: {year} 在bbox文本中找到，+200000分")
                    
                    # 数值匹配（降低权重）
                    query_numbers = query_keywords.get("numbers", set())
                    bbox_numbers, _ = extract_numbers_and_amounts(bbox_text)
                    for num in query_numbers:
                        if num in bbox_numbers:
                            score += 3000  # 从1500降低到3000
                            print(f"[DEBUG] 查询数值匹配: {num} 在bbox文本中找到，+3000分")
                
                # 2. 查找bbox和检索文本在chunk中的位置（大幅降低权重）
                bbox_start, bbox_end = find_text_position(bbox_text, chunk_text)
                retrieved_start, retrieved_end = find_text_position(retrieved_text, chunk_text)
                
                # 3. 位置重叠度评分（大幅降低权重）
                if bbox_start != -1 and retrieved_start != -1:
                    # 计算重叠长度
                    overlap_start = max(bbox_start, retrieved_start)
                    overlap_end = min(bbox_end, retrieved_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    
                    # 重叠比例
                    overlap_ratio = overlap_length / min(bbox_end - bbox_start, retrieved_end - retrieved_start)
                    score += overlap_ratio * 100  # 从500降低到100
                
                # 4. 完全包含检查（大幅降低权重）
                if normalized_retrieved in normalized_bbox:
                    score += 50  # 从400降低到50
                if normalized_bbox in normalized_retrieved:
                    score += 30  # 从300降低到30
                
                # 5. 长文本的片段匹配检查（大幅降低权重）
                if len(normalized_retrieved) >= 50:
                    # 检查检索文本的前半部分
                    snippet = normalized_retrieved[:100]
                    if snippet in normalized_bbox:
                        score += 10  # 从75降低到10
                    
                    # 检查检索文本的后半部分
                    snippet = normalized_retrieved[-100:]
                    if snippet in normalized_bbox:
                        score += 10  # 从75降低到10
                    
                    # 检查检索文本的中间部分
                    if len(normalized_retrieved) >= 200:
                        mid_start = len(normalized_retrieved) // 2 - 50
                        mid_end = len(normalized_retrieved) // 2 + 50
                        snippet = normalized_retrieved[mid_start:mid_end]
                        if snippet in normalized_bbox:
                            score += 10  # 从75降低到10
                
                # 6. 滑动窗口匹配检查（大幅降低权重）
                window_size = 100
                step = 50
                matched_windows = 0
                for i in range(0, len(normalized_retrieved), step):
                    snippet = normalized_retrieved[i:i+window_size]
                    if snippet and snippet in normalized_bbox:
                        matched_windows += 1
                score += matched_windows * 2  # 从15降低到2
                
                # 7. 公共字符数量（大幅降低权重）
                common_chars = set(normalized_bbox) & set(normalized_retrieved)
                score += len(common_chars) * 0.01  # 从0.1降低到0.01
                
                # 8. 长度相似度（大幅降低权重）
                length_ratio = min(len(normalized_bbox), len(normalized_retrieved)) / max(len(normalized_bbox), len(normalized_retrieved))
                score += length_ratio * 5  # 从25降低到5
                
                # 9. 数字和金额匹配评分（大幅降低权重）
                bbox_numbers, bbox_amounts = extract_numbers_and_amounts(bbox_text)
                retrieved_numbers, retrieved_amounts = extract_numbers_and_amounts(retrieved_text)
                
                # 计算数字重叠度（大幅降低权重）
                if bbox_numbers and retrieved_numbers:
                    bbox_numbers_set = set(bbox_numbers)
                    retrieved_numbers_set = set(retrieved_numbers)
                    overlap_numbers = bbox_numbers_set & retrieved_numbers_set
                    
                    # 数字重叠数量越多，分数越高
                    if overlap_numbers:
                        # 每个重叠数字给予低分
                        score += len(overlap_numbers) * 10  # 从200降低到10
                        
                        # 如果重叠数字数量较多，额外加分
                        if len(overlap_numbers) >= 3:
                            score += 20  # 从400降低到20
                        elif len(overlap_numbers) >= 2:
                            score += 10  # 从200降低到10
                
                # 计算金额重叠度（大幅降低权重）
                if bbox_amounts and retrieved_amounts:
                    bbox_amounts_set = set(bbox_amounts)
                    retrieved_amounts_set = set(retrieved_amounts)
                    overlap_amounts = bbox_amounts_set & retrieved_amounts_set
                    
                    # 金额重叠数量越多，分数越高
                    if overlap_amounts:
                        # 每个重叠金额给予低分
                        score += len(overlap_amounts) * 20  # 从300降低到20
                        
                        # 如果有金额重叠，额外加分
                        score += 50  # 从600降低到50
                
                # 10. 关键词匹配（财务指标）- 大幅降低权重
                financial_keywords = ['营业收入', '营收', '收入', '利润', '财务', '资产', '负债', '现金流', '净利润', '毛利率', '净利率', '每股收益', '市盈率', '市净率', '增长率', '同比', '环比']
                for keyword in financial_keywords:
                    if keyword in bbox_text and keyword in retrieved_text:
                        score += 10  # 从100降低到10
                
                # 11. 指标名称和数值的精确匹配（降低权重）
                metric_value_score = calculate_metric_value_match_score(bbox_text, retrieved_text)
                score += metric_value_score
                
                return score
            
            # 提取查询中的年份和关键词信息
            question_years = None
            query_keywords = None
            if question:
                question_years = extract_years_from_text(question)
                query_keywords = extract_query_keywords(question)
                print(f"[DEBUG] 从查询中提取的年份: {question_years}")
                print(f"[DEBUG] 从查询中提取的关键词: metrics={query_keywords.get('metrics', set())}, numbers={query_keywords.get('numbers', set())}")
            
            # 对每个bbox进行评分
            scored_bboxes = []
            valid_bbox_count = 0
            for bbox_info in bboxes:
                bbox_text = bbox_info.get("text", "")
                bbox_page = bbox_info.get("page", chunk_page)
                bbox_coords = bbox_info.get("bbox", [])
                
                # chunk可能跨越多个页面，bbox的page是实际页码，可能大于chunk的page
                # 不进行page过滤，让评分逻辑决定哪个bbox最相关
                print(f"[DEBUG] 处理bbox: bbox_page={bbox_page}, chunk_page={chunk_page}, text='{bbox_text[:50]}...'")
                
                # 检查bbox是否为空白或无关
                if not bbox_text or len(bbox_text.strip()) < 5:
                    print(f"[DEBUG] 跳过空白或过短的bbox: page={bbox_page}, text='{bbox_text}'")
                    continue
                
                valid_bbox_count += 1
                
                # 使用chunk的完整文本进行评分
                chunk_text_in_file = target_chunk.get("text", "")
                bbox_type = bbox_info.get("type", "text")
                bbox_score = simple_match_score(bbox_text, chunk_text, chunk_text_in_file, question_years, query_keywords, question, bbox_type)
                
                print(f"[DEBUG] bbox评分: page={chunk_page}, bbox={bbox_coords}, score={bbox_score:.2f}, text='{bbox_text[:50]}...'")
                
                scored_bboxes.append({
                    "bbox_info": bbox_info,
                    "page": bbox_page,
                    "score": bbox_score
                })
            
            print(f"[DEBUG] 有效bbox数量: {valid_bbox_count}, 评分后bbox数量: {len(scored_bboxes)}")
            
            # 按分数排序，返回分数最高的bbox的page和bbox_coords，同时返回所有bbox的评分信息
            scored_bboxes.sort(key=lambda x: x["score"], reverse=True)
            if scored_bboxes:
                best_bbox = scored_bboxes[0]
                bbox_coords = best_bbox["bbox_info"].get("bbox", [])
                bbox_text = best_bbox["bbox_info"].get("text", "")
                bbox_coords_str = ""
                if len(bbox_coords) == 4:
                    bbox_coords_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
                    print(f"[DEBUG] 找到最相关bbox: page={best_bbox['page']}, score={best_bbox['score']:.2f}, bbox_coords={bbox_coords_str}, text='{bbox_text[:100]}...'")
                else:
                    print(f"[DEBUG] bbox坐标格式不正确: bbox={bbox_coords}, 长度={len(bbox_coords)}")
                
                # 构建所有bbox的评分信息列表
                all_bboxes_info = []
                for bbox_item in scored_bboxes:
                    bbox_item_coords = bbox_item["bbox_info"].get("bbox", [])
                    bbox_item_coords_str = ""
                    if len(bbox_item_coords) == 4:
                        bbox_item_coords_str = f"{bbox_item_coords[0]},{bbox_item_coords[1]},{bbox_item_coords[2]},{bbox_item_coords[3]}"
                    all_bboxes_info.append({
                        "page": bbox_item["page"],
                        "bbox_coords": bbox_item_coords_str,
                        "score": bbox_item["score"]
                    })
                
                return {
                    "page": best_bbox['page'], 
                    "bbox_coords": bbox_coords_str,
                    "score": best_bbox['score'],
                    "all_bboxes": all_bboxes_info
                }
            else:
                print(f"[WARNING] 没有找到有效的bbox: chunk_page={chunk_page}, valid_bbox_count={valid_bbox_count}")
            
            return {"page": chunk_page, "bbox_coords": "", "score": 0.0, "all_bboxes": []}
        except Exception as e:
            print(f"[ERROR] 查找bbox时出错: {e}")
            return {"page": chunk_page, "bbox_coords": "", "score": 0.0, "all_bboxes": []}

    # 注释：_calculate_bbox_score_for_page函数已废弃，bbox评分和坐标信息现在由_find_most_relevant_bbox_page函数统一提供
    # def _calculate_bbox_score_for_page(self, page: int, retrieval_result: dict, question: str = None) -> dict:
        """
        计算指定页码对应的bbox的评分，并返回bbox坐标
        
        参数:
            page: bbox级别的page（0-based，来自chunked_reports的page）
            retrieval_result: 单个检索结果
            question: 原始查询问题，用于提取年份等信息
        
        返回:
            dict: 包含bbox评分和bbox坐标的字典，格式为 {"score": float, "bbox_coords": str}
        """
        file_name = retrieval_result.get('file_name', '')
        chunk_page = retrieval_result.get('page', 0)
        chunk_text = retrieval_result.get('text', '')
        
        # 查找对应的chunked JSON文件
        json_path = None
        for path in self.documents_dir.glob("*.json"):
            if path.name == file_name:
                json_path = path
                break
        
        if not json_path:
            return {"score": 0.0, "bbox_coords": ""}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get("content", {}).get("chunks", [])
            
            # 找到对应的chunk
            target_chunk = None
            for chunk in chunks:
                if chunk.get("page") == chunk_page:
                    chunk_text_in_file = chunk.get("text", "")
                    if chunk_text_in_file == chunk_text or chunk_text in chunk_text_in_file:
                        target_chunk = chunk
                        break
            
            if not target_chunk:
                return {"score": 0.0, "bbox_coords": ""}
            
            # 获取该chunk的所有bbox
            bboxes = target_chunk.get("bboxes", [])
            if not bboxes:
                return {"score": 0.0, "bbox_coords": ""}
            
            # 找到指定page的bbox并计算评分
            def normalize_text(text):
                text = text.lower().replace(' ', '').replace('\n', '').replace('\t', '')
                return text
            
            def find_text_position(text, chunk_text):
                if not text or not chunk_text:
                    return -1, -1
                
                pos = chunk_text.find(text)
                if pos != -1:
                    return pos, pos + len(text)
                
                normalized_text = normalize_text(text)
                normalized_chunk = normalize_text(chunk_text)
                pos = normalized_chunk.find(normalized_text)
                if pos != -1:
                    return pos, pos + len(normalized_text)
                
                return -1, -1
            
            def extract_numbers_and_amounts(text):
                numbers = re.findall(r'\d+\.?\d*%', text)
                numbers += re.findall(r'\d+\.?\d*', text)
                
                amounts = re.findall(r'[\d,]+\.?\d*\s*(元|万元|亿元|美元|万|亿)', text)
                amounts += re.findall(r'¥[\d,]+\.?\d*', text)
                amounts += re.findall(r'\$[\d,]+\.?\d*', text)
                
                return list(set(numbers)), list(set(amounts))
            
            def extract_metric_value_pairs(text):
                """
                提取指标名称和对应的数值对
                
                参数:
                    text: 文本内容
                
                返回:
                    list: 指标名称和数值对的列表，格式为 [{"metric": "指标名", "value": "数值"}]
                """
                pairs = []
                
                # 定义财务指标关键词
                financial_metrics = [
                    '营业收入', '营收', '收入', '利润', '财务', '资产', '负债', 
                    '现金流', '净利润', '毛利率', '净利率', '每股收益', '市盈率', 
                    '市净率', '增长率', '同比', '环比', '总资产', '净资产', 
                    '营业成本', '营业利润', '投资收益', '财务费用', '管理费用',
                    '销售费用', '研发费用', '营业外收入', '营业外支出', '所得税',
                    '每股净资产', '每股收益', '净资产收益率', '资产负债率',
                    '流动比率', '速动比率', '存货周转率', '应收账款周转率',
                    '总资产周转率', '毛利率', '净利率', '营业利润率'
                ]
                
                # 提取金额（包含货币符号或单位）
                amount_pattern = r'([\d,]+\.?\d*)\s*(元|万元|亿元|美元|万|亿|%)'
                amounts = re.finditer(amount_pattern, text)
                
                for metric in financial_metrics:
                    # 查找指标名称附近是否有数值
                    metric_positions = []
                    for match in re.finditer(metric, text):
                        metric_positions.append(match.start())
                    
                    for metric_pos in metric_positions:
                        # 在指标名称附近查找数值（前后50个字符）
                        search_start = max(0, metric_pos - 50)
                        search_end = min(len(text), metric_pos + len(metric) + 50)
                        search_text = text[search_start:search_end]
                        
                        # 查找数值
                        value_matches = re.finditer(amount_pattern, search_text)
                        for value_match in value_matches:
                            value = value_match.group(1) + value_match.group(2)
                            pairs.append({
                                "metric": metric,
                                "value": value,
                                "position": metric_pos
                            })
                
                return pairs
            
            def extract_years_from_text(text):
                """
                从文本中提取年份信息
                
                参数:
                    text: 输入文本
                
                返回:
                    set: 包含年份的集合
                """
                all_years = set()
                
                # 匹配各种年份格式：
                # 1. 2024年格式
                years_with_nian = re.findall(r'(\d{4})年', text)
                all_years.update(years_with_nian)
                
                # 2. 2024A, 2024E格式
                years_with_letter = re.findall(r'(\d{4})[AaEe]', text)
                all_years.update(years_with_letter)
                
                # 3. 独立的4位年份，前后有空格或标点
                standalone_years = re.findall(r'(?:^|[\s\(（])(\d{4})(?=[\s\)）]|$|[^\d])', text)
                all_years.update(standalone_years)
                
                # 4. 年份-年份格式（如2024-2025）
                range_pattern = r'(\d{4})-(\d{4})'
                ranges = re.findall(range_pattern, text)
                for start_year, end_year in ranges:
                    start = int(start_year)
                    end = int(end_year)
                    for year in range(start, end + 1):
                        all_years.add(str(year))
                
                return all_years
            
            def calculate_metric_value_match_score(bbox_text, retrieved_text):
                """
                计算指标名称和数值的匹配分数
                
                参数:
                    bbox_text: bbox的文本内容
                    retrieved_text: 检索到的文本内容
                
                返回:
                    float: 匹配分数，分数越高表示指标和数值匹配度越高
                """
                bbox_pairs = extract_metric_value_pairs(bbox_text)
                retrieved_pairs = extract_metric_value_pairs(retrieved_text)
                
                if not bbox_pairs or not retrieved_pairs:
                    return 0
                
                match_score = 0
                
                # 检查是否有相同的指标名称和数值
                for bbox_pair in bbox_pairs:
                    for retrieved_pair in retrieved_pairs:
                        # 指标名称相同
                        if bbox_pair["metric"] == retrieved_pair["metric"]:
                            # 数值也相同
                            if bbox_pair["value"] == retrieved_pair["value"]:
                                match_score += 2000
                                print(f"[DEBUG] 指标和数值完全匹配: {bbox_pair['metric']} = {bbox_pair['value']}")
                            else:
                                # 指标名称相同但数值不同，给予较低分数
                                match_score += 200
                                print(f"[DEBUG] 指标名称匹配但数值不同: {bbox_pair['metric']} (bbox={bbox_pair['value']}, retrieved={retrieved_pair['value']})")
                
                return match_score
            
            def simple_match_score(bbox_text, retrieved_text, chunk_text, question_years=None):
                """
                计算bbox文本与检索文本的匹配分数
                
                参数:
                    bbox_text: bbox的文本内容
                    retrieved_text: 从向量数据库检索到的文本
                    chunk_text: chunk的完整文本
                    question_years: 查询中的年份信息，用于优先匹配
                
                返回:
                    匹配分数，分数越高表示越相关
                """
                if not bbox_text or not retrieved_text:
                    return 0
                
                normalized_bbox = normalize_text(bbox_text)
                normalized_retrieved = normalize_text(retrieved_text)
                
                if not normalized_bbox or not normalized_retrieved:
                    return 0
                
                score = 0
                
                bbox_start, bbox_end = find_text_position(bbox_text, chunk_text)
                retrieved_start, retrieved_end = find_text_position(retrieved_text, chunk_text)
                
                if bbox_start != -1 and retrieved_start != -1:
                    overlap_start = max(bbox_start, retrieved_start)
                    overlap_end = min(bbox_end, retrieved_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    overlap_ratio = overlap_length / min(bbox_end - bbox_start, retrieved_end - retrieved_start)
                    score += overlap_ratio * 1000
                
                if normalized_retrieved in normalized_bbox:
                    score += 800
                if normalized_bbox in normalized_retrieved:
                    score += 600
                
                if len(normalized_retrieved) >= 50:
                    snippet = normalized_retrieved[:100]
                    if snippet in normalized_bbox:
                        score += 150
                    
                    snippet = normalized_retrieved[-100:]
                    if snippet in normalized_bbox:
                        score += 150
                    
                    if len(normalized_retrieved) >= 200:
                        mid_start = len(normalized_retrieved) // 2 - 50
                        mid_end = len(normalized_retrieved) // 2 + 50
                        snippet = normalized_retrieved[mid_start:mid_end]
                        if snippet in normalized_bbox:
                            score += 150
                
                window_size = 100
                step = 50
                matched_windows = 0
                for i in range(0, len(normalized_retrieved), step):
                    snippet = normalized_retrieved[i:i+window_size]
                    if snippet and snippet in normalized_bbox:
                        matched_windows += 1
                score += matched_windows * 30
                
                common_chars = set(normalized_bbox) & set(normalized_retrieved)
                score += len(common_chars) * 0.3
                
                length_ratio = min(len(normalized_bbox), len(normalized_retrieved)) / max(len(normalized_bbox), len(normalized_retrieved))
                score += length_ratio * 50
                
                bbox_numbers, bbox_amounts = extract_numbers_and_amounts(bbox_text)
                retrieved_numbers, retrieved_amounts = extract_numbers_and_amounts(retrieved_text)
                
                if bbox_numbers and retrieved_numbers:
                    bbox_numbers_set = set(bbox_numbers)
                    retrieved_numbers_set = set(retrieved_numbers)
                    overlap_numbers = bbox_numbers_set & retrieved_numbers_set
                    
                    if overlap_numbers:
                        score += len(overlap_numbers) * 500
                        
                        if len(overlap_numbers) >= 3:
                            score += 1000
                        elif len(overlap_numbers) >= 2:
                            score += 600
                
                if bbox_amounts and retrieved_amounts:
                    bbox_amounts_set = set(bbox_amounts)
                    retrieved_amounts_set = set(retrieved_amounts)
                    overlap_amounts = bbox_amounts_set & retrieved_amounts_set
                    
                    if overlap_amounts:
                        score += len(overlap_amounts) * 800
                        score += 1500
                
                financial_keywords = ['营业收入', '营收', '收入', '利润', '财务', '资产', '负债', '现金流', '净利润', '毛利率', '净利率', '每股收益', '市盈率', '市净率', '增长率', '同比', '环比']
                for keyword in financial_keywords:
                    if keyword in bbox_text and keyword in retrieved_text:
                        score += 200
                
                metric_value_score = calculate_metric_value_match_score(bbox_text, retrieved_text)
                score += metric_value_score
                
                # 11. 年份匹配评分（如果提供了查询中的年份）
                if question_years:
                    bbox_years = extract_years_from_text(bbox_text)
                    if '2024A' in bbox_text:
                        print(f"[DEBUG] Year check: question_years={question_years}, bbox_years={bbox_years}")
                    for year in question_years:
                        if year in bbox_years:
                            score += 200000  # 年份匹配给予极高分
                            print(f"[DEBUG] 年份匹配: {year} 在bbox文本中找到，+200000分")
                
                return score
            
            # 找到指定page的bbox并计算评分
            chunk_text_in_file = target_chunk.get("text", "")
            for bbox_info in bboxes:
                bbox_page = bbox_info.get("page", chunk_page)
                if bbox_page == page:
                    bbox_text = bbox_info.get("text", "")
                    bbox_coords = bbox_info.get("bbox", [])
                    # 检查bbox是否为空白或无关
                    if not bbox_text or len(bbox_text.strip()) < 5:
                        print(f"[DEBUG] bbox内容为空或过短: page={page}, text='{bbox_text}'")
                        return {"score": 0.0, "bbox_coords": ""}
                    
                    # 提取查询中的年份信息
                    question_years = None
                    if question:
                        question_years = extract_years_from_text(question)
                        print(f"[DEBUG] 从查询中提取的年份: {question_years}")
                    
                    bbox_score = simple_match_score(bbox_text, chunk_text, chunk_text_in_file, question_years)
                    
                    # 将bbox坐标转换为字符串格式
                    bbox_coords_str = ""
                    if len(bbox_coords) == 4:
                        bbox_coords_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
                    
                    print(f"[DEBUG] bbox评分: page={page}, score={bbox_score:.2f}, bbox_coords={bbox_coords_str}, text='{bbox_text[:50]}...'")
                    return {"score": bbox_score, "bbox_coords": bbox_coords_str}
            
            return {"score": 0.0, "bbox_coords": ""}
        except Exception as e:
            print(f"[ERROR] 计算bbox评分时出错: {e}")
            return {"score": 0.0, "bbox_coords": ""}

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, question: str = None, min_pages: int = 1, max_pages: int = 3, min_score: float = 50.0) -> list:
        """
        校验LLM答案中引用的页码是否真实存在于检索结果中。
        如果LLM引用的页码不在检索结果中，则过滤掉这些无效页码。
        如果过滤后页码不足最小页数，则补充检索结果中的页码。
        增强版:不仅检查页码存在性,还验证页面内容是否包含相关财务数据。
        使用bbox级别的page而不是chunk级别的page。
        新增:对每个bbox进行评分,只返回评分最高的bbox对应的页码和bbox坐标。
        
        返回:
            list: 包含page和bbox_coords的字典列表，格式为 [{"page": page, "bbox_coords": bbox_coords}]
        """
        print(f"\n{'='*80}")
        print(f"[DEBUG] _validate_page_references 开始执行")
        print(f"[DEBUG] 输入:")
        print(f"[DEBUG]   claimed_pages={claimed_pages}")
        print(f"[DEBUG]   retrieval_results数量={len(retrieval_results) if retrieval_results else 0}")
        print(f"[DEBUG]   question={question}")
        print(f"[DEBUG]   min_pages={min_pages}, max_pages={max_pages}, min_score={min_score}")
        print(f"{'='*80}\n")
        
        if claimed_pages is None:
            claimed_pages = []
        
        # 从检索结果中提取公司名和文件名
        company_name = "未知公司"
        file_name = "未知文件"
        if retrieval_results:
            company_name = retrieval_results[0].get('company_name', '未知公司')
            file_name = retrieval_results[0].get('file_name', '未知文件')
        
        # 获取检索结果中实际存在的页码及其内容（使用bbox级别的page）
        # 同时保存每个page对应的bbox坐标和评分信息
        retrieved_pages_dict = {}
        retrieved_pages_bbox_info = {}  # 存储每个page对应的bbox信息和评分
        for result in retrieval_results:
            # 找到该检索结果最相关的bbox的page和坐标
            bbox_result = self._find_most_relevant_bbox_page(result, question)
            bbox_page = bbox_result.get("page", result.get('page', 0))
            bbox_coords = bbox_result.get("bbox_coords", "")
            bbox_score = bbox_result.get("score", 0.0)
            all_bboxes = bbox_result.get("all_bboxes", [])
            
            if bbox_page not in retrieved_pages_bbox_info or bbox_score > retrieved_pages_bbox_info[bbox_page]["score"]:
                retrieved_pages_dict[bbox_page] = result['text']
                retrieved_pages_bbox_info[bbox_page] = {
                    "bbox_coords": bbox_coords,
                    "score": bbox_score,
                    "all_bboxes": all_bboxes
                }
                print(f"[DEBUG] 更新page {bbox_page} 的bbox信息 (更优分数): coords={bbox_coords}, score={bbox_score:.2f}")
        
        retrieved_pages = list(retrieved_pages_dict.keys())
        retrieved_pages_set = set(retrieved_pages)
        
        # 定义财务数据相关关键词
        financial_keywords = ['营业收入', '营收', '收入', '利润', '财务', '资产', '负债', '现金流', '净利润', '毛利率', '净利率']
        
        # 优先保留LLM声称的页码中实际存在于检索结果中的页码（不过滤财务关键词）
        validated_pages = []
        for page in claimed_pages:
            if page in retrieved_pages_dict:
                # 直接保留LLM引用的页码，不检查财务关键词
                validated_pages.append(page)
            else:
                print(f"[WARNING] 页面 {page} 不存在于检索结果中,已过滤")
        
        # 如果LLM引用的页码全部无效,则使用检索结果中包含相关内容的页码
        if not validated_pages and retrieval_results:
            print(f"[WARNING] LLM引用的所有页码 {claimed_pages} 都无效,从检索结果中选择包含相关内容的页码")
            for page, content in retrieved_pages_dict.items():
                if any(keyword in content for keyword in financial_keywords):
                    validated_pages.append(page)
                if len(validated_pages) >= min_pages:
                    break
        
        # 如果页码不足最小页数,补充检索结果中包含相关内容的页码
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            
            for page, content in retrieved_pages_dict.items():
                if page not in existing_pages:
                    # 只补充包含财务数据的页面
                    if any(keyword in content for keyword in financial_keywords):
                        validated_pages.append(page)
                        existing_pages.add(page)
                        
                        if len(validated_pages) >= min_pages:
                            break
        
        # 对每个验证过的页码，使用之前保存的bbox信息和评分
        scored_pages = []
        for page in validated_pages:
            if page in retrieved_pages_bbox_info:
                bbox_info = retrieved_pages_bbox_info[page]
                bbox_score = bbox_info.get("score", 0.0)
                bbox_coords = bbox_info.get("bbox_coords", "")
                scored_pages.append({
                    "page": page,
                    "score": bbox_score,
                    "bbox_coords": bbox_coords
                })
                print(f"[DEBUG] page {page} 使用保存的bbox坐标: {bbox_coords}, 评分: {bbox_score:.2f}")
            else:
                print(f"[WARNING] page {page} 没有对应的bbox信息")
        
        # 按分数排序，只保留评分高于阈值的bbox
        scored_pages.sort(key=lambda x: x["score"], reverse=True)
        
        # 打印所有bbox的评分信息，便于调试
        print(f"[DEBUG] 所有bbox评分信息:")
        for item in scored_pages:
            print(f"  page={item['page']}, score={item['score']:.2f}, bbox_coords='{item['bbox_coords']}'")
        
        # 降低评分阈值，从500.0改为50.0
        high_score_pages = [{"page": item["page"], "bbox_coords": item["bbox_coords"]} for item in scored_pages if item["score"] >= min_score]
        
        # 如果没有高分bbox，则返回评分最高的bbox
        if not high_score_pages and scored_pages:
            print(f"[WARNING] 没有bbox达到最低分数阈值 {min_score}，返回评分最高的bbox")
            high_score_pages = [{"page": scored_pages[0]["page"], "bbox_coords": scored_pages[0]["bbox_coords"]}]
            print(f"[DEBUG] 返回评分最高的bbox: page={scored_pages[0]['page']}, score={scored_pages[0]['score']:.2f}, bbox_coords='{scored_pages[0]['bbox_coords']}'")
        
        # 限制最大页数
        if len(high_score_pages) > max_pages:
            print(f"Trimming references from {len(high_score_pages)} to {max_pages} pages")
            high_score_pages = high_score_pages[:max_pages]
        
        print(f"[DEBUG] 公司名: {company_name}, 文件名: {file_name}")
        print(f"[DEBUG] claimed_pages: {claimed_pages}")
        print(f"[DEBUG] retrieved_pages (bbox级别): {retrieved_pages}")
        
        # 打印每个bbox的详细信息
        print(f"\n[DEBUG] 所有bbox详细信息:")
        for page, bbox_info in retrieved_pages_bbox_info.items():
            bbox_coords = bbox_info.get("bbox_coords", "")
            bbox_score = bbox_info.get("score", 0.0)
            all_bboxes = bbox_info.get("all_bboxes", [])
            print(f"  页码 {page}:")
            print(f"    - 最相关bbox坐标: {bbox_coords}")
            print(f"    - 评分: {bbox_score:.2f}")
            print(f"    - 该chunk包含的所有bbox数量: {len(all_bboxes)}")
            if len(all_bboxes) > 0 and len(all_bboxes) <= 10:
                print(f"    - 所有bbox详情:")
                for idx, bbox in enumerate(all_bboxes):
                    bbox_page = bbox.get("page", "N/A")
                    bbox_coords_list = bbox.get("bbox", [])
                    bbox_text = bbox.get("text", "")
                    # 打印完整的bbox文本内容
                    if len(bbox_text) > 100:
                        bbox_text_preview = bbox_text[:100] + "..."
                    else:
                        bbox_text_preview = bbox_text
                    coords_str = f"{bbox_coords_list[0]},{bbox_coords_list[1]},{bbox_coords_list[2]},{bbox_coords_list[3]}" if len(bbox_coords_list) == 4 else "N/A"
                    print(f"      [{idx}] page={bbox_page}, coords={coords_str}")
                    print(f"          text='{bbox_text_preview}'")
        
        print(f"[DEBUG] scored_pages: {[(item['page'], item['score'], item['bbox_coords']) for item in scored_pages]}")
        print(f"[DEBUG] validated_pages (过滤后): {[item['page'] for item in high_score_pages]}")
        print(f"[DEBUG] bbox_coords: {[item['bbox_coords'] for item in high_score_pages]}")
        
        print(f"\n{'='*80}")
        print(f"[DEBUG] _validate_page_references 完成")
        print(f"[DEBUG] 输出: high_score_pages={high_score_pages}")
        print(f"{'='*80}\n")
        
        return high_score_pages

    def get_answer_for_company(self, company_name: str, question: str, schema: str) -> dict:
        # 针对单个公司，检索上下文并调用LLM生成答案
        query_start_time = PerformanceMonitor.start_query()
        t0 = time.time()
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        t1 = time.time()
        print(f"[计时] [get_answer_for_company] 检索器初始化耗时: {t1-t0:.2f} 秒")
        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:           
            t2 = time.time()
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
            t3 = time.time()
            print(f"[计时] [get_answer_for_company] 检索耗时: {t3-t2:.2f} 秒")
        if not retrieval_results:
            raise ValueError("No relevant context found")
        
        # --- Data Validation Start ---
        if self.enable_data_validation:
            print(f"\n{'='*80}")
            print(f"[Data Validation] Starting validation for {len(retrieval_results)} chunks")
            validated_results = []
            for result in retrieval_results:
                # Add company_name to source_meta for cross-check
                source_meta = {'company_name': company_name}
                if DataValidator.validate_chunk(result, source_meta):
                    validated_results.append(result)
                else:
                    print(f"[Data Validation] Filtered out chunk from {result.get('file_name')} page {result.get('page')}")
            
            print(f"[Data Validation] Retained {len(validated_results)}/{len(retrieval_results)} chunks")
            retrieval_results = validated_results
            
            # Generate cleaning report
            DataValidator.generate_cleaning_report()
            print(f"{'='*80}\n")
        # --- Data Validation End ---
        
        # 打印检索到的所有chunk内容
        print(f"\n{'='*80}")
        print(f"[检索结果] 共检索到 {len(retrieval_results)} 个结果")
        print(f"{'='*80}\n")
        for idx, result in enumerate(retrieval_results, 1):
            print(f"[Chunk {idx}]")
            print(f"公司名: {result.get('company_name', 'N/A')}")
            print(f"文件名: {result.get('file_name', 'N/A')}")
            print(f"页码: {result.get('page', 'N/A')}")
            print(f"距离: {result.get('distance', 'N/A')}")
            print(f"内容:\n{result.get('text', '')}")
            print(f"\n{'-'*80}\n")
        
        t4 = time.time()
        rag_context = self._format_retrieval_results(retrieval_results)
        t5 = time.time()
        print(f"[计时] [get_answer_for_company] 构建rag_context耗时: {t5-t4:.2f} 秒")
        
        print(f"\n{'='*80}")
        print(f"[DEBUG] 调用LLM生成答案")
        print(f"[DEBUG] 输入:")
        print(f"[DEBUG]   question={question}")
        print(f"[DEBUG]   rag_context长度={len(rag_context)}")
        print(f"[DEBUG]   schema={schema}")
        print(f"{'='*80}\n")
        
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        t6 = time.time()
        print(f"[计时] [get_answer_for_company] LLM调用耗时: {t6-t5:.2f} 秒")
        
        print(f"\n{'='*80}")
        print(f"[DEBUG] LLM返回结果")
        print(f"[DEBUG] answer_dict={answer_dict}")
        print(f"[DEBUG] relevant_pages={answer_dict.get('relevant_pages', [])}")
        print(f"{'='*80}\n")
        
        self.response_data = self.openai_processor.response_data
        if self.new_challenge_pipeline:
            # 从检索结果中提取公司名和文件名用于调试信息
            debug_company_name = "未知公司"
            debug_file_name = "未知文件"
            if retrieval_results:
                debug_company_name = retrieval_results[0].get('company_name', '未知公司')
                debug_file_name = retrieval_results[0].get('file_name', '未知文件')
            
            pages = answer_dict.get("relevant_pages", [])
            print(f"[DEBUG] 公司名: {debug_company_name}, 文件名: {debug_file_name}")
            print(f"[DEBUG] LLM返回的 relevant_pages: {pages}")
            validated_pages = self._validate_page_references(pages, retrieval_results, question)
            print(f"[DEBUG] 验证后的 validated_pages: {validated_pages}")
            answer_dict["relevant_pages"] = validated_pages
            answer_dict["references"] = self._extract_references(validated_pages, company_name, retrieval_results, question)
            # 保存检索结果，用于PDF高亮时找到最相关的chunk
            answer_dict["retrieval_results"] = retrieval_results
            print(f"[DEBUG] 最终 answer_dict['relevant_pages']: {answer_dict['relevant_pages']}")
        print(f"[计时] [get_answer_for_company] 总耗时: {t6-t0:.2f} 秒")
        PerformanceMonitor.end_query(query_start_time, f"Company: {company_name}")
        return answer_dict

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """从问题文本中提取公司名，匹配subset文件中的公司"""
        if not hasattr(self, 'companies_df'):
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                self.companies_df = pd.read_csv(self.subset_path, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                self.companies_df = pd.read_csv(self.subset_path, encoding='gbk')
        
        found_companies = []
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)
        
        for company in company_names:
            # 只要公司名在问题文本中出现就算匹配（包含关系）
            if company in question_text:
                found_companies.append(company)
                question_text = question_text.replace(company, '')
        
        return found_companies

    def process_question(self, question: str, schema: str):
        # 处理单个问题，支持多公司比较
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=schema)
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_companies, schema)
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """创建答案详情的引用ID，并存储详细内容"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """统计处理结果，包括总数、错误数、N/A数、成功数"""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, pipeline_details: str = "") -> dict:
        # 批量处理问题列表，支持并行与断点保存，返回处理结果和统计信息
        total_questions = len(questions_list)
        # 给每个问题加索引，便于后续答案详情定位
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # 预分配答案详情列表
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            # 单线程顺序处理
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
        else:
            # 多线程并行处理
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map 保证结果顺序与输入一致
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        
        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        处理问题处理过程中的异常。
        记录错误详情并返回包含错误信息的字典。
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        提交格式后处理：
        1. 页码从1-based转为0-based
        2. N/A答案清空引用
        3. 格式化为比赛提交schema
        4. 包含step_by_step_analysis
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                # 同时保留bbox_coords字段
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1,
                        "bbox_coords": ref.get("bbox_coords", "")
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """
        处理多公司比较类问题：
        1. 先将比较问题重写为单公司问题
        2. 并行处理每个公司
        3. 汇总结果并生成最终比较答案
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            
            answer_dict = self.get_answer_for_company(
                company_name=company, 
                question=sub_question, 
                schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer

    def process_single_question(self, question: str, kind: str = "string"):
        """
        单条问题推理，返回结构化答案。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] [单问] 开始公司名抽取 ...")
        # 公司名抽取
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        t1 = time.time()
        print(f"[计时] [单问] 公司名抽取耗时: {t1-t0:.2f} 秒")
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            print("[计时] [单问] 开始检索与LLM推理 ...")
            t2 = time.time()
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=kind)
            t3 = time.time()
            print(f"[计时] [单问] 检索+LLM推理耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict
        else:
            print("[计时] [单问] 开始多公司比较 ...")
            t2 = time.time()
            answer_dict = self.process_comparative_question(question, extracted_companies, kind)
            t3 = time.time()
            print(f"[计时] [单问] 多公司比较耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict

    def calculate_semantic_relevance_score(self, question: str, bbox_text: str) -> float:
        """
        使用大模型评估问题与bbox文本的语义相关性
        
        参数:
            question: 用户查询问题
            bbox_text: bbox的文本内容
        
        返回:
            float: 语义相关性分数，0-1之间，1表示完全相关
        """
        try:
            # 构建提示词
            system_prompt = """你是一个专业的财务文档分析助手。请评估用户查询问题与文档片段之间的语义相关性。

评估标准：
1. 指标名称一致性：如果文档片段包含用户查询的准确指标名称，相关性应该很高
2. 数据类型匹配：如果用户查询具体数值，而文档片段包含该数值，相关性应该很高
3. 语义相关性：即使指标名称不完全一致，如果语义上相关，也应该给予一定分数

请返回一个0到1之间的分数，保留2位小数。"""
            
            user_prompt = f"""用户查询问题：{question}

文档片段内容：
{bbox_text[:500]}

请评估这个文档片段与用户查询问题的语义相关性，返回一个0到1之间的分数。"""
            
            # 调用大模型
            response = self.openai_processor.processor.send_message(
                model="qwen-turbo-latest",
                temperature=0.1,
                system_content=system_prompt,
                human_content=user_prompt
            )
            
            # 解析响应，提取分数
            try:
                # 如果响应已经是数字，直接使用
                if isinstance(response, (int, float)):
                    semantic_score = float(response)
                    print(f"[DEBUG] 大模型语义相关性评分: {semantic_score}")
                    return semantic_score
                
                # 如果响应是字符串，尝试解析
                response_str = str(response).strip()
                if response_str.replace('.', '').isdigit():
                    semantic_score = float(response_str)
                    print(f"[DEBUG] 大模型语义相关性评分: {semantic_score}")
                    return semantic_score
                
                # 使用正则表达式提取分数
                import re
                score_match = re.search(r'0\.\d+|1\.0|0\.0', response_str)
                if score_match:
                    semantic_score = float(score_match.group())
                    print(f"[DEBUG] 大模型语义相关性评分: {semantic_score}")
                    return semantic_score
                else:
                    print(f"[DEBUG] 大模型响应无法解析分数: {response}")
                    return 0.5
            except Exception as parse_error:
                print(f"[DEBUG] 解析分数时出错: {parse_error}, 响应: {response}")
                return 0.5
        
        except Exception as e:
            print(f"[WARNING] 大模型语义评分失败: {e}")
            return 0.0
    