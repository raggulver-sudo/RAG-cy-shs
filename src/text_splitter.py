import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os

# 文本分块工具类，支持按页分块、表格插入、token统计等
class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """按页分组已序列化表格，便于后续插入到对应页面分块中"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """将报告按页分块，保留markdown表格内容，可选插入序列化表格块。"""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            # 加载序列化表格，按页分组
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            # 普通文本分块
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            # 插入序列化表格分块
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        # 按页码排序，使内容顺序与PDF文档一致
        chunks.sort(key=lambda x: x.get('page', float('inf')))
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        # 统计字符串的token数，支持自定义编码
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """将单页文本分块，保留原始markdown表格。"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    #对 json 文件分块，输出还是 json
    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """
        批量处理目录下所有报告（json文件），对每个报告进行文本分块，并输出到目标目录。
        如果提供了序列化表格目录，会尝试将表格内容插入到对应页面的分块中。
        主要用于后续向量化和检索的预处理。
        参数：
            all_report_dir: 存放待处理报告json的目录
            output_dir: 分块后输出的目标目录
            serialized_tables_dir: （可选）存放序列化表格的目录
        """
        # 获取所有报告文件路径
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        # 遍历每个报告文件
        for report_path in all_report_paths:
            serialized_tables_path = None
            # 如果提供了表格序列化目录，查找对应表格文件
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"警告：未找到 {report_path.name} 的序列化表格报告")
                
            # 读取报告内容
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            # 分块处理，插入表格分块（如有）
            updated_report = self._split_report(report_data, serialized_tables_path)
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入分块后的报告到目标目录
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        # 输出处理文件数统计
        print(f"已分块处理 {len(all_report_paths)} 个文件")

    def split_markdown_file(self, md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5):
        """
        按行分割 markdown 文件，每个分块记录起止行号和内容。
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        chunks = []
        i = 0
        total_lines = len(lines)
        while i < total_lines:
            start = i
            end = min(i + chunk_size, total_lines)
            chunk_text = ''.join(lines[start:end])
            chunks.append({
                'lines': [start + 1, end],  # 行号从1开始
                'text': chunk_text
            })
            i += chunk_size - chunk_overlap
        return chunks

    def _load_content_list_json(self, md_path: Path) -> Optional[List[Dict]]:
        """
        加载对应的 content_list.json 文件，包含页码信息
        :param md_path: markdown 文件路径
        :return: content_list.json 的内容列表，如果文件不存在则返回 None
        """
        content_list_path = md_path.parent / f"{md_path.stem}_content_list.json"
        if not content_list_path.exists():
            return None
        
        try:
            with open(content_list_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告：无法读取 {content_list_path}: {str(e)}")
            return None

    def _build_line_to_page_mapping(self, content_list: List[Dict], md_lines: List[str]) -> Dict[int, int]:
        """
        根据 content_list.json 构建行号到页码的映射
        :param content_list: content_list.json 的内容
        :param md_lines: markdown 文件的行列表
        :return: 行号到页码的映射字典（行号从1开始）
        """
        line_to_page = {}
        
        # 遍历 content_list 中的每个元素
        for item in content_list:
            if item.get('type') != 'text':
                continue
            
            text = item.get('text', '').strip()
            if not text:
                continue
            
            page_idx = item.get('page_idx', 0)
            
            # 在 markdown 文件中查找该文本所在的行
            for line_num, line in enumerate(md_lines, 1):
                if text in line:
                    # 如果该行还没有映射，或者找到了更精确的匹配
                    if line_num not in line_to_page or len(text) > len(md_lines[line_num - 1]):
                        line_to_page[line_num] = page_idx
        
        # 如果没有找到任何映射，返回空字典
        if not line_to_page:
            return {}
        
        # 填充缺失的行号映射（使用前一个已知页码）
        sorted_lines = sorted(line_to_page.keys())
        for line_num in range(1, len(md_lines) + 1):
            if line_num not in line_to_page:
                # 找到小于当前行号的最大行号
                prev_line = max([l for l in sorted_lines if l < line_num], default=1)
                line_to_page[line_num] = line_to_page.get(prev_line, 0)
        
        return line_to_page

    def split_markdown_file_with_page_numbers(self, md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5):
        """
        按行分割 markdown 文件，每个分块记录起止行号、页码和内容。
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :return: 分块列表，包含页码信息
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 尝试加载 content_list.json
        content_list = self._load_content_list_json(md_path)
        line_to_page = {}
        
        if content_list:
            line_to_page = self._build_line_to_page_mapping(content_list, lines)
            if line_to_page:
                print(f"成功加载页码映射，共 {len(line_to_page)} 行")
            else:
                print("警告：无法构建页码映射，将不包含页码信息")
        else:
            print(f"未找到对应的 content_list.json 文件，将不包含页码信息")
        
        chunks = []
        i = 0
        total_lines = len(lines)
        while i < total_lines:
            start = i
            end = min(i + chunk_size, total_lines)
            chunk_text = ''.join(lines[start:end])
            
            # 确定该分块的页码（使用起始行的页码）
            start_line = start + 1
            page_number = line_to_page.get(start_line, None)
            
            chunk_data = {
                'lines': [start_line, end],
                'text': chunk_text
            }
            
            # 如果有页码信息，添加到分块中
            if page_number is not None:
                chunk_data['page'] = page_number
            
            chunks.append(chunk_data)
            i += chunk_size - chunk_overlap
        
        # 按页码排序，使内容顺序与PDF文档一致
        chunks.sort(key=lambda x: x.get('page', float('inf')))
        
        return chunks

    def split_content_list_file(self, content_list_path: Path, chunk_size: int = 30, chunk_overlap: int = 5):
        """
        直接使用 content_list.json 文件进行分块，包含文本、表格、图片等所有内容
        :param content_list_path: content_list.json 文件路径
        :param chunk_size: 每个分块的最大内容数量
        :param chunk_overlap: 分块重叠数量
        :return: 分块列表，包含页码信息
        """
        # 加载 content_list.json
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        
        # 按页码排序
        content_list.sort(key=lambda x: x.get('page_idx', float('inf')))
        
        chunks = []
        chunk_id = 0
        
        i = 0
        total_items = len(content_list)
        
        while i < total_items:
            start = i
            end = min(i + chunk_size, total_items)
            items = content_list[start:end]
            
            # 收集该分块中的所有内容
            chunk_text_parts = []
            page_numbers = set()
            bboxes = []  # 收集所有bbox信息
            
            for item in items:
                page_idx = item.get('page_idx', 0)
                page_numbers.add(page_idx)
                
                item_type = item.get('type', '')
                
                if item_type == 'text':
                    # 文本内容
                    text = item.get('text', '')
                    if text:
                        chunk_text_parts.append(text)
                        # 记录bbox信息
                        bbox = item.get('bbox', [])
                        if bbox:
                            bboxes.append({
                                'page': page_idx,
                                'bbox': bbox,
                                'type': 'text',
                                'text': text
                            })
                elif item_type == 'table':
                    # 表格内容
                    table_body = item.get('table_body', '')
                    if table_body:
                        chunk_text_parts.append(f"[表格]\n{table_body}\n[/表格]")
                        # 记录bbox信息
                        bbox = item.get('bbox', [])
                        if bbox:
                            bboxes.append({
                                'page': page_idx,
                                'bbox': bbox,
                                'type': 'table',
                                'text': f"[表格]\n{table_body}\n[/表格]"
                            })
                elif item_type == 'image':
                    # 图片内容
                    image_caption = item.get('image_caption', [])
                    if image_caption:
                        caption_text = ' '.join(image_caption)
                        chunk_text_parts.append(f"[图片说明: {caption_text}]")
                        # 记录bbox信息
                        bbox = item.get('bbox', [])
                        if bbox:
                            bboxes.append({
                                'page': page_idx,
                                'bbox': bbox,
                                'type': 'image',
                                'text': f"[图片说明: {caption_text}]"
                            })
            
            # 合并文本
            chunk_text = '\n'.join(chunk_text_parts)
            
            # 确定该分块的页码（使用最小的页码）
            page_number = min(page_numbers) if page_numbers else None
            
            chunk_data = {
                'id': chunk_id,
                'text': chunk_text
            }
            
            # 添加页码信息
            if page_number is not None:
                chunk_data['page'] = page_number
            
            # 添加bbox信息
            if bboxes:
                chunk_data['bboxes'] = bboxes
            
            chunks.append(chunk_data)
            chunk_id += 1
            i += chunk_size - chunk_overlap
        
        return chunks

    def split_markdown_reports(self, all_md_dir: Path, output_dir: Path, chunk_size: int = 30, chunk_overlap: int = 5, subset_csv: Path = None, use_page_numbers: bool = True, use_content_list: bool = False):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。
        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        :param use_page_numbers: 是否使用页码信息（默认为True）
        :param use_content_list: 是否使用content_list.json进行分块（包含文本、表格、图片）
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if use_content_list:
            # 使用 content_list.json 进行分块
            all_content_list_paths = list(all_md_dir.glob("*_content_list.json"))
            for content_list_path in all_content_list_paths:
                chunks = self.split_content_list_file(content_list_path, chunk_size, chunk_overlap)
                
                # 生成输出文件名（去掉 _content_list 后缀）
                output_json_name = content_list_path.stem.replace('_content_list', '') + ".json"
                output_json_path = output_dir / output_json_name
                
                # 查找 company_name 和 sha1
                file_no_ext = content_list_path.stem.replace('_content_list', '')
                company_name = file2company.get(file_no_ext, "")
                sha1 = file2sha1.get(file_no_ext, "")
                # metainfo 只保留 sha1、company_name、file_name 字段
                metainfo = {"sha1": sha1, "company_name": company_name, "file_name": output_json_name}
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
                print(f"已处理: {content_list_path.name} -> {output_json_path.name}")
            print(f"共分割 {len(all_content_list_paths)} 个 content_list.json 文件")
        else:
            # 使用 markdown 文件进行分块
            all_md_paths = list(all_md_dir.glob("*.md"))
            for md_path in all_md_paths:
                # 根据参数选择使用带页码还是不带页码的分块方法
                if use_page_numbers:
                    chunks = self.split_markdown_file_with_page_numbers(md_path, chunk_size, chunk_overlap)
                else:
                    chunks = self.split_markdown_file(md_path, chunk_size, chunk_overlap)
                
                output_json_path = output_dir / (md_path.stem + ".json")
                # 查找 company_name 和 sha1
                file_no_ext = md_path.stem
                company_name = file2company.get(file_no_ext, "")
                sha1 = file2sha1.get(file_no_ext, "")
                # metainfo 只保留 sha1、company_name、file_name 字段
                metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
                print(f"已处理: {md_path.name} -> {output_json_path.name}")
            print(f"共分割 {len(all_md_paths)} 个 markdown 文件")
