# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import shutil
import time
import hashlib

from src.questions_processing import QuestionsProcessor

# Lazy imports for heavy modules to optimize startup time
# from src.pdf_parsing import PDFParser
# from src import pdf_mineru
# from src.parsed_reports_merging import PageTextPreparation
# from src.text_splitter import TextSplitter
# from src.ingestion import VectorDBIngestor
# from src.ingestion import BM25Ingestor
# from src.tables_serialization import TableSerializer

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "1_pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "3_mineru_json"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # self.parsed_reports_dirname = "01_parsed_reports"
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        # self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        #self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        #self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        #self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 10 # 增加并行请求数
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "dashscope" #openai
    answering_model: str = "qwen-turbo-latest" # gpt-4o-mini-2024-07-18 or "gpt-4o-2024-08-06"
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "1_pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()
        self._convert_excel_to_csv_if_needed()
        
        # 初始化 QuestionsProcessor 实例，避免每次回答问题时重新初始化（包含耗时的glob操作）
        self.processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问模式初始无需文件，批处理时会覆盖
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    def _calculate_file_sha1(self, file_path: Path) -> str:
        """
        计算文件的SHA1哈希值
        :param file_path: 文件路径
        :return: SHA1哈希字符串
        """
        sha1_hash = hashlib.sha1()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha1_hash.update(chunk)
            return sha1_hash.hexdigest()
        except Exception as e:
            print(f"Warning: Failed to calculate SHA1 for {file_path}: {str(e)}")
            return ""

    def _convert_excel_to_csv_if_needed(self):
        """
        检查是否存在pdf_reports.xlsx。
        如果subset.csv不存在，或者pdf_reports.xlsx比subset.csv新，则自动转换为CSV。
        从Excel文件读取报告信息，计算SHA1，生成subset.csv
        """
        excel_path = self.paths.root_path / "pdf_reports.xlsx"
        csv_path = self.paths.root_path / "subset.csv"
        
        if not excel_path.exists():
            return

        should_convert = False
        if not csv_path.exists():
            should_convert = True
        else:
            # Check modification times
            if excel_path.stat().st_mtime > csv_path.stat().st_mtime:
                print(f"检测到 {excel_path.name} 已更新，准备重新生成 {csv_path.name}...")
                should_convert = True
        
        if should_convert:
            try:
                print(f"正在从 {excel_path.name} 生成 subset.csv...")
                
                df = pd.read_excel(excel_path)
                
                subset_data = []
                sha1_counter = 10001
                
                for _, row in df.iterrows():
                    stock_name = row.get('stock_name', '')
                    file_name = row.get('file_name', '')
                    
                    if not file_name:
                        continue
                    
                    pdf_path = self.paths.pdf_reports_dir / file_name
                    
                    if pdf_path.exists():
                        sha1_value = self._calculate_file_sha1(pdf_path)
                    else:
                        print(f"Warning: PDF file not found: {pdf_path}")
                        sha1_value = f"stock_{sha1_counter}"
                        sha1_counter += 1
                    
                    subset_data.append({
                        'sha1': sha1_value,
                        'file_name': file_name,
                        'company_name': stock_name
                    })
                
                subset_df = pd.DataFrame(subset_data)
                subset_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                print(f"成功更新 {csv_path.name}，共 {len(subset_df)} 条记录")
                
            except Exception as e:
                print(f"Error converting Excel to CSV: {str(e)}")

    @staticmethod
    def download_docling_models(): 
        # 下载Docling所需模型，避免首次运行时自动下载
        from src.pdf_parsing import PDFParser
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        from src.pdf_parsing import PDFParser
        logging.basicConfig(level=logging.DEBUG)
        
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def export_reports_to_markdown(self, file_name):
        """
        使用 pdf_mineru.py，将指定 PDF 文件转换为 markdown，并放到 reports_markdown_dirname 目录下。
        同时保存 content_list.json 文件，包含页码信息。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        # 调用 pdf_mineru 获取 task_id 并下载、解压
        print(f"开始处理: {file_name}")
        task_id = pdf_mineru.get_task_id(file_name)
        print(f"task_id: {task_id}")
        pdf_mineru.get_result(task_id)

        # 解压后目录名与 task_id 相同
        extract_dir = f"{task_id}"
        md_path = os.path.join(extract_dir, "full.md")
        if not os.path.exists(md_path):
            print(f"未找到 markdown 文件: {md_path}")
            return
        
        # 查找 content_list.json 文件
        content_list_json_path = None
        for file in os.listdir(extract_dir):
            if file.endswith("_content_list.json"):
                content_list_json_path = os.path.join(extract_dir, file)
                break
        
        # 目标目录
        os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")
        shutil.move(md_path, target_path)
        print(f"已将 {md_path} 移动到 {target_path}")
        
        # 如果找到 content_list.json，也移动到目标目录
        if content_list_json_path and os.path.exists(content_list_json_path):
            target_json_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}_content_list.json")
            shutil.move(content_list_json_path, target_json_path)
            print(f"已将 {content_list_json_path} 移动到 {target_json_path}")

    def export_all_reports_to_markdown(self, skip_existing: bool = True):
        """
        批量将所有 PDF 文件转换为 markdown
        :param skip_existing: 是否跳过已存在的 markdown 文件，默认为 True
        """
        print("=" * 60)
        print("开始批量转换 PDF 文件为 Markdown")
        print("=" * 60)
        
        try:
            df = pd.read_csv(self.paths.subset_path, encoding='utf-8-sig')
        except Exception as e:
            print(f"读取 subset.csv 失败: {str(e)}")
            return
        
        total_files = len(df)
        print(f"共 {total_files} 个文件需要处理")
        
        success_count = 0
        skipped_count = 0
        failed_count = 0
        failed_files = []
        
        for idx, row in df.iterrows():
            file_name = row.get('file_name', '')
            company_name = row.get('company_name', '')
            
            if not file_name:
                print(f"警告: 第 {idx+1} 行缺少 file_name，跳过")
                failed_count += 1
                continue
            
            base_name = os.path.splitext(file_name)[0]
            md_filename = f"{base_name}.md"
            md_path = os.path.join(self.paths.reports_markdown_path, md_filename)
            
            if skip_existing and os.path.exists(md_path):
                print(f"[{idx+1}/{total_files}] 跳过已存在: {file_name}")
                skipped_count += 1
                continue
            
            print(f"\n[{idx+1}/{total_files}] 处理中: {file_name} ({company_name})")
            print("-" * 60)
            
            try:
                self.export_reports_to_markdown(file_name)
                success_count += 1
                print(f"✓ 成功转换: {file_name}")
            except Exception as e:
                failed_count += 1
                failed_files.append(file_name)
                print(f"✗ 转换失败: {file_name}")
                print(f"  错误信息: {str(e)}")
        
        print("\n" + "=" * 60)
        print("批量转换完成")
        print("=" * 60)
        print(f"总计: {total_files} 个文件")
        print(f"成功: {success_count} 个")
        print(f"跳过: {skipped_count} 个")
        print(f"失败: {failed_count} 个")
        
        if failed_files:
            print(f"\n失败的文件列表:")
            for file_name in failed_files:
                print(f"  - {file_name}")
        
        print("=" * 60)

    def chunk_reports(self, include_serialized_tables: bool = False, use_content_list: bool = True):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        :param include_serialized_tables: 是否包含序列化表格（已废弃，保留用于兼容性）
        :param use_content_list: 是否使用content_list.json进行分块（默认True，包含表格和图片）
        """
        from src.text_splitter import TextSplitter
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的文件...")
        if use_content_list:
            print("使用 content_list.json 进行分块（包含文本、表格、图片）")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path,
            use_content_list=use_content_list
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 解析PDF报告，支持并行处理
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string"):
        """
        单条问题即时推理，返回结构化答案（dict）。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] 开始调用 answer_single_question ...")
        
        # 使用预初始化的 processor
        # 如果配置有变动（如llm_reranking），这里可能需要更新processor配置，但通常RunConfig是固定的
        # 为了安全起见，我们确保processor的parallel_requests为1（单问不需要并发）
        # 但其实QuestionsProcessor.process_single_question内部并不使用parallel_requests
        
        t1 = time.time()
        # print(f"[计时] QuestionsProcessor 获取耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始调用 process_single_question ...")
        
        # 调用处理方法
        answer = self.processor.process_single_question(question, kind=kind)
        
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2-t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2-t0:.2f} 秒")
        return answer

preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

base_config = RunConfig(
    parallel_requests=10,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

## 这里
max_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=10,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + NO reranking + SO CoT; llm = qwen-turbo",
    answering_model="qwen-turbo-latest",
    config_suffix="_qwen_turbo"
)


configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
# 你也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    #print(type(root_path))
    # 初始化主流程，使用推荐的最佳配置
    pipeline = Pipeline(root_path, run_config=max_config)
    
    print('4. 将pdf转化为纯markdown文本')
    pipeline.export_all_reports_to_markdown(skip_existing=True)

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    print('5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports() 
    
    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('6. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    pipeline.create_vector_dbs()     
    
    # 7. 处理问题并生成答案，具体逻辑取决于 run_config
    # 默认questions.json
    print('7. 处理问题并生成答案，具体逻辑取决于 run_config')
    pipeline.process_questions() 
    
    print('完成')
    print('完成')
