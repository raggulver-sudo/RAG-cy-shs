import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker
import hashlib
import pandas as pd
import time

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        # 初始化BM25检索器，指定BM25索引和文档目录
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        # 按公司名检索相关文本块，返回BM25分数最高的top_n个块
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
        # 加载对应的BM25索引，文件名用 sha1
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # 获取文档内容和BM25索引
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # 计算BM25分数
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_provider: str = "dashscope"):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.db_index = self._build_db_index()
        # 默认使用 dashscope 作为 embedding provider
        self.embedding_provider = embedding_provider.lower()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        # 根据 embedding_provider 初始化对应的 LLM 客户端
        load_dotenv()
        if self.embedding_provider == "openai":
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=None,
                max_retries=2
            )
            return llm
        elif self.embedding_provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return None  # dashscope 不需要 client 对象
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _get_embedding(self, text: str):
        # 根据 embedding_provider 获取文本的向量表示
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return embedding.data[0].embedding
        elif self.embedding_provider == "dashscope":
            import dashscope
            from http import HTTPStatus
            
            max_retries = 3
            retry_delay = 1
            last_exception = None

            for attempt in range(max_retries):
                try:
                    rsp = dashscope.TextEmbedding.call(
                        model="text-embedding-v1",
                        input=[text]
                    )
                    
                    if rsp.status_code == HTTPStatus.OK:
                        # 兼容 dashscope 返回格式，不能用 resp.output，需用 resp['output']
                        if 'output' in rsp and 'embeddings' in rsp['output']:
                            # 多条输入（本处只有一条）
                            emb = rsp['output']['embeddings'][0]
                            if emb['embedding'] is None or len(emb['embedding']) == 0:
                                raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
                            return emb['embedding']
                        elif 'output' in rsp and 'embedding' in rsp['output']:
                            # 兼容单条输入格式
                            if rsp['output']['embedding'] is None or len(rsp['output']['embedding']) == 0:
                                raise RuntimeError("DashScope返回的embedding为空")
                            return rsp['output']['embedding']
                        else:
                            raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")
                    else:
                        # If status code is not OK, treat as an error to retry or fail
                        error_msg = f"DashScope API failed with status {rsp.status_code}: {rsp.message}"
                        _log.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                        last_exception = RuntimeError(error_msg)
                        
                except Exception as e:
                    _log.warning(f"Attempt {attempt + 1}/{max_retries} failed with exception: {e}")
                    last_exception = e
                
                # Wait before retrying
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
            
            # If we exhausted retries
            raise last_exception or RuntimeError("Failed to get embeddings from DashScope after retries")
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    @staticmethod
    def set_up_llm():
        # 静态方法，初始化OpenAI LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def _build_db_index(self):
        # 建立轻量级索引，只加载元数据，不加载 heavy 的向量库
        db_index = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        for document_path in all_documents_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            sha1 = document.get('metainfo', {}).get('sha1', None)
            company_name = document.get('metainfo', {}).get('company_name', None)
            file_name = document.get('metainfo', {}).get('file_name', None)
            
            if not sha1:
                _log.warning(f"No sha1 found in metainfo for document {document_path.name}")
                continue
            
            faiss_path = self.vector_db_dir / f"{sha1}.faiss"
            if not faiss_path.exists():
                _log.warning(f"No matching vector DB found for document {document_path.name} (sha1={sha1})")
                continue
                
            entry = {
                "sha1": sha1,
                "company_name": company_name,
                "file_name": file_name,
                "json_path": document_path,
                "faiss_path": faiss_path
            }
            db_index.append(entry)
        return db_index

    def _load_specific_db(self, entry):
        # 按需加载特定的向量库和文档
        sha1 = entry["sha1"]
        faiss_path = entry["faiss_path"]
        json_path = entry["json_path"]
        
        try:
            # 切换到向量库目录以避免中文路径问题
            original_cwd = os.getcwd()
            os.chdir(str(self.vector_db_dir))
            try:
                vector_db = faiss.read_index(f"{sha1}.faiss")
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            raise RuntimeError(f"Error reading vector DB for {sha1}: {e}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            document = json.load(f)
            
        return vector_db, document

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 计算两个字符串的余弦相似度（通过嵌入）
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        # 按公司名检索相关文本块，返回向量距离最近的top_n个块
        # 如果提供了 llm_reranking_sample_size，则使用该值作为检索数量，否则使用 top_n
        actual_top_n = llm_reranking_sample_size if llm_reranking_sample_size is not None else top_n
        
        # 查找匹配的文档条目 (Metadata Search)
        target_entry = None
        for entry in self.db_index:
            if entry.get("company_name") == company_name:
                target_entry = entry
                break
            elif company_name in entry.get("file_name", ""):
                target_entry = entry
                break
                
        if target_entry is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        # On-demand load
        vector_db, document = self._load_specific_db(target_entry)
        
        chunks = document["content"]["chunks"]
        pages = document["content"].get("pages", [])
        file_name = document["metainfo"].get("file_name", "未知文件")
        company_name = document["metainfo"].get("company_name", "未知公司")
        actual_top_n = min(actual_top_n, len(chunks))
        # 获取 query 的 embedding，支持 openai/dashscope
        embedding = self._get_embedding(query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
        retrieval_results = []
        seen_pages = set()
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = None
            if pages:
                parent_page = next((page for page in pages if page["page"] == chunk.get("page")), None)
            if return_parent_pages and parent_page:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"],
                        "file_name": file_name,
                        "company_name": company_name
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk.get("page", 0),
                    "text": chunk["text"],
                    "file_name": file_name,
                    "company_name": company_name
                }
                retrieval_results.append(result)
        # 如果启用了父文档检索，且检索结果数量不足 top_n，则补充更多结果
        if return_parent_pages and len(retrieval_results) < top_n and actual_top_n < len(chunks):
            additional_top_n = min(top_n - len(retrieval_results), len(chunks) - actual_top_n)
            if additional_top_n > 0:
                additional_k = actual_top_n + additional_top_n
                additional_distances, additional_indices = vector_db.search(x=embedding_array, k=additional_k)
                for distance, index in zip(additional_distances[0][actual_top_n:], additional_indices[0][actual_top_n:]):
                    distance = round(float(distance), 4)
                    chunk = chunks[index]
                    parent_page = None
                    if pages:
                        parent_page = next((page for page in pages if page["page"] == chunk.get("page")), None)
                    if parent_page and parent_page["page"] not in seen_pages:
                        seen_pages.add(parent_page["page"])
                        result = {
                            "distance": distance,
                            "page": parent_page["page"],
                            "text": parent_page["text"],
                            "file_name": file_name,
                            "company_name": company_name
                        }
                        retrieval_results.append(result)
                        if len(retrieval_results) >= top_n:
                            break
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        # 检索公司所有文本块，返回全部内容
        target_entry = None
        for entry in self.db_index:
            if entry.get("company_name") == company_name:
                target_entry = entry
                break
        
        if target_entry is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        # On-demand load
        _, document = self._load_specific_db(target_entry)
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        使用混合检索方法进行检索和重排。
        
        参数：
            company_name: 需要检索的公司名称
            query: 检索查询语句
            llm_reranking_sample_size: 首轮向量检索返回的候选数量
            documents_batch_size: 每次送入LLM重排的文档数
            top_n: 最终返回的重排结果数量
            llm_weight: LLM分数权重（0-1）
            return_parent_pages: 是否返回完整页面（而非分块）
        
        返回：
            经过重排的文档字典列表，包含分数
        """
        t0 = time.time()
        # 首先用向量检索器获取初步结果
        print("[计时] [HybridRetriever] 开始向量检索 ...")
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        t1 = time.time()
        print(f"[计时] [HybridRetriever] 向量检索耗时: {t1-t0:.2f} 秒")
        # 使用LLM对结果进行重排
        print("[计时] [HybridRetriever] 开始LLM重排 ...")
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        t2 = time.time()
        print(f"[计时] [HybridRetriever] LLM重排耗时: {t2-t1:.2f} 秒")
        print(f"[计时] [HybridRetriever] 总耗时: {t2-t0:.2f} 秒")
        return reranked_results[:top_n]
