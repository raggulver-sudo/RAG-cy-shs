from pathlib import Path
from pyprojroot import here
from src.pipeline import Pipeline, RunConfig

if __name__ == "__main__":
    # 设置数据集根目录
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    
    # 初始化主流程
    pipeline = Pipeline(root_path)
    
    print('4. 将pdf转化为纯markdown文本')
    pipeline.export_all_reports_to_markdown(skip_existing=False)
    
    print('5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports()
