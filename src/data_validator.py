import logging
from typing import Dict, List, Any, Optional, Union
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configure logging for validation
validation_logger = logging.getLogger("data_validator")
validation_logger.setLevel(logging.INFO)
handler = logging.FileHandler("data_validation.log", encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
validation_logger.addHandler(handler)

class DataValidator:
    """
    Validator for retrieval results and chunk data.
    Ensures data integrity before presentation.
    """
    _system_data = None
    _subset_path = Path("data/stock_data/subset.csv")

    @classmethod
    def set_subset_path(cls, path: Union[str, Path]):
        """Sets the path to the system data (subset.csv)."""
        cls._subset_path = Path(path)
        cls._system_data = None # Reset to reload from new path

    @classmethod
    def _load_system_data(cls):
        if cls._system_data is None:
            try:
                if cls._subset_path.exists():
                    cls._system_data = pd.read_csv(cls._subset_path)
                else:
                    validation_logger.warning(f"{cls._subset_path} not found, skipping system data validation")
                    cls._system_data = pd.DataFrame()
            except Exception as e:
                validation_logger.error(f"Failed to load system data from {cls._subset_path}: {e}")
                cls._system_data = pd.DataFrame()
        return cls._system_data
    
    @classmethod
    def validate_chunk(cls, chunk: Dict[str, Any], source_meta: Optional[Dict] = None) -> bool:
        """
        Validates a single chunk of data.
        
        Args:
            chunk: The chunk dictionary containing text, page, bbox, etc.
            source_meta: Optional metadata from the source document for cross-reference.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # 1. Field Integrity Check
            required_fields = ['text', 'page', 'file_name']
            for field in required_fields:
                if field not in chunk:
                    validation_logger.error(f"Missing required field: {field} in chunk: {str(chunk)[:100]}...")
                    return False
            
            # 2. Data Format Verification
            if not isinstance(chunk['page'], (int, str)): # Allow str for now if parsed that way, but preferably int
                 validation_logger.error(f"Invalid page format: {chunk['page']} (type: {type(chunk['page'])})")
                 return False
                 
            if not chunk['text'] or not isinstance(chunk['text'], str) or len(chunk['text'].strip()) == 0:
                validation_logger.error(f"Empty or invalid text content in chunk from {chunk.get('file_name')}")
                return False

            # 3. Business Logic Consistency (if source_meta provided)
            if source_meta:
                # Validate File Name if present in both
                if 'file_name' in source_meta and chunk.get('file_name'):
                     src_fname = source_meta.get('file_name')
                     chk_fname = chunk.get('file_name')
                     if src_fname and chk_fname:
                        if chk_fname != src_fname:
                             # Partial match is acceptable due to extensions
                             if chk_fname not in src_fname and src_fname not in chk_fname:
                                validation_logger.warning(f"Filename mismatch: Chunk says {chk_fname}, Source says {src_fname}")

                # Validate Company Name if present in both
                if 'company_name' in source_meta and chunk.get('company_name'):
                    src_company = source_meta.get('company_name')
                    chk_company = chunk.get('company_name')
                    if src_company and chk_company and src_company != chk_company:
                         validation_logger.error(f"Company mismatch: Chunk says {chk_company}, Source says {src_company}")
                         return False

            # 4. System Data Cross-check (subset.csv)
            sys_data = cls._load_system_data()
            if not sys_data.empty:
                 fname = chunk.get('file_name', '')
                 # Match by filename stem to avoid extension issues
                 fname_stem = Path(fname).stem
                 matched = sys_data[sys_data['file_name'].apply(lambda x: Path(str(x)).stem == fname_stem)]
                 
                 if matched.empty:
                     validation_logger.warning(f"Chunk file {fname} not found in system data (subset.csv)")
                     # We tag/filter it
                     return False
                 
                 # Check company consistency if available
                 row = matched.iloc[0]
                 sys_company = row.get('company_name')
                 chunk_company = source_meta.get('company_name') if source_meta else chunk.get('company_name')
                 
                 if chunk_company and sys_company and sys_company != chunk_company:
                     validation_logger.error(f"Company mismatch: System says {sys_company}, Chunk says {chunk_company}")
                     return False

            # 5. BBox Validation (if present)
            if 'bbox' in chunk:
                bbox = chunk['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                     validation_logger.warning(f"Invalid bbox format: {bbox}")
                else:
                    # Check coordinates validity (x0, y0, x1, y1)
                    if not (bbox[0] <= bbox[2] and bbox[1] <= bbox[3]):
                         validation_logger.warning(f"Invalid bbox coordinates (x0>x1 or y0>y1): {bbox}")

            return True
            
        except Exception as e:
            validation_logger.error(f"Exception during chunk validation: {e}")
            return False

    @classmethod
    def filter_invalid_chunks(cls, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters a list of chunks, keeping only valid ones.
        Logs dropped chunks.
        """
        valid_chunks = []
        for chunk in chunks:
            if cls.validate_chunk(chunk):
                valid_chunks.append(chunk)
            else:
                validation_logger.info(f"Dropped invalid chunk from {chunk.get('file_name', 'unknown')}")
        
        return valid_chunks

    @classmethod
    def generate_cleaning_report(cls, report_path: str = "validation_report.txt"):
        """
        Generates a summary report of validation issues (Data Cleaning Report).
        """
        try:
            with open("data_validation.log", "r", encoding="utf-8") as f:
                logs = f.readlines()
            
            error_count = sum(1 for line in logs if "ERROR" in line)
            warning_count = sum(1 for line in logs if "WARNING" in line)
            
            report = f"""
            === Data Cleaning Report ===
            Generated at: {datetime.now()}
            
            Total Errors (Filtered): {error_count}
            Total Warnings (Flagged): {warning_count}
            
            Recent Issues:
            """
            # Add last 20 lines for better context
            report += "".join(logs[-20:])
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
                
        except FileNotFoundError:
            pass
