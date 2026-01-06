import time
import psutil
import os
from datetime import datetime
from typing import Dict, Any, List

class PerformanceMonitor:
    _instance = None
    _metrics = {
        "startup_time": 0,
        "queries": []
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerformanceMonitor, cls).__new__(cls)
            cls._instance.start_time = time.time()
        return cls._instance

    @classmethod
    def record_startup_time(cls):
        """Record the time taken for the system to start up."""
        if cls._instance:
            cls._metrics["startup_time"] = time.time() - cls._instance.start_time

    @classmethod
    def start_query(cls) -> float:
        """Start tracking a query. Returns start timestamp."""
        return time.time()

    @classmethod
    def end_query(cls, start_time: float, query_info: str = ""):
        """End tracking a query and record metrics."""
        duration = time.time() - start_time
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        cls._metrics["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "memory_mb": memory_usage,
            "info": query_info
        })

    @classmethod
    def generate_report(cls, report_path: str = "performance_report.txt"):
        """Generate a performance report."""
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=== Performance Optimization Report ===\n")
                f.write(f"Generated at: {datetime.now()}\n\n")
                
                f.write(f"Startup Time: {cls._metrics.get('startup_time', 0):.4f} seconds\n")
                f.write(f"Total Queries Processed: {len(cls._metrics['queries'])}\n\n")
                
                if cls._metrics["queries"]:
                    total_duration = sum(q['duration'] for q in cls._metrics['queries'])
                    avg_duration = total_duration / len(cls._metrics['queries'])
                    f.write(f"Average Query Latency: {avg_duration:.4f} seconds\n")
                    
                    max_memory = max(q['memory_mb'] for q in cls._metrics['queries'])
                    f.write(f"Peak Memory Usage: {max_memory:.2f} MB\n\n")
                    
                    f.write("--- Detailed Query Log ---\n")
                    for q in cls._metrics["queries"][-20:]: # Last 20 queries
                        f.write(f"[{q['timestamp']}] Duration: {q['duration']:.4f}s | Memory: {q['memory_mb']:.2f} MB | Info: {q['info']}\n")
                else:
                    f.write("No queries processed yet.\n")
                    
        except Exception as e:
            print(f"Failed to generate performance report: {e}")
