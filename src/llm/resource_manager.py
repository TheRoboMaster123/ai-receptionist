import psutil
import torch
import logging
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import threading
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class ResourceMetrics:
    def __init__(self):
        self.gpu_metrics = deque(maxlen=100)  # Store last 100 measurements
        self.cpu_metrics = deque(maxlen=100)
        self.memory_metrics = deque(maxlen=100)
        self.request_metrics = deque(maxlen=1000)  # Store last 1000 requests
        self.last_update = None
    
    def add_metric(self, gpu_usage: float, cpu_usage: float, memory_usage: float):
        """Add a new resource usage measurement."""
        timestamp = datetime.utcnow()
        self.gpu_metrics.append((timestamp, gpu_usage))
        self.cpu_metrics.append((timestamp, cpu_usage))
        self.memory_metrics.append((timestamp, memory_usage))
        self.last_update = timestamp
    
    def add_request_metric(self, duration: float, gpu_memory: float):
        """Add a new request processing measurement."""
        self.request_metrics.append({
            'timestamp': datetime.utcnow(),
            'duration': duration,
            'gpu_memory': gpu_memory
        })
    
    def get_average_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """Get average metrics for the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        # Filter metrics within time range
        recent_gpu = [(t, v) for t, v in self.gpu_metrics if t > cutoff]
        recent_cpu = [(t, v) for t, v in self.cpu_metrics if t > cutoff]
        recent_memory = [(t, v) for t, v in self.memory_metrics if t > cutoff]
        
        return {
            'gpu_usage': np.mean([v for _, v in recent_gpu]) if recent_gpu else 0,
            'cpu_usage': np.mean([v for _, v in recent_cpu]) if recent_cpu else 0,
            'memory_usage': np.mean([v for _, v in recent_memory]) if recent_memory else 0
        }

class ResourceManager:
    def __init__(self, gpu_memory_threshold: float = 0.9, cpu_threshold: float = 0.8):
        self.gpu_memory_threshold = gpu_memory_threshold
        self.cpu_threshold = cpu_threshold
        self.metrics = ResourceMetrics()
        self._monitor_thread = None
        self._stop_monitoring = False
        
        # Load balancing settings
        self.active_requests = 0
        self.max_concurrent_requests = self._calculate_max_concurrent()
    
    def _calculate_max_concurrent(self) -> int:
        """Calculate maximum concurrent requests based on available resources."""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Estimate based on GPU memory (assuming each request needs about 2GB)
            return max(1, int(gpu_mem / (2 * 1024 * 1024 * 1024)))
        else:
            # CPU-based calculation
            cpu_count = psutil.cpu_count()
            return max(1, cpu_count - 1)
    
    def start_monitoring(self):
        """Start the resource monitoring thread."""
        if self._monitor_thread is None:
            self._stop_monitoring = False
            self._monitor_thread = threading.Thread(target=self._monitor_resources)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        if self._monitor_thread is not None:
            self._stop_monitoring = True
            self._monitor_thread.join()
            self._monitor_thread = None
            logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources periodically."""
        while not self._stop_monitoring:
            try:
                # Get GPU metrics
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(0)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_usage = gpu_memory_allocated / gpu_memory_total
                else:
                    gpu_usage = 0.0
                
                # Get CPU and memory metrics
                cpu_usage = psutil.cpu_percent() / 100
                memory_usage = psutil.virtual_memory().percent / 100
                
                # Store metrics
                self.metrics.add_metric(gpu_usage, cpu_usage, memory_usage)
                
                # Check thresholds and log warnings
                if gpu_usage > self.gpu_memory_threshold:
                    logger.warning(f"GPU memory usage ({gpu_usage:.1%}) exceeds threshold")
                if cpu_usage > self.cpu_threshold:
                    logger.warning(f"CPU usage ({cpu_usage:.1%}) exceeds threshold")
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
            
            time.sleep(1)  # Update every second
    
    async def check_resources(self) -> Tuple[bool, str]:
        """Check if there are sufficient resources for a new request."""
        if self.active_requests >= self.max_concurrent_requests:
            return False, "Maximum concurrent requests reached"
        
        recent_metrics = self.metrics.get_average_metrics(minutes=1)
        
        if recent_metrics['gpu_usage'] > self.gpu_memory_threshold:
            return False, "GPU memory threshold exceeded"
        
        if recent_metrics['cpu_usage'] > self.cpu_threshold:
            return False, "CPU threshold exceeded"
        
        return True, "Resources available"
    
    def get_resource_metrics(self, minutes: int = 5) -> Dict[str, any]:
        """Get resource usage metrics."""
        avg_metrics = self.metrics.get_average_metrics(minutes)
        
        # Calculate request statistics
        recent_requests = [
            r for r in self.metrics.request_metrics
            if r['timestamp'] > datetime.utcnow() - timedelta(minutes=minutes)
        ]
        
        avg_duration = np.mean([r['duration'] for r in recent_requests]) if recent_requests else 0
        avg_gpu_memory = np.mean([r['gpu_memory'] for r in recent_requests]) if recent_requests else 0
        
        return {
            'current': {
                'gpu_usage': avg_metrics['gpu_usage'],
                'cpu_usage': avg_metrics['cpu_usage'],
                'memory_usage': avg_metrics['memory_usage'],
                'active_requests': self.active_requests,
                'max_concurrent_requests': self.max_concurrent_requests
            },
            'request_stats': {
                'average_duration': avg_duration,
                'average_gpu_memory': avg_gpu_memory,
                'requests_processed': len(recent_requests),
                'requests_per_minute': len(recent_requests) / minutes if minutes > 0 else 0
            },
            'thresholds': {
                'gpu_memory': self.gpu_memory_threshold,
                'cpu': self.cpu_threshold
            },
            'last_update': self.metrics.last_update.isoformat() if self.metrics.last_update else None
        }
    
    async def acquire_resources(self) -> bool:
        """Attempt to acquire resources for a new request."""
        resources_available, _ = await self.check_resources()
        if resources_available:
            self.active_requests += 1
            return True
        return False
    
    def release_resources(self):
        """Release resources after request completion."""
        self.active_requests = max(0, self.active_requests - 1)
    
    def record_request(self, duration: float, gpu_memory: float):
        """Record metrics for a completed request."""
        self.metrics.add_request_metric(duration, gpu_memory)

# Create singleton instance
resource_manager = ResourceManager()

# Start monitoring on module import
resource_manager.start_monitoring() 