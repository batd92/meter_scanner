#!/usr/bin/env python3
"""
Performance Monitor Service - Track and optimize system performance
"""

import time
import psutil
import threading
from typing import Dict, Any, List
from collections import deque
import json

class PerformanceMonitor:
    """Monitor and track system performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.cpu_threshold = 80.0  # CPU usage threshold
        self.memory_threshold = 85.0  # Memory usage threshold
        self.inference_time_threshold = 2.0  # Inference time threshold (seconds)
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🔍 Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # Check for performance issues
                self._check_performance_alerts(metrics)
                
                time.sleep(interval)
            except Exception as e:
                print(f"❌ Error in performance monitoring: {e}")
                time.sleep(interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_cpu_percent = process.cpu_percent()
            process_memory_mb = process.memory_info().rss / (1024**2)
            
            return {
                'timestamp': time.time(),
                'uptime': time.time() - self.start_time,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'process_percent': process_cpu_percent
                },
                'memory': {
                    'percent': memory_percent,
                    'used_gb': round(memory_used_gb, 2),
                    'total_gb': round(memory_total_gb, 2),
                    'process_mb': round(process_memory_mb, 2)
                },
                'disk': {
                    'percent': disk_percent,
                    'used_gb': round(disk_used_gb, 2),
                    'total_gb': round(disk_total_gb, 2)
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                }
            }
        except Exception as e:
            return {
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance issues and generate alerts"""
        alerts = []
        
        # CPU alerts
        if metrics.get('cpu', {}).get('percent', 0) > self.cpu_threshold:
            alerts.append(f"High CPU usage: {metrics['cpu']['percent']:.1f}%")
        
        # Memory alerts
        if metrics.get('memory', {}).get('percent', 0) > self.memory_threshold:
            alerts.append(f"High memory usage: {metrics['memory']['percent']:.1f}%")
        
        # Process memory alerts
        process_memory = metrics.get('memory', {}).get('process_mb', 0)
        if process_memory > 1000:  # 1GB threshold
            alerts.append(f"High process memory: {process_memory:.1f}MB")
        
        if alerts:
            print(f"⚠️  Performance alerts: {'; '.join(alerts)}")
    
    def record_inference_time(self, service: str, inference_time: float):
        """Record inference time for a specific service"""
        metric = {
            'timestamp': time.time(),
            'type': 'inference',
            'service': service,
            'inference_time': inference_time
        }
        
        self.metrics_history.append(metric)
        
        # Check inference time threshold
        if inference_time > self.inference_time_threshold:
            print(f"⚠️  Slow inference detected: {service} took {inference_time:.3f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        # Filter system metrics (exclude inference metrics)
        system_metrics = [m for m in self.metrics_history if 'cpu' in m]
        
        if not system_metrics:
            return {'error': 'No system metrics available'}
        
        # Calculate averages
        cpu_percentages = [m['cpu']['percent'] for m in system_metrics]
        memory_percentages = [m['memory']['percent'] for m in system_metrics]
        
        # Filter inference metrics
        inference_metrics = [m for m in self.metrics_history if m.get('type') == 'inference']
        
        return {
            'uptime_seconds': time.time() - self.start_time,
            'total_metrics': len(self.metrics_history),
            'system_metrics': len(system_metrics),
            'inference_metrics': len(inference_metrics),
            'averages': {
                'cpu_percent': sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0,
                'memory_percent': sum(memory_percentages) / len(memory_percentages) if memory_percentages else 0
            },
            'current': self.get_current_metrics()
        }
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics"""
        return list(self.metrics_history)[-count:]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        try:
            metrics_list = list(self.metrics_history)
            with open(filepath, 'w') as f:
                json.dump(metrics_list, f, indent=2)
            print(f"✅ Metrics exported to: {filepath}")
        except Exception as e:
            print(f"❌ Error exporting metrics: {e}")
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        print("🗑️  Metrics history cleared")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        current_metrics = self.get_current_metrics()
        
        # CPU recommendations
        cpu_percent = current_metrics.get('cpu', {}).get('percent', 0)
        if cpu_percent > 70:
            recommendations.append("Consider reducing batch size or using GPU acceleration")
        elif cpu_percent < 20:
            recommendations.append("System is underutilized - consider increasing batch size")
        
        # Memory recommendations
        memory_percent = current_metrics.get('memory', {}).get('percent', 0)
        if memory_percent > 80:
            recommendations.append("High memory usage - consider enabling memory cleanup")
        
        # Process memory recommendations
        process_memory = current_metrics.get('memory', {}).get('process_mb', 0)
        if process_memory > 500:
            recommendations.append("High process memory - consider model optimization")
        
        return recommendations

# Global performance monitor instance
performance_monitor = PerformanceMonitor() 