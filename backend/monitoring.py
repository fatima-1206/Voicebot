"""
WebSocket monitoring and health check utilities
"""
import asyncio
import time
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from backend.ws_manager import manager
from backend.workers.utilities import get_redis_health, get_model_status
import json

logger = logging.getLogger(__name__)

class WebSocketMonitor:
    """Comprehensive WebSocket system monitor"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.is_running = False
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        self.alerts: List[Dict[str, Any]] = []
        
    async def start_monitoring(self):
        """Start the monitoring loop"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🔍 Starting WebSocket monitoring")
        
        while self.is_running:
            try:
                health_data = await self.collect_health_data()
                self.store_health_data(health_data)
                await self.check_alerts(health_data)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(30)  # Shorter retry on error
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.is_running = False
        logger.info("🛑 Stopping WebSocket monitoring")
    
    async def collect_health_data(self) -> Dict[str, Any]:
        """Collect comprehensive health data"""
        timestamp = datetime.now()
        
        # WebSocket health
        ws_health = await manager.health_check()
        
        # Redis health
        redis_health = get_redis_health()
        
        # Model status
        model_status = get_model_status()
        
        # System metrics
        connection_count = manager.get_connection_count()
        connection_metrics = {
            "active_connections": connection_count,
            "total_created": manager.metrics.connections_created,
            "total_closed": manager.metrics.connections_closed,
            "successful_sends": manager.metrics.successful_sends,
            "failed_sends": manager.metrics.failed_sends,
            "send_success_rate": (
                manager.metrics.successful_sends / 
                max(1, manager.metrics.successful_sends + manager.metrics.failed_sends)
            ) * 100
        }
        
        return {
            "timestamp": timestamp.isoformat(),
            "websocket_health": ws_health,
            "redis_health": redis_health,
            "model_status": model_status,
            "connection_metrics": connection_metrics,
            "overall_status": self.calculate_overall_status(
                ws_health, redis_health, model_status, connection_metrics
            )
        }
    
    def calculate_overall_status(self, ws_health: Dict, redis_health: Dict, 
                               model_status: Dict, connection_metrics: Dict) -> str:
        """Calculate overall system status"""
        issues = []
        
        # Check WebSocket health
        if ws_health.get("status") != "healthy":
            issues.append("websocket_unhealthy")
        
        # Check Redis health
        if not redis_health.get("is_healthy", False):
            issues.append("redis_unhealthy")
        
        # Check critical models
        if not model_status.get("whisper_model", False):
            issues.append("whisper_model_missing")
        
        # Check send success rate
        send_rate = connection_metrics.get("send_success_rate", 100)
        if send_rate < 95:
            issues.append("low_send_success_rate")
        
        # Check for too many failed connections
        failed_sends = connection_metrics.get("failed_sends", 0)
        if failed_sends > 100:
            issues.append("high_failure_count")
        
        if not issues:
            return "healthy"
        elif len(issues) == 1 and "low_send_success_rate" in issues:
            return "degraded"
        else:
            return "unhealthy"
    
    def store_health_data(self, health_data: Dict[str, Any]):
        """Store health data with rotation"""
        self.health_history.append(health_data)
        
        # Rotate history if too large
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    async def check_alerts(self, health_data: Dict[str, Any]):
        """Check for alert conditions"""
        alerts = []
        current_time = datetime.now()
        
        # Check overall status
        overall_status = health_data["overall_status"]
        if overall_status in ["degraded", "unhealthy"]:
            alerts.append({
                "type": "system_status",
                "severity": "warning" if overall_status == "degraded" else "critical",
                "message": f"System status is {overall_status}",
                "timestamp": current_time.isoformat(),
                "data": health_data
            })
        
        # Check connection count trends
        if len(self.health_history) >= 5:
            recent_counts = [
                h["connection_metrics"]["active_connections"] 
                for h in self.health_history[-5:]
            ]
            
            # Check for sudden connection drops
            if len(recent_counts) >= 2:
                latest = recent_counts[-1]
                previous = recent_counts[-2]
                
                if previous > 0 and latest < previous * 0.5:  # 50% drop
                    alerts.append({
                        "type": "connection_drop",
                        "severity": "warning",
                        "message": f"Significant connection drop: {previous} -> {latest}",
                        "timestamp": current_time.isoformat()
                    })
        
        # Check Redis health
        if not health_data["redis_health"]["is_healthy"]:
            alerts.append({
                "type": "redis_health",
                "severity": "critical",
                "message": "Redis is unhealthy",
                "timestamp": current_time.isoformat()
            })
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"🚨 ALERT: {alert['message']}")
        
        # Rotate alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        
        latest = self.health_history[-1]
        
        # Calculate uptime metrics
        healthy_count = sum(
            1 for h in self.health_history 
            if h["overall_status"] == "healthy"
        )
        uptime_percentage = (healthy_count / len(self.health_history)) * 100
        
        return {
            "current_status": latest["overall_status"],
            "timestamp": latest["timestamp"],
            "uptime_percentage": uptime_percentage,
            "total_checks": len(self.health_history),
            "active_connections": latest["connection_metrics"]["active_connections"],
            "recent_alerts": self.alerts[-5:] if self.alerts else [],
            "redis_healthy": latest["redis_health"]["is_healthy"],
            "models_loaded": {
                "whisper": latest["model_status"]["whisper_model"],
                "vad": latest["model_status"]["vad_model"],
                "groq": latest["model_status"]["groq_client"]
            }
        }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed monitoring report"""
        if not self.health_history:
            return {"status": "no_data"}
        
        return {
            "summary": self.get_health_summary(),
            "recent_history": self.health_history[-10:],
            "all_alerts": self.alerts,
            "trends": self.calculate_trends(),
            "recommendations": self.get_recommendations()
        }
    
    def calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(self.health_history) < 5:
            return {"status": "insufficient_data"}
        
        recent_data = self.health_history[-10:]
        
        # Connection count trend
        connections = [h["connection_metrics"]["active_connections"] for h in recent_data]
        conn_trend = "stable"
        if len(connections) >= 2:
            if connections[-1] > connections[0] * 1.2:
                conn_trend = "increasing"
            elif connections[-1] < connections[0] * 0.8:
                conn_trend = "decreasing"
        
        # Success rate trend
        success_rates = [h["connection_metrics"]["send_success_rate"] for h in recent_data]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        return {
            "connection_trend": conn_trend,
            "average_success_rate": avg_success_rate,
            "stability": "stable" if avg_success_rate > 95 else "unstable"
        }
    
    def get_recommendations(self) -> List[str]:
        """Get system recommendations based on health data"""
        if not self.health_history:
            return ["No data available for recommendations"]
        
        latest = self.health_history[-1]
        recommendations = []
        
        # Redis recommendations
        if not latest["redis_health"]["is_healthy"]:
            recommendations.append("Check Redis server health and restart if necessary")
        
        # Connection recommendations
        metrics = latest["connection_metrics"]
        if metrics["send_success_rate"] < 95:
            recommendations.append("Investigate WebSocket send failures")
        
        if metrics["active_connections"] > 1000:
            recommendations.append("Consider scaling WebSocket infrastructure")
        
        # Model recommendations
        models = latest["model_status"]
        if not models["whisper_model"]:
            recommendations.append("Reload Whisper model")
        
        if not models["cuda_available"] and models["whisper_model"]:
            recommendations.append("Consider GPU acceleration for better performance")
        
        if not recommendations:
            recommendations.append("System is healthy - no specific recommendations")
        
        return recommendations

# Global monitor instance
monitor = WebSocketMonitor()

async def start_monitoring():
    """Start the monitoring system"""
    await monitor.start_monitoring()

def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    return monitor.get_health_summary()

def get_monitoring_report() -> Dict[str, Any]:
    """Get detailed monitoring report"""
    return monitor.get_detailed_report()