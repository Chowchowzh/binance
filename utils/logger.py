# -*- coding: utf-8 -*-
"""
日志管理工具模块
统一管理所有日志输出和结果记录
"""

import os
import sys
import datetime
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class ResultLogger:
    """结果日志管理器 - 统一管理日志输出和文件记录"""
    
    def __init__(self, results_dir: str = 'logs', log_level: str = 'INFO'):
        """
        初始化日志管理器
        
        Args:
            results_dir: 日志文件保存目录
            log_level: 日志级别
        """
        self.results_dir = results_dir
        self.log_lines: List[str] = []
        self.start_time = datetime.datetime.now()
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        # 设置日志文件路径
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(results_dir, f'operation_log_{timestamp}.txt')
        
        # 配置Python标准日志
        self.logger = logging.getLogger(f'ResultLogger_{timestamp}')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"日志管理器初始化完成，日志文件: {self.log_file}")
    
    def log(self, message: str, level: str = 'INFO', also_print: bool = True):
        """
        记录日志消息
        
        Args:
            message: 日志消息
            level: 日志级别
            also_print: 是否同时打印到控制台
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # 添加到内部列表
        self.log_lines.append(formatted_message)
        
        # 使用标准日志记录
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
        # 额外的控制台输出（如果需要）
        if also_print and level.upper() not in ['DEBUG']:
            print(message)
    
    def log_section(self, title: str, content: str = "", level: str = 'INFO', also_print: bool = True):
        """
        记录一个完整的分析章节
        
        Args:
            title: 章节标题
            content: 章节内容
            level: 日志级别
            also_print: 是否打印到控制台
        """
        separator = "=" * 60
        section_lines = [
            "",
            separator,
            f" {title}",
            separator
        ]
        
        if content:
            section_lines.extend(content.split('\n'))
        
        for line in section_lines:
            self.log(line, level, also_print)
    
    def log_dict(self, data: Dict[str, Any], title: str = "数据记录", level: str = 'INFO'):
        """
        记录字典数据
        
        Args:
            data: 要记录的字典数据
            title: 数据标题
            level: 日志级别
        """
        self.log_section(title, level=level)
        for key, value in data.items():
            self.log(f"{key}: {value}", level)
    
    def log_list(self, items: List[Any], title: str = "列表数据", level: str = 'INFO'):
        """
        记录列表数据
        
        Args:
            items: 要记录的列表
            title: 数据标题
            level: 日志级别
        """
        self.log_section(title, level=level)
        for i, item in enumerate(items):
            self.log(f"{i+1}. {item}", level)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.log(message, 'DEBUG', also_print=False)
    
    def info(self, message: str):
        """记录一般信息"""
        self.log(message, 'INFO')
    
    def warning(self, message: str):
        """记录警告信息"""
        self.log(f"⚠️ {message}", 'WARNING')
    
    def error(self, message: str):
        """记录错误信息"""
        self.log(f"❌ {message}", 'ERROR')
    
    def success(self, message: str):
        """记录成功信息"""
        self.log(f"✅ {message}", 'INFO')
    
    def progress(self, current: int, total: int, message: str = "处理进度"):
        """
        记录进度信息
        
        Args:
            current: 当前进度
            total: 总数
            message: 进度描述
        """
        percentage = (current / total) * 100 if total > 0 else 0
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        self.log(f"{message}: [{progress_bar}] {current}/{total} ({percentage:.1f}%)", 'INFO')
    
    def save_to_file(self, additional_content: str = "", file_suffix: str = ""):
        """
        保存日志到文件
        
        Args:
            additional_content: 额外要保存的内容
            file_suffix: 文件名后缀
        """
        try:
            if file_suffix:
                base_name = os.path.splitext(self.log_file)[0]
                save_file = f"{base_name}_{file_suffix}.txt"
            else:
                save_file = self.log_file
            
            with open(save_file, 'w', encoding='utf-8') as f:
                # 写入标题信息
                f.write(f"操作日志报告\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"持续时间: {datetime.datetime.now() - self.start_time}\n")
                f.write("=" * 80 + "\n\n")
                
                # 写入日志内容
                for line in self.log_lines:
                    f.write(line + '\n')
                
                # 写入额外内容
                if additional_content:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("额外信息:\n")
                    f.write(additional_content)
            
            self.info(f"日志已保存到: {save_file}")
            
        except Exception as e:
            self.error(f"保存日志文件失败: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        获取日志统计信息
        
        Returns:
            日志统计信息字典
        """
        total_lines = len(self.log_lines)
        duration = datetime.datetime.now() - self.start_time
        
        # 按级别统计
        level_counts = {}
        for handler in self.logger.handlers:
            if hasattr(handler, 'buffer'):
                # 统计不同级别的日志数量
                pass
        
        return {
            'total_log_lines': total_lines,
            'start_time': self.start_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'log_file': self.log_file,
            'results_dir': self.results_dir
        }
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """
        操作上下文管理器
        
        Args:
            operation_name: 操作名称
        """
        start_time = datetime.datetime.now()
        self.log_section(f"开始操作: {operation_name}")
        
        try:
            yield self
        except Exception as e:
            self.error(f"操作失败: {operation_name} - {e}")
            raise
        finally:
            duration = datetime.datetime.now() - start_time
            self.log_section(f"操作完成: {operation_name}")
            self.info(f"耗时: {duration}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error(f"程序异常退出: {exc_val}")
        
        # 保存最终日志
        self.save_to_file()
        
        # 关闭日志处理器
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class PerformanceLogger:
    """性能监控日志器"""
    
    def __init__(self, logger: ResultLogger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        self.timers[name] = datetime.datetime.now()
        self.logger.debug(f"开始计时: {name}")
    
    def end_timer(self, name: str):
        """结束计时并记录"""
        if name in self.timers:
            duration = datetime.datetime.now() - self.timers[name]
            self.logger.info(f"⏱️ {name} 耗时: {duration}")
            del self.timers[name]
            return duration
        else:
            self.logger.warning(f"计时器 {name} 未找到")
            return None
    
    @contextmanager
    def timer(self, name: str):
        """计时上下文管理器"""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)


# 兼容性函数 - 保持与原有代码的兼容性
def create_result_logger(results_dir: str = 'backtest_results') -> ResultLogger:
    """
    创建结果日志记录器（兼容性函数）
    
    Args:
        results_dir: 结果目录
        
    Returns:
        ResultLogger实例
    """
    return ResultLogger(results_dir)


def setup_logger(name: str = 'main', log_level: str = 'INFO', log_dir: str = 'logs') -> logging.Logger:
    """
    设置标准日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别
        log_dir: 日志目录
        
    Returns:
        配置好的Logger实例
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建文件处理器
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 