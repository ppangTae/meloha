import rclpy
from rclpy.logging import LoggingSeverity

# 유효한 로그 레벨 매핑 딕셔너리
LOG_LEVEL_MAP = {
    'DEBUG': LoggingSeverity.DEBUG,
    'INFO': LoggingSeverity.INFO,
    'WARN': LoggingSeverity.WARN,
    'WARNING': LoggingSeverity.WARN,
    'ERROR': LoggingSeverity.ERROR,
    'FATAL': LoggingSeverity.FATAL,
    'CRITICAL': LoggingSeverity.FATAL,
}

def normalize_log_level(level_str: str) -> int:
    """
    사용자 입력 문자열을 받아서 적절한 LoggingSeverity 상수로 변환

    :param level_str: 사용자 입력 (e.g., "debug", "INFO", "Warning")
    :return: rclpy.logging.LoggingSeverity 상수
    :raises: ValueError
    """
    level_str_upper = level_str.strip().upper()
    if level_str_upper not in LOG_LEVEL_MAP:
        raise ValueError(
            f"Invalid logging level: '{level_str}'. "
            f"Valid options are: {list(LOG_LEVEL_MAP.keys())}"
        )
    return LOG_LEVEL_MAP[level_str_upper]