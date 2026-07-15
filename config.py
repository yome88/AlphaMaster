"""config.py — 统一配置模块（项目根目录）

所有子模块从此文件导入 Config，废弃各自的 config.py。
MT5 连接凭证通过环境变量或 .env 文件加载。

A股适配修改（2026-07-11）：
  - 新增 A_SHARE_SYMBOLS / A_SHARE_TRAINABLE_SYMBOLS / A_SHARE_FEATURE_SYMBOLS
  - 新增 MARKET_TYPE 自动判断（a_share / forex）
  - 新增 A股交易成本（印花税、过户费、佣金）
  - 新增 A股交易时间处理（9:30-11:30, 13:00-15:00）
  - 新增 A股仅做多模式（ALLOW_SHORT = False）
"""
import os
import re

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    # 测试环境无 MT5 时使用整数占位常量（与真实 MT5 值一致）
    class _MT5Stub:
        TIMEFRAME_M1  = 1
        TIMEFRAME_M5  = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1  = 16385
        TIMEFRAME_H4  = 16388
        TIMEFRAME_D1  = 16408
        TIMEFRAME_W1  = 32769
        TIMEFRAME_MN1 = 49153
    mt5 = _MT5Stub()

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()


# ── A股相关工具函数 ──────────────────────────────────────────

def _is_a_share_symbol(symbol: str) -> bool:
    """判断是否为 A 股品种。

    支持格式：
      - 纯数字代码：6xxxxx（上海）/ 0xxxxx/3xxxxx（深圳）/ 68xxxxx（科创）/ 30xxxxx（创业板）
      - 带前缀：SH600519 / SZ000001 / sz300750
    """
    s = (symbol or "").strip().upper().replace(".", "")
    # 纯数字
    if s.isdigit() and len(s) == 6:
        return True
    # SH/SZ 前缀
    if s.startswith("SH") and len(s) == 8 and s[2:].isdigit():
        return True
    if s.startswith("SZ") and len(s) == 8 and s[2:].isdigit():
        return True
    return False


def detect_market_type(symbol: str) -> str:
    """根据品种代码自动判断市场类型。

    Returns:
        'a_share' — A 股（含指数）
        'forex'   — 外汇/贵金属/美股/加密等
    """
    if _is_a_share_symbol(symbol):
        return "a_share"
    return "forex"


class Config:
    # ── MT5 连接 ──────────────────────────────────────────
    MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER   = os.getenv("MT5_SERVER", "")

    # ── 市场类型切换 ────────────────────────────────────
    # 设为 'a_share' 则全局使用 A 股参数；'auto' 按品种自动判断；'forex' 保持原行为
    MARKET_TYPE = os.getenv("MARKET_TYPE", "auto")

    # ── 做空权限 ──────────────────────────────────────────
    # A 股普通账户不支持做空（融券/股指期货除外），训练时仅使用做多信号
    ALLOW_SHORT_FOREX = True
    ALLOW_SHORT_A_SHARE = False

    @classmethod
    def allow_short(cls, symbol: str | None = None) -> bool:
        """根据当前品种/市场类型返回是否允许做空。"""
        mt = cls.MARKET_TYPE
        if mt == "a_share":
            return cls.ALLOW_SHORT_A_SHARE
        if mt == "forex":
            return cls.ALLOW_SHORT_FOREX
        if symbol is None:
            # auto 且无 symbol 时保守返回 True（回测可统一处理）
            return cls.ALLOW_SHORT_FOREX
        return cls.ALLOW_SHORT_A_SHARE if _is_a_share_symbol(symbol) else cls.ALLOW_SHORT_FOREX

    # ── 品种与周期（外汇/贵金属/美股）────────────────────────
    SYMBOLS_FOREX = [
        # 外汇
        "EURUSD", "USDJPY",
        # 贵金属
        "XAUUSD", "XAGUSD",
        # 美国指数
        "US30.cash", "US100.cash", "US500.cash", "US2000.cash",
        # 其他指数
        "JP225.cash",
    ]

    TRAINABLE_SYMBOLS_FOREX = [
        "EURUSD", "USDJPY",
        "XAUUSD",
        "US30.cash", "US100.cash", "US500.cash", "US2000.cash",
        "JP225.cash",
    ]

    # [deprecated] 相关性分组（已废弃，改用 TRAINABLE_SYMBOLS 单品种训练）
    SYMBOL_GROUPS_FOREX = {
        "forex":          ["EURUSD", "USDJPY"],
        "precious_metals":["XAUUSD", "XAGUSD"],
        "index":          ["US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash"],
    }

    FEATURE_SYMBOLS_FOREX = [
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
        "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURAUD",
        "EURCAD", "EURCHF", "GBPAUD", "GBPCAD", "GBPCHF",
        "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "NZDJPY",
        "CADCHF", "CADJPY", "CHFJPY",
        "XAUUSD", "XAGUSD", "XPTUSD",
        "DXY.cash",
        "USOIL.cash", "UKOIL.cash",
        "US30.cash", "US500.cash", "US100.cash", "UK100.cash",
        "DE30.cash", "FR40.cash", "JP225.cash", "AUS200.cash",
    ]

    # ── A 股品种列表 ──────────────────────────────────────
    # 蓝筹股示例（可用于训练/回测/实时监控）
    A_SHARE_SYMBOLS = [
        # 大盘蓝筹
        "SH600519",   # 贵州茅台
        "SH601318",   # 中国平安
        "SH600036",   # 招商银行
        "SH601888",   # 中国中免
        # 中小盘
        "SZ000001",   # 平安银行
        "SZ000858",   # 五粮液
        "SZ002594",   # 比亚迪
        # 创业板/科创板
        "SZ300750",   # 宁德时代
        "SH688981",   # 中芯国际
        # 指数
        "SH000001",   # 上证指数
        "SZ399006",   # 创业板指
        "SH000300",   # 沪深300
    ]

    A_SHARE_TRAINABLE_SYMBOLS = [
        "SH600519", "SH601318", "SH600036",
        "SZ000001", "SZ000858", "SZ002594",
        "SZ300750", "SH688981",
        "SH000001", "SZ399006", "SH000300",
    ]

    A_SHARE_FEATURE_SYMBOLS = [
        "SH600519", "SH601318", "SH600036", "SH601888",
        "SZ000001", "SZ000858", "SZ002594", "SZ300750",
        "SH688981", "SH000001", "SZ399006", "SH000300",
    ]

    # 兼容旧代码：默认使用外汇品种（单品种训练时由调用方传入具体品种）
    SYMBOLS = SYMBOLS_FOREX
    TRAINABLE_SYMBOLS = TRAINABLE_SYMBOLS_FOREX
    SYMBOL_GROUPS = SYMBOL_GROUPS_FOREX
    FEATURE_SYMBOLS = FEATURE_SYMBOLS_FOREX

    @classmethod
    def get_symbols(cls, market_type: str | None = None) -> list[str]:
        """根据市场类型返回对应的品种列表。"""
        mt = market_type or cls.MARKET_TYPE
        if mt == "a_share":
            return cls.A_SHARE_SYMBOLS
        return cls.SYMBOLS_FOREX

    @classmethod
    def get_trainable_symbols(cls, market_type: str | None = None) -> list[str]:
        mt = market_type or cls.MARKET_TYPE
        if mt == "a_share":
            return cls.A_SHARE_TRAINABLE_SYMBOLS
        return cls.TRAINABLE_SYMBOLS_FOREX

    @classmethod
    def get_feature_symbols(cls, market_type: str | None = None) -> list[str]:
        mt = market_type or cls.MARKET_TYPE
        if mt == "a_share":
            return cls.A_SHARE_FEATURE_SYMBOLS
        return cls.FEATURE_SYMBOLS_FOREX

    # ── 数据参数 ──────────────────────────────────────────
    TIMEFRAME             = mt5.TIMEFRAME_H1   # K 线周期（外汇默认 H1）
    A_SHARE_TIMEFRAME     = mt5.TIMEFRAME_D1   # A 股默认日线（日线更适合 A 股）
    BARS_COUNT            = 10_000_000
    MIN_BARS              = 3000
    DATA_REFRESH_INTERVAL = 300
    KLINE_CACHE_DIR       = os.getenv("KLINE_CACHE_DIR", r"D:\K线数据")
    A_SHARE_CACHE_DIR     = os.getenv("A_SHARE_CACHE_DIR", r"D:\K线数据\A股")

    # ── 模型参数（仅供参考，训练实际使用 model_core.config.ModelConfig）────
    INPUT_DIM       = 20
    BATCH_SIZE      = 128
    TRAIN_STEPS     = 300
    MAX_FORMULA_LEN = 8
    DEVICE          = (
        torch.device("cpu")
        if _TORCH_AVAILABLE
        else "cpu"
    )

    # ── 风控参数（外汇默认值）──────────────────────────────
    RISK_PER_TRADE     = 0.01
    COST_RATE_FOREX    = 0.0001     # 单边点差+佣金（forex/metals，约 0.01%）
    MAX_OPEN_POSITIONS = 4
    MAX_LOT_PER_TRADE  = 5.0
    EXCLUDED_TRADE_SYMBOLS = ["XAGUSD"]

    # ── A 股交易成本（万分之级别）──────────────────────────
    # 印花税：卖出时 0.05%（单边）
    # 过户费：上海 0.001%（双边），深圳免
    # 佣金：约 0.025%（双边），最低 5 元
    # 总单边成本 ≈ 0.025% + 0.001% = 0.026%（买入）；卖出额外 +0.05% 印花税
    # 简化模型：统一 cost_rate 取 0.00035（0.035%）≈ 买卖合计平均
    COST_RATE_A_SHARE        = 0.00035   # 单边综合费率（佣金+过户费，不含印花税）
    A_SHARE_STAMP_TAX        = 0.0005    # 印花税（仅卖出，0.05%）
    A_SHARE_TRANSFER_FEE     = 0.00001   # 过户费（0.001%，上海）
    A_SHARE_COMMISSION       = 0.00025   # 佣金（0.025%）
    A_SHARE_MIN_COMMISSION   = 5.0       # 最低佣金 5 元（实盘用，回测忽略）

    @classmethod
    def get_cost_rate(cls, symbol: str | None = None) -> float:
        """根据品种/市场类型返回适用的单边成本率。"""
        mt = cls.MARKET_TYPE
        if mt == "a_share":
            return cls.COST_RATE_A_SHARE
        if mt == "forex":
            return cls.COST_RATE_FOREX
        if symbol is None:
            return cls.COST_RATE_FOREX
        return cls.COST_RATE_A_SHARE if _is_a_share_symbol(symbol) else cls.COST_RATE_FOREX

    FIXED_LOT_BY_SYMBOL = {
        "XAUUSD": 0.01,
    }
    VOL_TARGET_REFERENCE_SYMBOL = "XAUUSD"
    VOL_TARGET_REFERENCE_LOT = 0.01
    VOL_TARGET_SHARPE_REFERENCE = 2.447
    VOL_TARGET_SHARPE_EXPONENT = 0.50
    VOL_TARGET_MIN_SHARPE_WEIGHT = 0.50
    VOL_TARGET_MAX_SHARPE_WEIGHT = 1.50
    VOL_TARGET_SHARPE_BY_SYMBOL = {
        "XAUUSD": 2.447,
        "US100.cash": 1.811,
        "US500.cash": 0.959,
        "US2000.cash": 0.575,
        "US30.cash": 0.923,
        "JP225.cash": -0.653,
    }
    MIN_TRADE_EXPOSURE = 0.05

    # ── 策略参数 ──────────────────────────────────────────
    SIGNAL_MODE = "backtest_parity"
    EXIT_MODE = "signal"
    BUY_THRESHOLD       = 0.70
    SELL_THRESHOLD      = 0.40
    STOP_LOSS_PCT       = -0.02
    TAKE_PROFIT_PCT     = 0.04
    TRAILING_ACTIVATION = 0.03
    TRAILING_DROP       = 0.015
    REBALANCE_ON_BAR_CLOSE = True
    EXECUTION_LAG_BARS     = 1
    MAX_OPEN_POSITIONS: int | None = None

    # ── A 股交易时间（Unix 秒，北京时间 UTC+8）──────────────
    # 用于实时分析时判断是否在交易时段内
    A_SHARE_TRADE_HOURS = {
        "morning": (9 * 3600 + 30 * 60, 11 * 3600 + 30 * 60),   # 09:30 - 11:30
        "afternoon": (13 * 3600 + 0 * 60, 15 * 3600 + 0 * 60),   # 13:00 - 15:00
    }
    A_SHARE_TZ_OFFSET = 8 * 3600  # UTC+8

    # ── 文件路径 ──────────────────────────────────────────
    STRATEGY_FILE  = "best_mt5_strategy.json"
    PORTFOLIO_FILE = "portfolio_state.json"
    STOP_SIGNAL    = "STOP_SIGNAL"

    # ── Magic Number ──────────────────────────────────────
    MAGIC_NUMBER = 20250101

    @classmethod
    def get_timeframe(cls, tf_str: str) -> int:
        """将字符串（如 'H1'）映射为 MT5 时间周期常量。

        支持的周期：M1, M5, M15, M30, H1, H4, D1, W1, MN1
        """
        mapping = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        if tf_str not in mapping:
            raise ValueError(
                f"Unknown timeframe: '{tf_str}'. "
                f"Supported values: {list(mapping.keys())}"
            )
        return mapping[tf_str]

    # ── A 股辅助方法 ──────────────────────────────────────
    @classmethod
    def is_a_share_trade_time(cls, timestamp: int | None = None) -> bool:
        """判断给定 Unix 时间戳是否处于 A 股交易时段（仅考虑小时，忽略节假日）。

        Args:
            timestamp: Unix 秒，默认取当前时间
        """
        import time as _time
        ts = timestamp if timestamp is not None else int(_time.time())
        # 转换为北京时间当日秒数
        local_sec = (ts + cls.A_SHARE_TZ_OFFSET) % 86400
        for start, end in cls.A_SHARE_TRADE_HOURS.values():
            if start <= local_sec <= end:
                return True
        return False

    @classmethod
    def is_a_share_trade_day(cls, timestamp: int | None = None) -> bool:
        """判断是否为 A 股交易日（周一至周五，忽略节假日）。

        注意：未内置交易所休市日历，需外部数据源或手动维护 holiday list。
        """
        import time as _time
        from datetime import datetime as _datetime, timezone as _timezone
        ts = timestamp if timestamp is not None else int(_time.time())
        dt = _datetime.fromtimestamp(ts, tz=_timezone.utc)
        # 周一=0 ... 周五=4
        return dt.weekday() <= 4
