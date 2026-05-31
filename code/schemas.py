#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于模具
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/28 20:37
version: V1.0
Target Python: 
"""

from pydantic import BaseModel, Field


# ==========================================
# 🌟 A63 传感器数据提取模具
# ==========================================
class SensorRecord(BaseModel):
    """A63 传感器特定时间点的数据提取与风险评估"""
    target_time: str = Field(description="目标提取时间，例如：2023/10/10 00:00")
    air_temperature :float = Field(description="提取到的空气温度数值 (℃)")
    air_humidity: float = Field(description="提取到的空气湿度数值 (%)")
    wall_temperature: float = Field(description="提取到的壁面温度数值 (℃)")

    # 大模型会自动根据这段 prompt 去计算 True 还是 False
    has_risk: bool = Field(description="风险判定：如果空气温度超过 20度 或 空气湿度超过 60%，必须判定为 True，否则为 False")
    action_advice: str = Field(description="结合提取的数据，给出不超过 50 个字的架构师专业排险建议")


class ChatResponse(BaseModel):
    # 未来你定义的其他模具
    pass