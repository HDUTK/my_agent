#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于MCP服务
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/18 20:17
version: V1.0
Target Python: 
"""

import logging
from mcp.server.fastmcp import FastMCP
import inspect
import tools

# 🌟 屏蔽底层库的 INFO 级别刷屏日志，只放行 WARNING 和 ERROR
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)


# 1. 创建标准化 Server 实例
mcp = FastMCP("my_server")

# 2. 自动扫描并注册 tools.py 中的所有公开函数
for name, func in inspect.getmembers(tools, inspect.isfunction):
    if not name.startswith("_"):
        mcp.add_tool(func)
        # print(f"✅ 工具已挂载: {name}") # 在实际 stdio 运行中，最好不要用 print，会污染通信通道

if __name__ == "__main__":
    # 3. 启动标准 stdio 服务
    mcp.run()
