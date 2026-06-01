#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 此文件用于
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2026.1.1
time: 2026/05/12 22:20
version: V1.0
Target Python:  3.14 -> 3.12
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
function description: 诊断版 - 检查可用的 Gemini 模型并自带网络自检功能
"""

import os
import requests
import urllib3
import socket
import traceback
from dotenv import load_dotenv

# 🌟 屏蔽因关闭 SSL 验证而产生的烦人黄色警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. 加载你的 .env 文件中的 API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ 找不到 API Key，请检查 .env 文件！")
    exit()

PROXY_PORT = "7897"

# ==========================================
# 🌟 新增：代理端口连通性自检
# ==========================================
def check_proxy_port(port):
    """尝试连接本地代理端口，检查代理软件是否真正存活"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)  # 2秒超时
        result = s.connect_ex(('127.0.0.1', int(port)))
        return result == 0

print("🔍 正在进行网络自检...")
if not check_proxy_port(PROXY_PORT):
    print(f"⚠️  警告：检测到本地端口 {PROXY_PORT} 并未开放！")
    print("💡 请确认您的代理软件（如 Clash/v2ray）已经启动，或者端口是否真的是 7890。")
    print("-" * 50)
else:
    print(f"✅ 代理端口 {PROXY_PORT} 畅通。")

print(f"⏳ 正在通过代理连接 Google 服务器...\n")

proxies = {
    "http": f"http://127.0.0.1:{PROXY_PORT}",
    "https": f"http://127.0.0.1:{PROXY_PORT}"
}

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    # 加入了 timeout=15 防止程序无休止地死等
    response = requests.get(url, proxies=proxies, verify=False, timeout=15)

    if response.status_code == 200:
        models = response.json().get('models', [])
        print("✅ 成功获取！你目前可用的 Gemini 模型有：\n" + "=" * 50)

        for m in models:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                clean_name = m['name'].replace('models/', '')
                print(f"🚀 模型名称: {clean_name}")
                print(f"📝 模型介绍: {m.get('description', '暂无介绍')}")
                print("-" * 50)
    else:
        print(f"❌ 获取失败！HTTP 状态码: {response.status_code}")
        print(f"💡 接口返回内容: {response.text}")

# ==========================================
# 🌟 新增：精准的错误捕获与诊断说明
# ==========================================
except requests.exceptions.ProxyError:
    print("\n❌ 代理错误 (ProxyError)！")
    print(f"💡 原因：无法通过 127.0.0.1:{PROXY_PORT} 发送请求。请彻底检查代理软件设置。")
except requests.exceptions.Timeout:
    print("\n❌ 请求超时 (Timeout)！")
    print("💡 原因：已经连接上了您的代理，但是代理节点无法连接到 Google（节点失效或被墙），请尝试切换节点。")
except Exception as e:
    print("\n❌ 发生了未知错误，详细报错信息如下：")
    print("-" * 50)
    # 打印完整的红色错误追踪栈，方便我们定位问题
    traceback.print_exc()
    print("-" * 50)