#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. 安装你的 Python 依赖
pip install -r requirements.txt

# 2. (核心) 更新系统包列表并安装字体管理工具和备用中文字体
apt-get update && apt-get install -y fontconfig fonts-wqy-zenhei

# 3. (重要) 清理 matplotlib 的旧字体缓存，强制它在下次启动时重建
rm -rf $(python -c "import matplotlib; print(matplotlib.get_cachedir())")