#!/bin/bash
# NVIDIA 驱动回滚脚本

set -e

if [ "$EUID" -ne 0 ]; then 
    echo "错误: 请使用 sudo 运行此脚本"
    exit 1
fi

echo "开始回滚到旧驱动..."

# 卸载新驱动
apt purge -y nvidia-driver-535 nvidia-dkms-535 2>/dev/null || true

# 重新安装旧驱动
apt install -y nvidia-driver-470

# 重启
echo "回滚完成，请重启系统"
read -p "是否立即重启？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    reboot
fi
