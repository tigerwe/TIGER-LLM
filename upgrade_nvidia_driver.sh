#!/bin/bash
# NVIDIA 驱动升级脚本
# 适用于 Ubuntu 20.04

set -e  # 遇到错误立即停止

echo "=========================================="
echo "  NVIDIA 驱动升级脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否为 root 或使用 sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}错误: 请使用 sudo 运行此脚本${NC}"
    echo "用法: sudo bash upgrade_nvidia_driver.sh"
    exit 1
fi

# 记录开始时间
START_TIME=$(date +%s)

# 1. 备份当前状态
echo -e "${YELLOW}[1/7] 备份当前驱动信息...${NC}"
mkdir -p /root/nvidia_driver_backup
nvidia-smi > /root/nvidia_driver_backup/nvidia-smi-before.txt 2>/dev/null || true
dpkg -l | grep nvidia > /root/nvidia_driver_backup/nvidia-packages-before.txt 2>/dev/null || true
echo -e "${GREEN}✓ 备份完成${NC}"

# 2. 添加 PPA 源
echo -e "${YELLOW}[2/7] 添加 Graphics Drivers PPA...${NC}"
add-apt-repository -y ppa:graphics-drivers/ppa
apt update
echo -e "${GREEN}✓ PPA 添加完成${NC}"

# 3. 查看推荐驱动
echo -e "${YELLOW}[3/7] 检测推荐驱动版本...${NC}"
echo "可用的驱动:"
ubuntu-drivers devices | grep -E "driver|model" | head -10
echo ""
RECOMMENDED=$(ubuntu-drivers devices | grep "recommended" | awk '{print $3}')
echo -e "推荐安装: ${GREEN}$RECOMMENDED${NC}"

# 4. 卸载旧驱动（可选，但推荐）
echo -e "${YELLOW}[4/7] 准备卸载旧驱动...${NC}"
read -p "是否先卸载旧驱动？(推荐，输入 y 确认): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    apt purge -y nvidia-*
    apt autoremove -y
    echo -e "${GREEN}✓ 旧驱动已卸载${NC}"
else
    echo "跳过卸载步骤"
fi

# 5. 安装新驱动
echo -e "${YELLOW}[5/7] 安装 NVIDIA 驱动 535...${NC}"
apt install -y nvidia-driver-535
apt install -y nvidia-dkms-535
echo -e "${GREEN}✓ 驱动安装完成${NC}"

# 6. 验证安装
echo -e "${YELLOW}[6/7] 验证安装...${NC}"
if [ -f /usr/bin/nvidia-smi ]; then
    echo -e "${GREEN}✓ nvidia-smi 已安装${NC}"
else
    echo -e "${RED}✗ nvidia-smi 未找到，安装可能失败${NC}"
    exit 1
fi

# 7. 完成
echo -e "${YELLOW}[7/7] 安装完成！${NC}"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  驱动升级完成！${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "耗时: $((DURATION / 60)) 分 $((DURATION % 60)) 秒"
echo ""
echo -e "${YELLOW}重要: 必须重启系统才能使新驱动生效！${NC}"
echo ""
read -p "是否立即重启？(输入 y 立即重启，其他键稍后手动重启): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "系统将在 5 秒后重启..."
    sleep 5
    reboot
else
    echo "请稍后手动重启系统: sudo reboot"
fi
