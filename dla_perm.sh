#!/bin/bash

echo "🔧 Setting permissions for DLA utilization access..."

# Root check
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run this script with sudo or as root."
  exit 1
fi

echo ""
echo "▶ Granting +rx to /sys/kernel/debug..."
if [ -e /sys/kernel/debug ]; then
  chmod +rx /sys/kernel/debug
  echo "✅ Done"
else
  echo "⚠️ Path not found: /sys/kernel/debug"
fi

echo ""
echo "▶ Granting permissions for nvdla0..."
if [ -e /sys/kernel/debug/nvdla0 ]; then
  chmod +rx /sys/kernel/debug/nvdla0
  echo "✅ +rx for nvdla0"
else
  echo "⚠️ Path not found: /sys/kernel/debug/nvdla0"
fi

if [ -e /sys/kernel/debug/nvdla0/firmware/utilization_rate ]; then
  chmod +r /sys/kernel/debug/nvdla0/firmware/utilization_rate
  echo "✅ +r for nvdla0 utilization_rate"
else
  echo "⚠️ Path not found: /sys/kernel/debug/nvdla0/firmware/utilization_rate"
fi

echo ""
echo "▶ Granting permissions for nvdla1..."
if [ -e /sys/kernel/debug/nvdla1 ]; then
  chmod +rx /sys/kernel/debug/nvdla1
  echo "✅ +rx for nvdla1"
else
  echo "⚠️ Path not found: /sys/kernel/debug/nvdla1"
fi

if [ -e /sys/kernel/debug/nvdla1/firmware/utilization_rate ]; then
  chmod +r /sys/kernel/debug/nvdla1/firmware/utilization_rate
  echo "✅ +r for nvdla1 utilization_rate"
else
  echo "⚠️ Path not found: /sys/kernel/debug/nvdla1/firmware/utilization_rate"
fi

echo ""
echo "🔍 Final permission check:"
ls -l /sys/kernel/debug/nvdla*/firmware/utilization_rate 2>/dev/null || echo "⚠️ Unable to list utilization_rate files."

echo ""
echo "✅ Permission setup complete."

