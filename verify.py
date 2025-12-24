# verify.py
import sys
import os

print("🔍 验证项目环境...")

# 检查关键文件
required_files = [
    'requirements.txt',
    'config.py',
    'main.py',
    'src/data_generator.py',
    'src/preprocessor.py',
    'src/lstm_model.py',
    'app/app.py'
]

print("1. 检查项目结构...")
for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} 不存在")

# 检查Python包
print("\n2. 检查Python包...")
required_packages = ['numpy', 'pandas', 'tensorflow', 'flask']
for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} 未安装")

print("\n🎉 验证完成！运行 `python main.py` 开始训练模型。")