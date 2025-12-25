import sys
import os

print("Verifying project environment...")

# Check critical project files
required_files = [
    'requirements.txt',
    'config.py',
    'main.py',
    'src/data_generator.py',
    'src/preprocessor.py',
    'src/lstm_model.py',
    'app/app.py'
]

print("1. Checking project structure...")
for file in required_files:
    if os.path.exists(file):
        print(f"   {file} - OK")
    else:
        print(f"   {file} - NOT FOUND")

# Check required Python packages
print("\n2. Checking Python packages...")
required_packages = ['numpy', 'pandas', 'tensorflow', 'flask']
for package in required_packages:
    try:
        __import__(package)
        print(f"   {package} - Installed")
    except ImportError:
        print(f"   {package} - Not installed")

print("\nVerification completed! Run `python main.py` to start model training.")