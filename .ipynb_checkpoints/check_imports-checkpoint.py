# check_imports.py
import os

def check_file_imports(filepath):
    """Revisa si un archivo tiene imports relativos"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    relative_imports = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        if 'from .' in line and 'import' in line:
            relative_imports.append((i, line.strip()))
    
    return relative_imports

print("🔍 Revisando imports relativos en archivos src/...")

src_files = [
    'src/data_loader.py',
    'src/feature_engineer.py', 
    'src/preprocessor.py',
    'src/main.py',
    'src/utils.py'
]

all_errors = []

for filepath in src_files:
    if os.path.exists(filepath):
        errors = check_file_imports(filepath)
        if errors:
            print(f"❌ {filepath} tiene imports relativos:")
            for line_num, line_content in errors:
                print(f"   Línea {line_num}: {line_content}")
            all_errors.extend([(filepath, line_num, line_content) for line_num, line_content in errors])
        else:
            print(f"✅ {filepath} - OK")
    else:
        print(f"⚠️  {filepath} no encontrado")

if not all_errors:
    print("\n🎉 Todos los imports están correctos!")
else:
    print(f"\n🔧 Se encontraron {len(all_errors)} imports relativos que necesitan corrección")