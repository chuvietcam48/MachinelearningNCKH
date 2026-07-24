import os
import json
import subprocess

EXCLUDE_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', 'outputs', 'data'}
EXCLUDE_FILES = {'TREE_BEFORE_CLEANUP.md', 'generate_tree.py', 'snapshot_repo.py'}

def get_tree(start_path):
    tree_str = "```text\n"
    file_count = 0
    dir_count = 0
    
    for root, dirs, files in os.walk(start_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        dirs.sort()
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree_str += f"{indent}{os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            if f not in EXCLUDE_FILES and not f.endswith('.pyc'):
                tree_str += f"{subindent}{f}\n"
                file_count += 1
        dir_count += 1
    tree_str += "```\n"
    return tree_str, file_count, dir_count

if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree_str, f_count, d_count = get_tree(repo_path)
    
    # Git info
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_path).decode().strip()
        git_branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo_path).decode().strip()
    except:
        git_hash = "unknown"
        git_branch = "unknown"

    with open(os.path.join(repo_path, 'TREE_BEFORE_CLEANUP.md'), 'w', encoding='utf-8') as f:
        f.write("# Repository Snapshot (Phase 0.5)\n\n")
        f.write(f"- **Git Branch**: `{git_branch}`\n")
        f.write(f"- **Git HEAD**: `{git_hash}`\n")
        f.write(f"- **Total Directories (excl. data/outputs)**: {d_count}\n")
        f.write(f"- **Total Files**: {f_count}\n\n")
        f.write("## Repository Tree\n")
        f.write(tree_str)
    
    print(f"Snapshot saved to TREE_BEFORE_CLEANUP.md (Files: {f_count}, Dirs: {d_count})")
