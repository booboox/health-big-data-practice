"""运行原课程集成学习示例；参数与 scripts/demo_ensemble.py 相同。"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1] / "scripts" / "demo_ensemble.py"
    runpy.run_path(str(source), run_name="__main__")
