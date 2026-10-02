"""프로젝트 루트에서 실행: python code/run_pipeline.py"""
from pathlib import Path
import os
import tempfile
import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="ds-mini-") as temporary:
    temporary = Path(temporary)
    install(prefix=str(temporary), kernel_name="ds-mini", display_name="Python (DS Mini)")
    os.environ["JUPYTER_PATH"] = str(temporary / "share/jupyter")
    os.environ["MPLCONFIGDIR"] = str(temporary / "matplotlib")
    os.environ["IPYTHONDIR"] = str(temporary / "ipython")
    path = ROOT / "code/model.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=180, kernel_name="ds-mini",
                   resources={"metadata": {"path": str(ROOT / "code")}}).execute()
    nbformat.write(notebook, path)
print("완료: model.ipynb 및 results/model_performance.csv 갱신")
