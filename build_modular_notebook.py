"""Create the single-file, code-only Colab deliverable."""
import json
from pathlib import Path
from colab_layout import make_notebook

ROOT = Path(__file__).parent
SOURCE_FILES = {
    'marso_experiment.py':'marso_experiment.py',
    'colab_layout.py':'colab_layout.py',
    'colab_policy.py':'t4_policy.py',
    'colab_eval_modular.py':'colab_eval_modular.py',
    'download_dataset_colab.py':'download_dataset.py',
    'test_policy_runtime.py':'test_t4_policy.py',
}


def sources():
    return {name:(ROOT/path).read_text(encoding='utf-8') for name,path in SOURCE_FILES.items()}


def build():
    notebook = make_notebook(sources())
    for i,cell in enumerate(notebook['cells']):
        assert cell['cell_type']=='code'
        compile(''.join(cell['source']),f'cell-{i}','exec')
    path = ROOT/'marso_colab_t4_modular.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Generated {path.name}: {len(notebook["cells"])} code cells, 0 text cells')


if __name__=='__main__':
    build()
