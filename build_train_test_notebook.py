"""Build the new T4 train/test .ipynb without changing earlier notebooks."""
import json
from build_modular_notebook import ROOT, sources as base_sources
from colab_train_test_layout import make_notebook


def sources():
    result=base_sources()
    for name in ('marso_train_test.py','colab_train_test_layout.py'):
        result[name]=(ROOT/name).read_text(encoding='utf-8')
    return result


def build():
    notebook=make_notebook(sources())
    for i,cell in enumerate(notebook['cells']):
        assert cell['cell_type']=='code'
        compile(''.join(cell['source']),f'cell-{i}','exec')
    path=ROOT/'marso_colab_t4_train_test.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{path.name}: {len(notebook["cells"])} code cells, 0 markdown cells')


if __name__=='__main__':
    build()
