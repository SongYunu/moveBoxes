"""Generate the standalone next-pick notebook, preserving previous .ipynb files."""
import json
from build_train_test_notebook import ROOT, sources as train_test_sources
from colab_next_pick_layout import make_notebook


def evaluator_source(source):
    anchor = '    from colab_policy import ChunkPolicy'
    if source.count(anchor) != 1:
        raise ValueError('Evaluator import layout changed.')
    source = source.replace(anchor, anchor+'\n    from next_pick_diagnostics import NextPickObserver')
    anchor = "        if not job.get('video_only'):"
    if source.count(anchor) != 1:
        raise ValueError('Evaluator agent layout changed.')
    source = source.replace(anchor, '        agent = NextPickObserver(agent)\n'+anchor)
    anchor = '                rows.append(dict(seed=seed,**result))'
    if source.count(anchor) != 1:
        raise ValueError('Evaluator result layout changed.')
    source = source.replace(anchor, '                result["next_pick"] = agent.report()\n'+anchor)
    return source


def sources():
    result = train_test_sources()
    for name in ('next_pick_sampling.py', 'next_pick_diagnostics.py', 'marso_next_pick.py',
                 'colab_next_pick_layout.py'):
        result[name] = (ROOT/name).read_text(encoding='utf-8')
    result['colab_eval_modular.py'] = evaluator_source(result['colab_eval_modular.py'])
    result['test_policy_runtime.py'] += '\n'+(ROOT/'test_next_pick_runtime.py').read_text(encoding='utf-8')
    return result


def build():
    bundle = sources()
    notebook = make_notebook(bundle)
    for name, source in bundle.items():
        compile(source, name, 'exec')
    for i, cell in enumerate(notebook['cells']):
        assert cell['cell_type'] == 'code'
        compile(''.join(cell['source']), f'cell-{i}', 'exec')
    path = ROOT/'marso_colab_t4_next_pick.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{path.name}: {len(notebook["cells"])} code cells, 0 markdown cells')


if __name__ == '__main__':
    build()
