import os
import importlib.util

def import_suggest_func():
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, 'automl-llm', 'app', 'llm_agent.py')
    spec = importlib.util.spec_from_file_location('llm_agent', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.suggest_target_and_features


def test_target_and_features_offline_heuristic():
    # Force offline to ensure deterministic heuristic path
    os.environ['LLM_OFFLINE'] = '1'

    profile = {
        "shape": {"rows": 100, "cols": 6},
        "columns": [
            {"name": "PassengerId", "dtype": "int64", "nunique": 100, "missing": 0},
            {"name": "Survived", "dtype": "int64", "nunique": 2, "missing": 0},
            {"name": "Pclass", "dtype": "int64", "nunique": 3, "missing": 0},
            {"name": "Name", "dtype": "object", "nunique": 100, "missing": 0},
            {"name": "Sex", "dtype": "object", "nunique": 2, "missing": 0},
            {"name": "AllOnes", "dtype": "int64", "nunique": 1, "missing": 0},
        ]
    }

    suggest = import_suggest_func()
    out = suggest(profile, user_goal="预测是否生还")

    assert isinstance(out, dict)
    # Expect target chosen as 'Survived'
    assert out.get('target') in ("Survived", None)
    # Keep and drop keys exist
    assert isinstance(out.get('keep_columns'), list)
    assert isinstance(out.get('drop_columns'), list)
    # Drop should include PassengerId or AllOnes for heuristic
    dropped_names = {d.get('name') for d in out.get('drop_columns')}
    assert 'AllOnes' in dropped_names or 'PassengerId' in dropped_names
