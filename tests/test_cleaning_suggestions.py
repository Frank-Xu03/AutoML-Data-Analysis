import os
import importlib.util


def import_clean_func():
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, 'automl-llm', 'app', 'llm_agent.py')
    spec = importlib.util.spec_from_file_location('llm_agent', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.suggest_cleaning_suggestions


def test_cleaning_suggestions_offline():
    os.environ['LLM_OFFLINE'] = '1'

    profile = {
        "shape": {"rows": 100, "cols": 7},
        "columns": [
            {"name": "id", "dtype": "int64", "nunique": 100, "missing": 0},
            {"name": "age", "dtype": "int64", "nunique": 70, "missing": 5},
            {"name": "income", "dtype": "float64", "nunique": 90, "missing": 10},
            {"name": "signup_date", "dtype": "object", "nunique": 80, "missing": 0},
            {"name": "gender", "dtype": "object", "nunique": 2, "missing": 1},
            {"name": "all_same", "dtype": "int64", "nunique": 1, "missing": 0},
            {"name": "free_text", "dtype": "object", "nunique": 95, "missing": 0},
        ]
    }

    suggest = import_clean_func()
    out = suggest(profile, user_goal="预测是否购买")

    assert isinstance(out, dict)
    assert 'drop_columns' in out
    assert 'imputations' in out
    assert 'parse_dates' in out
    # Expect id/all_same in drops
    dropped = {d.get('name') for d in out['drop_columns']}
    assert 'id' in dropped or 'all_same' in dropped
    # Expect signup_date in parse_dates due to name hint
    assert 'signup_date' in (out.get('parse_dates') or [])
