"""Phase-two v3 protocol: frozen, versioned, and hashed."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = {
    'test_start': '2004-01-01',
    'max_horizon': 7,
    'include_current_discharge': False,
    'native_multioutput': True,
    'seeds': [17, 42, 73],
    'primary_metric': 'rmse_cfs',
    'development_years': [2000, 2001, 2002, 2003],
    'features_excluded': ['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear',
                          'Swe', 'Discharge'],
}
PROTOCOL_PATH = ROOT / 'results/phase2_v3/protocol.json'


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode('utf-8')).hexdigest()


def load_protocol(write=True):
    """Return the frozen protocol; refuse to silently change an existing one."""
    current = dict(PROTOCOL)
    current['code_digest'] = _digest({k: v for k, v in current.items()
                                      if k not in ('code_digest',)})
    if PROTOCOL_PATH.exists():
        saved = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
        if saved != current:
            raise ValueError('protocol.json 与当前协议不一致，请确认是否应新开实验目录')
        return saved
    if write:
        PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROTOCOL_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                                 encoding='utf-8')
    return current


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_key(data_path, horizon, params, seed, extra=None):
    """Cache key covering data, task, full params, seed and environment."""
    import sklearn
    payload = {
        'protocol': _digest(PROTOCOL),
        'data': file_digest(data_path),
        'horizon': horizon,
        'params': params,
        'seed': seed,
        'sklearn': sklearn.__version__,
        'python': sys.version.split()[0],
    }
    if extra:
        payload['extra'] = extra
    return _digest(payload)


if __name__ == '__main__':
    print(json.dumps(load_protocol(), ensure_ascii=False, indent=2))
