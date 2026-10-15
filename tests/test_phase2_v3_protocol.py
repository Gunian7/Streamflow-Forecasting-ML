import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import phase2_v3_config as cfg


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_frozen(self):
        p = cfg.load_protocol(write=False)
        self.assertEqual(p['test_start'], '2004-01-01')
        self.assertEqual(p['max_horizon'], 7)
        self.assertFalse(p['include_current_discharge'])
        self.assertTrue(p['native_multioutput'])
        self.assertEqual(p['seeds'], [17, 42, 73])

    def test_protocol_write_is_idempotent(self):
        first = cfg.load_protocol(write=True)
        second = cfg.load_protocol(write=True)
        self.assertEqual(first, second)
        self.assertTrue(cfg.PROTOCOL_PATH.exists())

    def test_candidate_key_covers_task_and_params(self):
        data = cfg.ROOT / '01047000.csv'
        a = cfg.candidate_key(data, 1, {'a': 1}, 42)
        b = cfg.candidate_key(data, 1, {'a': 1}, 42)
        self.assertEqual(a, b)
        self.assertNotEqual(a, cfg.candidate_key(data, 2, {'a': 1}, 42))
        self.assertNotEqual(a, cfg.candidate_key(data, 1, {'a': 2}, 42))
        self.assertNotEqual(a, cfg.candidate_key(data, 1, {'a': 1}, 43))


if __name__ == '__main__':
    unittest.main()
