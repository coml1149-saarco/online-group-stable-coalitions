"""Runs the property-based checks (tests/check_*.py) as part of the unittest suite."""

import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PropertyChecks(unittest.TestCase):
    def _run(self, name):
        r = subprocess.run([sys.executable, str(HERE / name)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])

    def test_exact_checker(self):
        self._run("check_full_checker.py")

    def test_witness_probing(self):
        self._run("check_witness_properties.py")

    def test_bandit_feedback(self):
        self._run("check_bandit.py")


if __name__ == "__main__":
    unittest.main()
