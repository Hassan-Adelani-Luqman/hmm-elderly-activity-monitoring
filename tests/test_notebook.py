"""The notebook must define the same HMM that the pipeline scripts import."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookSyncTests(unittest.TestCase):
    def test_notebook_hmm_cell_equals_scripts_hmm(self):
        cells = json.loads((ROOT/'notebooks/hmm_activity_recognition.ipynb').read_text())['cells']
        sources = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
        matches = [s for s in sources if 'class GaussianHMM' in s]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], (ROOT/'scripts/hmm.py').read_text().rstrip('\n'))


if __name__ == '__main__':
    unittest.main()
