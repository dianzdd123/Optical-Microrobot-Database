"""Exercise split contracts with filenames; no real images or PyTorch needed."""
import tempfile
import unittest
from pathlib import Path
from pose_data import load_dataframes


class PoseSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def populate(self, count=10):
        for name in ('P-10_R0', 'P0_R20', 'P10_R20'):
            folder = self.root / name
            folder.mkdir()
            for index in range(count):
                (folder / f'{index:03}.JPG').touch()

    def test_contract_coverage_and_disjoint_splits(self):
        self.populate()
        train, val, test, n_p, n_r = load_dataframes(self.root)
        self.assertEqual((len(train), len(val), len(test)), (18, 6, 6))
        self.assertEqual((n_p, n_r), (3, 2))
        sets = [set(frame.image_path) for frame in (train, val, test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 30)
        for frame in (train, val, test):
            self.assertEqual(frame.attrs['P_map'], {-10: 0, 0: 1, 10: 2})
            self.assertEqual(frame.attrs['R_map'], {0: 0, 20: 1})
            self.assertEqual(frame.pose_class.nunique(), 3)

    def test_repeatable_split_and_sampling_metadata(self):
        self.populate()
        first, second = load_dataframes(self.root), load_dataframes(self.root)
        for left, right in zip(first[:3], second[:3]):
            self.assertEqual(left.image_path.tolist(), right.image_path.tolist())
        sampled = first[0].groupby(['P', 'R']).sample(n=1, random_state=42).reset_index(drop=True)
        self.assertEqual(sampled.attrs, first[0].attrs)

    def test_empty_and_missing_inputs_explain_problem(self):
        with self.assertRaisesRegex(ValueError, 'No pose images'):
            load_dataframes(self.root)
        with self.assertRaisesRegex(FileNotFoundError, 'directory not found'):
            load_dataframes(self.root / 'missing')

    def test_too_few_examples_fail_before_training(self):
        self.populate(4)
        with self.assertRaisesRegex(ValueError, 'at least 5'):
            load_dataframes(self.root)

    def test_non_pose_directories_are_ignored(self):
        self.populate()
        extra = self.root / 'P10_R20_backup'
        extra.mkdir()
        (extra / 'image.jpg').touch()
        self.assertEqual(sum(len(f) for f in load_dataframes(self.root)[:3]), 30)


if __name__ == '__main__':
    unittest.main()
