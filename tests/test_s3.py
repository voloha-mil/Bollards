import unittest

from bollards.io.s3 import s3_key


class TestS3Key(unittest.TestCase):
    def test_s3_key_joins_cleanly(self) -> None:
        key = s3_key("runs/", "osv5m_cpu", "w00", "test", "state", "filtered.csv")
        self.assertEqual(key, "runs/osv5m_cpu/w00/test/state/filtered.csv")

    def test_s3_key_normalizes_backslashes(self) -> None:
        key = s3_key("", "a\\b", "c")
        self.assertEqual(key, "a/b/c")


if __name__ == "__main__":
    unittest.main()
