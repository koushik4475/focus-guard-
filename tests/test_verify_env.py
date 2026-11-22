import unittest
from unittest.mock import patch

import verify_env


class CheckDependenciesTests(unittest.TestCase):
    def test_reports_missing_imports(self):
        def importer(module_name):
            if module_name == "pygame":
                raise ImportError("pygame is not installed")
            return object()

        with patch("builtins.print"):
            failures = verify_env.check_dependencies(importer)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "pygame")
        self.assertIn("not installed", failures[0][1])

    def test_returns_no_failures_when_all_imports_work(self):
        with patch("builtins.print"):
            failures = verify_env.check_dependencies(lambda _name: object())

        self.assertEqual(failures, [])

    def test_main_returns_nonzero_when_dependency_is_missing(self):
        with patch.object(verify_env, "check_dependencies", return_value=[("pygame", "missing")]):
            with patch("builtins.print"):
                self.assertEqual(verify_env.main(), 1)

    def test_main_returns_zero_when_all_dependencies_are_available(self):
        with patch.object(verify_env, "check_dependencies", return_value=[]):
            with patch("builtins.print"):
                self.assertEqual(verify_env.main(), 0)


if __name__ == "__main__":
    unittest.main()
