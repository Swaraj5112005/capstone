"""Focused tests for the temporal decision layer."""

import unittest

from temporal import DrowsinessMonitor


class TemporalMonitorTests(unittest.TestCase):
    """Verify required closure and yawn decisions."""

    def test_blink_is_ignored(self) -> None:
        monitor = DrowsinessMonitor()
        monitor.update(0.0, False, False)
        monitor.update(0.2, True, False)
        result = monitor.update(0.4, False, False)
        self.assertEqual(result.state, "OK")
        self.assertEqual(result.closure_s, 0.0)

    def test_two_second_closure_alerts(self) -> None:
        monitor = DrowsinessMonitor()
        monitor.update(0.0, False, False)
        monitor.update(0.1, True, False)
        result = monitor.update(2.1, True, False)
        self.assertEqual(result.state, "ALERT")
        self.assertIn("continuous eye closure", result.reasons)

    def test_two_sustained_yawns_alert(self) -> None:
        monitor = DrowsinessMonitor()
        monitor.update(0.0, False, True)
        monitor.update(1.1, False, True)
        monitor.update(1.2, False, False)
        monitor.update(3.0, False, True)
        result = monitor.update(4.1, False, True)
        self.assertEqual(result.state, "ALERT")
        self.assertEqual(result.yawns, 2)

    def test_one_sustained_yawn_is_caution(self) -> None:
        monitor = DrowsinessMonitor()
        monitor.update(0.0, False, True)
        result = monitor.update(1.1, False, True)
        self.assertEqual(result.state, "CAUTION")
        self.assertEqual(result.yawns, 1)


if __name__ == "__main__":
    unittest.main()

