"""Tests for deterministic scheduling boundaries; not a booking-service test."""
import unittest
from scheduling_core import conflicts, validate_interval


class SchedulingCoreTests(unittest.TestCase):
    def test_adjacent_appointments_do_not_conflict(self):
        existing=[{"start":"2026-11-09T14:00:00Z","end":"2026-11-09T15:00:00Z","resources":["room-1","doctor-1"]}]
        self.assertEqual(conflicts("2026-11-09T15:00:00Z","2026-11-09T16:00:00Z",existing,{"room-1"}),[])
        self.assertEqual(conflicts("2026-11-09T14:30:00Z","2026-11-09T15:30:00Z",existing,{"room-2"}),[])
        self.assertEqual(len(conflicts("2026-11-09T14:30:00Z","2026-11-09T15:30:00Z",existing,{"doctor-1"})),1)

    def test_offsets_normalize_to_utc(self):
        self.assertEqual(validate_interval("2026-11-09T09:00:00-05:00","2026-11-09T10:00:00-05:00"),
                         validate_interval("2026-11-09T14:00:00Z","2026-11-09T15:00:00Z"))

    def test_rejects_ambiguous_or_invalid_time(self):
        for start,end in [("2026-11-09T09:00:00","2026-11-09T10:00:00"),
                          ("2026-11-09T10:00:00Z","2026-11-09T10:00:00Z"),
                          ("2026-11-09T11:00:00Z","2026-11-09T10:00:00Z")]:
            with self.assertRaises(ValueError):validate_interval(start,end)

    def test_fail_closed_on_missing_resources(self):
        with self.assertRaises(ValueError):conflicts("2026-11-09T14:00:00Z","2026-11-09T15:00:00Z",[],set())
        with self.assertRaises(ValueError):
            conflicts("2026-11-09T14:00:00Z","2026-11-09T15:00:00Z",
                      [{"start":"2026-11-09T14:00:00Z","end":"2026-11-09T15:00:00Z","resources":None}],{"room-1"})
