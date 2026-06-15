from datetime import datetime, timedelta
from unittest import TestCase
from zoneinfo import ZoneInfo

from gdhprediction.weather import WeatherRequestData

# Berlin — a valid coordinate so timezonefinder resolves a tz.
LAT, LON = 52.52, 13.41
BERLIN = ZoneInfo("Europe/Berlin")
TOKYO = ZoneInfo("Asia/Tokyo")


def _make(start: datetime | None = None, duration: int = 10) -> WeatherRequestData:
    return WeatherRequestData(LAT, LON, start, duration)


class TestWeatherRequestDataSnapping(TestCase):
    """Tests :class:`WeatherRequestData` construction.

    WeatherRequestData.__post_init__ snaps the start hour down to the
    nearest SNAPPED=6h boundary in the GPS-derived timezone, then writes
    `date` and `last_date` as ISO strings spanning `duration` days.
    """

    def test_hour_snaps_down_to_nearest_six(self) -> None:
        # (hour // 6) * 6 produces buckets at 0, 6, 12, 18.
        for hour, expected in [
            (0, 0),
            (5, 0),
            (6, 6),
            (11, 6),
            (12, 12),
            (14, 12),  # the off-boundary case used elsewhere in the suite
            (17, 12),
            (18, 18),
            (23, 18),
        ]:
            with self.subTest(hour=hour):
                rd = _make(datetime(2024, 6, 15, hour, 30))
                start = datetime.fromisoformat(rd.date)
                self.assertEqual(start.hour, expected)
                # Sub-hour fields are zeroed.
                self.assertEqual(
                    (start.minute, start.second, start.microsecond),
                    (0, 0, 0),
                )

    def test_naive_input_is_stamped_with_gps_tz(self) -> None:
        # 14:30 naive → 12:00 Berlin (snapped). Naive input must NOT be
        # converted, only stamped. Note: ISO round-trip yields a fixed-
        # offset tzinfo, so we compare utcoffset() rather than identity.
        rd = _make(datetime(2024, 6, 15, 14, 30))
        start = datetime.fromisoformat(rd.date)
        expected_offset = datetime(2024, 6, 15, 12, tzinfo=BERLIN).utcoffset()
        self.assertEqual(start.utcoffset(), expected_offset)
        self.assertEqual(
            (start.year, start.month, start.day, start.hour),
            (2024, 6, 15, 12),
        )

    def test_aware_input_is_converted_then_snapped(self) -> None:
        # 14:30 Tokyo (UTC+9) → 07:30 Berlin (CEST, UTC+2) → snapped to 06:00.
        aware = datetime(2024, 6, 15, 14, 30, tzinfo=TOKYO)
        rd = _make(aware)
        start = datetime.fromisoformat(rd.date)
        # Local hour after conversion + snapping.
        self.assertEqual(start.hour, 6)
        # Same instant on the timeline as the snapped Berlin equivalent.
        expected = aware.astimezone(BERLIN).replace(
            hour=6,
            minute=0,
            second=0,
            microsecond=0,
        )
        self.assertEqual(start, expected)

    def test_last_date_is_exactly_duration_days_after_start(self) -> None:
        rd = _make(datetime(2024, 6, 15, 14, 30), duration=7)
        start = datetime.fromisoformat(rd.date)
        end = datetime.fromisoformat(rd.last_date)
        self.assertEqual(end - start, timedelta(days=7))

    def test_default_duration_is_ten_days(self) -> None:
        rd = _make(datetime(2024, 6, 15, 12, 0))
        start = datetime.fromisoformat(rd.date)
        end = datetime.fromisoformat(rd.last_date)
        self.assertEqual(end - start, timedelta(days=10))

    def test_none_start_uses_now_in_gps_tz(self) -> None:
        # Hard to assert exact "now", but the result must be tz-aware in
        # the GPS tz (offset-wise) and snapped to a 6h boundary within
        # the current bucket.
        before = datetime.now(BERLIN)
        rd = _make(None)
        after = datetime.now(BERLIN)
        start = datetime.fromisoformat(rd.date)
        # The ISO round-trip yields a fixed-offset tzinfo, so compare
        # utcoffset() against Berlin's offset at this moment.
        self.assertEqual(start.utcoffset(), before.utcoffset())
        # `start` should sit in the snapped-down bucket of `before` (or
        # `after` if the clock crossed a SNAPPED boundary during the call).
        snap = WeatherRequestData.SNAPPED
        floor_before = before.replace(
            hour=(before.hour // snap) * snap,
            minute=0,
            second=0,
            microsecond=0,
        )
        floor_after = after.replace(
            hour=(after.hour // snap) * snap,
            minute=0,
            second=0,
            microsecond=0,
        )
        self.assertGreaterEqual(start, floor_before)
        self.assertLessEqual(start, floor_after)
