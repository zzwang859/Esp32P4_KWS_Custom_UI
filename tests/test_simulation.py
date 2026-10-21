"""PC simulation control ranges."""

from kws_custom_ui.simulation import cooldown_from_slider


def test_cooldown_slider_has_device_minimum_and_ten_ms_steps():
    assert [cooldown_from_slider(position) for position in (0, 1, 50, 100)] == [1, 10, 500, 1000]
