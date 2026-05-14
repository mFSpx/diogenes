from pypeline.math.algos.doomsday_calendar import doomsday


def test_doomsday_returns_sunday_for_2026_05_03():
    assert doomsday(2026, 5, 3) == 0


def test_doomsday_returns_monday_for_2024_01_01():
    assert doomsday(2024, 1, 1) == 1


def test_doomsday_returns_tuesday_for_2000_02_29():
    assert doomsday(2000, 2, 29) == 2


def test_doomsday_returns_thursday_for_2024_02_29():
    assert doomsday(2024, 2, 29) == 4
