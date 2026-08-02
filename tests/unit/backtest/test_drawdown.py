from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.drawdown import analyze_drawdowns
from fxbot.backtest.results import EquityPoint

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def point(index: int, equity: float) -> EquityPoint:
    return EquityPoint(
        timestamp=BASE + timedelta(days=index),
        balance=equity,
        equity=equity,
        margin_used=0,
        free_margin=equity,
        unrealized_pnl=0,
        drawdown_amount=0,
        drawdown_fraction=0,
    )


def test_recovered_drawdown_period() -> None:
    analysis = analyze_drawdowns(
        (point(0, 100), point(1, 90), point(2, 80), point(3, 100), point(4, 110))
    )
    assert len(analysis.periods) == 1
    period = analysis.periods[0]
    assert period.started_at == BASE
    assert period.trough_at == BASE + timedelta(days=2)
    assert period.recovered_at == BASE + timedelta(days=3)
    assert period.amount == pytest.approx(20)
    assert period.fraction == pytest.approx(0.2)
    assert period.duration_seconds == pytest.approx(3 * 86_400)
    assert period.recovery_seconds == pytest.approx(86_400)
    assert analysis.unrecovered is False


def test_unrecovered_drawdown_uses_terminal_duration() -> None:
    analysis = analyze_drawdowns((point(0, 100), point(1, 95), point(3, 70)))
    assert analysis.unrecovered is True
    assert analysis.maximum_amount == pytest.approx(30)
    assert analysis.maximum_duration_seconds == pytest.approx(3 * 86_400)
    assert analysis.maximum_recovery_seconds is None


def test_multiple_drawdowns_and_maximum_period() -> None:
    analysis = analyze_drawdowns(
        (
            point(0, 100),
            point(1, 90),
            point(2, 100),
            point(3, 120),
            point(4, 60),
            point(5, 120),
        )
    )
    assert len(analysis.periods) == 2
    assert analysis.maximum_period is not None
    assert analysis.maximum_period.fraction == pytest.approx(0.5)


def test_empty_and_flat_curves_have_no_drawdown() -> None:
    assert analyze_drawdowns(()).periods == ()
    assert analyze_drawdowns((point(0, 100), point(1, 100))).maximum_fraction == 0.0


def test_out_of_order_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="chronologically"):
        analyze_drawdowns((point(1, 100), point(0, 90)))
