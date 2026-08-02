"""Deterministic Smart-Money-Concepts strategy composed from structure primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from fxbot.domain.enums import Timeframe
from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.filters import (
    MarketFilter,
    StrategyFilterConfig,
    clamp_confidence,
    format_metadata,
)
from fxbot.strategy.imbalances import (
    FairValueGap,
    ImbalanceConfig,
    detect_fair_value_gaps,
)
from fxbot.strategy.indicators import IndicatorConfig, IndicatorError, calculate_indicators
from fxbot.strategy.liquidity import (
    LiquidityConfig,
    LiquidityPool,
    LiquiditySweep,
    detect_liquidity_pools,
    detect_liquidity_sweeps,
    nearest_liquidity_target,
)
from fxbot.strategy.market_structure import (
    DealingRangeZone,
    MarketStructureConfig,
    MarketStructureState,
    StructureDirection,
    analyze_market_structure,
    structure_confluence,
)
from fxbot.strategy.models import (
    MarketRegime,
    SignalAction,
    StrategyConfig,
    StrategyDecision,
)
from fxbot.strategy.order_blocks import (
    OrderBlock,
    OrderBlockConfig,
    detect_order_blocks,
)
from fxbot.strategy.regime import RegimeClassifier


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _default_filters() -> StrategyFilterConfig:
    return StrategyFilterConfig(
        blocked_regimes=(MarketRegime.ILLIQUID,),
        max_spread_to_atr=0.25,
        max_atr_fraction=0.04,
    )


@dataclass(frozen=True, slots=True)
class SMCStrategyConfig:
    """Composition and risk parameters for the SMC strategy."""

    strategy: StrategyConfig
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    filters: StrategyFilterConfig = field(default_factory=_default_filters)
    structure: MarketStructureConfig = field(default_factory=MarketStructureConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    order_blocks: OrderBlockConfig = field(default_factory=OrderBlockConfig)
    imbalances: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    signal_age_bars: int = 2
    stop_buffer_atr: float = 0.20
    target_reward_to_risk: float = 2.0
    minimum_structure_alignment: float = 0.50
    require_structure_shift: bool = True
    require_displacement: bool = True
    require_zone_confluence: bool = False

    def __post_init__(self) -> None:
        if self.signal_age_bars < 0:
            raise ValueError("signal_age_bars cannot be negative")
        object.__setattr__(
            self,
            "stop_buffer_atr",
            _non_negative(self.stop_buffer_atr, "stop_buffer_atr"),
        )
        object.__setattr__(
            self,
            "target_reward_to_risk",
            _positive(self.target_reward_to_risk, "target_reward_to_risk"),
        )
        alignment = float(self.minimum_structure_alignment)
        if not isfinite(alignment) or not 0.0 <= alignment <= 1.0:
            raise ValueError("minimum_structure_alignment must be between 0 and 1")
        object.__setattr__(self, "minimum_structure_alignment", alignment)


class SMCStrategy:
    """Trade recent liquidity sweeps confirmed by structure and displacement."""

    def __init__(
        self,
        settings: SMCStrategyConfig,
        *,
        regime_classifier: RegimeClassifier | None = None,
    ) -> None:
        self.settings = settings
        self.regime_classifier = regime_classifier or RegimeClassifier()
        self.market_filter = MarketFilter(settings.filters)

    @property
    def config(self) -> StrategyConfig:
        return self.settings.strategy

    def evaluate(self, context: MultiTimeframeContext) -> StrategyDecision:
        primary = context.primary
        try:
            indicators = calculate_indicators(primary, self.settings.indicators)
            regime = self.regime_classifier.assess(primary)
        except (IndicatorError, KeyError, ValueError) as exc:
            return self._hold(
                context,
                "smc_indicator_or_regime_unavailable",
                metadata=(("error", str(exc)),),
            )

        atr = indicators.value("atr")
        state = analyze_market_structure(
            primary,
            atr=atr,
            config=self.settings.structure,
        )
        pools = detect_liquidity_pools(
            primary,
            state.swings,
            atr=atr,
            config=self.settings.liquidity,
        )
        sweeps = detect_liquidity_sweeps(
            primary,
            pools,
            atr=atr,
            config=self.settings.liquidity,
        )
        blocks = detect_order_blocks(
            primary,
            state.events,
            atr=atr,
            config=self.settings.order_blocks,
        )
        gaps = detect_fair_value_gaps(
            primary,
            atr=atr,
            config=self.settings.imbalances,
        )

        latest_index = len(primary.bars) - 1
        sweep = self._recent_sweep(sweeps, latest_index)
        direction = sweep.direction if sweep is not None else StructureDirection.NEUTRAL
        action = _action(direction)

        confluence_score = 1.0
        confluence_bias = direction
        if len(self.config.required_timeframes) > 1:
            atr_by_timeframe: dict[Timeframe, float] = {}
            try:
                for timeframe in self.config.required_timeframes:
                    atr_by_timeframe[timeframe] = calculate_indicators(
                        context.get(timeframe),
                        self.settings.indicators,
                    ).value("atr")
                confluence = structure_confluence(
                    context,
                    atr_by_timeframe=atr_by_timeframe,
                    config=self.settings.structure,
                    timeframes=self.config.required_timeframes,
                )
                confluence_score = confluence.alignment_score
                confluence_bias = confluence.dominant_bias
            except (IndicatorError, KeyError, ValueError) as exc:
                return self._hold(
                    context,
                    "smc_higher_timeframe_structure_unavailable",
                    regime=regime.regime,
                    metadata=(("error", str(exc)),),
                )

        filter_result = self.market_filter.evaluate(
            context=context,
            indicators=indicators,
            regime=regime,
            expected_action=action,
        )
        if not filter_result.allowed:
            return self._hold(
                context,
                "smc_market_filter_rejected",
                regime=regime.regime,
                metadata=tuple(
                    (f"filter_{index}", reason)
                    for index, reason in enumerate(filter_result.reasons)
                ),
            )
        if sweep is None or action is None:
            return self._hold(
                context,
                "smc_recent_liquidity_sweep_missing",
                regime=regime.regime,
                metadata=_audit_metadata(state, pools, sweeps, blocks, gaps),
            )

        recent_event = state.latest_event
        structure_ok = (
            recent_event is not None
            and recent_event.direction is direction
            and recent_event.index >= latest_index - self.settings.signal_age_bars
        )
        displacement = state.latest_displacement
        displacement_ok = (
            displacement is not None
            and displacement.direction is direction
            and displacement.index >= latest_index - self.settings.signal_age_bars
        )
        zone_ok = _zone_confluence(primary.latest.mid.low, primary.latest.mid.high, direction, blocks, gaps)
        dealing_zone_ok = (
            state.price_zone in {DealingRangeZone.DISCOUNT, DealingRangeZone.UNKNOWN}
            if direction is StructureDirection.BULLISH
            else state.price_zone in {DealingRangeZone.PREMIUM, DealingRangeZone.UNKNOWN}
        )
        higher_alignment_ok = (
            confluence_score >= self.settings.minimum_structure_alignment
            and confluence_bias in {direction, StructureDirection.NEUTRAL}
        )

        failures: list[str] = []
        if self.settings.require_structure_shift and not structure_ok:
            failures.append("structure_shift_missing")
        if self.settings.require_displacement and not displacement_ok:
            failures.append("displacement_missing")
        if self.settings.require_zone_confluence and not zone_ok:
            failures.append("order_block_or_fvg_confluence_missing")
        if not dealing_zone_ok:
            failures.append("dealing_range_location_mismatch")
        if not higher_alignment_ok:
            failures.append("higher_timeframe_structure_mismatch")
        if failures:
            return self._hold(
                context,
                "smc_confirmation_rejected",
                regime=regime.regime,
                metadata=tuple((f"failure_{index}", item) for index, item in enumerate(failures)),
            )

        entry = primary.latest.mid.close
        if direction is StructureDirection.BULLISH:
            stop = sweep.extreme_price - atr * self.settings.stop_buffer_atr
            risk = entry - stop
        else:
            stop = sweep.extreme_price + atr * self.settings.stop_buffer_atr
            risk = stop - entry
        if risk <= 0.0 or stop <= 0.0:
            return self._hold(context, "smc_invalid_stop_geometry", regime=regime.regime)

        target_pool = nearest_liquidity_target(pools, direction=direction, price=entry)
        reward_target = (
            entry + risk * self.settings.target_reward_to_risk
            if direction is StructureDirection.BULLISH
            else entry - risk * self.settings.target_reward_to_risk
        )
        target = reward_target
        if target_pool is not None:
            pool_reward = (
                target_pool.level - entry
                if direction is StructureDirection.BULLISH
                else entry - target_pool.level
            )
            if pool_reward >= risk * self.settings.target_reward_to_risk:
                target = target_pool.level
        if target <= 0.0:
            return self._hold(context, "smc_invalid_target_geometry", regime=regime.regime)

        penetration_score = min(1.0, sweep.penetration / max(atr * 0.5, 1e-12))
        structure_score = 1.0 if structure_ok else 0.5
        displacement_score = (
            min(1.0, displacement.body_atr / 2.0) if displacement_ok and displacement else 0.5
        )
        zone_score = 1.0 if zone_ok else 0.6
        confidence = clamp_confidence(
            0.30 * penetration_score
            + 0.25 * structure_score
            + 0.20 * displacement_score
            + 0.15 * confluence_score
            + 0.10 * zone_score
        )
        confidence *= filter_result.confidence_multiplier

        return StrategyDecision(
            strategy_id=self.config.strategy_id,
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            action=action,
            as_of=context.as_of,
            confidence=clamp_confidence(confidence),
            reasons=(
                "liquidity_pool_swept_and_reclaimed",
                "market_structure_confirmed",
                "displacement_confirmed",
                "dealing_range_and_zone_filters_passed",
            ),
            regime=regime.regime,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            metadata=format_metadata(
                {
                    "atr": atr,
                    "confluence_score": confluence_score,
                    "fair_value_gap_count": len(gaps),
                    "liquidity_pool_count": len(pools),
                    "order_block_count": len(blocks),
                    "sweep_penetration_atr": sweep.penetration / max(atr, 1e-12),
                    "target_reward_to_risk": abs(target - entry) / risk,
                }
            ),
        )

    def _recent_sweep(
        self,
        sweeps: tuple[LiquiditySweep, ...],
        latest_index: int,
    ) -> LiquiditySweep | None:
        recent = [
            sweep
            for sweep in sweeps
            if sweep.index >= latest_index - self.settings.signal_age_bars
        ]
        return recent[-1] if recent else None

    def _hold(
        self,
        context: MultiTimeframeContext,
        reason: str,
        *,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> StrategyDecision:
        return StrategyDecision.hold(
            strategy_id=self.config.strategy_id,
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            as_of=context.as_of,
            reason=reason,
            regime=regime,
            metadata=metadata,
        )


def _action(direction: StructureDirection) -> SignalAction | None:
    if direction is StructureDirection.BULLISH:
        return SignalAction.BUY
    if direction is StructureDirection.BEARISH:
        return SignalAction.SELL
    return None


def _zone_confluence(
    low: float,
    high: float,
    direction: StructureDirection,
    blocks: tuple[OrderBlock, ...],
    gaps: tuple[FairValueGap, ...],
) -> bool:
    block_side = direction.value
    block_hit = any(
        block.active and block.side.value == block_side and block.overlaps(low, high)
        for block in blocks
    )
    gap_hit = any(
        gap.active and gap.side.value == block_side and gap.overlaps(low, high)
        for gap in gaps
    )
    return block_hit or gap_hit


def _audit_metadata(
    state: MarketStructureState,
    pools: tuple[LiquidityPool, ...],
    sweeps: tuple[LiquiditySweep, ...],
    blocks: tuple[OrderBlock, ...],
    gaps: tuple[FairValueGap, ...],
) -> tuple[tuple[str, str], ...]:
    return format_metadata(
        {
            "bias": state.bias.value,
            "fair_value_gap_count": len(gaps),
            "liquidity_pool_count": len(pools),
            "liquidity_sweep_count": len(sweeps),
            "order_block_count": len(blocks),
            "swing_count": len(state.swings),
        }
    )
