"""Market frictions module modeling transaction costs, bid-ask spread, slippage, and liquidity constraints."""

from dataclasses import dataclass
import numpy as np


@dataclass
class MarketFrictionConfig:
    """Configuration parameters for realistic financial market frictions."""

    transaction_fee_rate: float = 0.0010  # 10 bps (0.10%)
    bid_ask_spread_rate: float = 0.0005  # 5 bps half-spread (0.05%)
    temporary_impact_gamma: float = 0.10  # Almgren-Chriss temporary impact coefficient
    permanent_impact_gamma: float = 0.05  # Almgren-Chriss permanent impact coefficient
    impact_exponent: float = 0.5  # Square-root law of market impact (Barra/Almgren-Chriss)
    max_volume_participation: float = 0.10  # Liquidity limit: max 10% of interval volume


class MarketFrictionModel:
    """Models market frictions, liquidity limits, and slippage based on Almgren-Chriss."""

    def __init__(self, config: MarketFrictionConfig = None):
        self.config = config or MarketFrictionConfig()

    def apply_liquidity_constraint(
        self, target_shares: float, market_volume: float
    ) -> float:
        """Caps order execution to prevent violating max volume participation rate."""
        max_allowed = self.config.max_volume_participation * max(market_volume, 1000.0)
        capped_shares = np.clip(target_shares, -max_allowed, max_allowed)
        return float(capped_shares)

    def calculate_execution(
        self,
        arrival_price: float,
        shares: float,
        market_volume: float,
        volatility: float = 0.20,
    ) -> dict:
        """Calculates execution price, slippage, fees, and implementation shortfall.

        Args:
            arrival_price: Benchmark decision/arrival price P_0.
            shares: Signed share quantity (+ for Buy, - for Sell).
            market_volume: Market volume during execution window.
            volatility: Asset annualized volatility.

        Returns:
            dict containing:
                executed_shares,
                effective_price,
                spread_cost,
                slippage_cost,
                fee_cost,
                total_cost,
                implementation_shortfall
        """
        if abs(shares) < 1e-6 or market_volume <= 0:
            return {
                "executed_shares": 0.0,
                "effective_price": arrival_price,
                "spread_cost": 0.0,
                "slippage_cost": 0.0,
                "fee_cost": 0.0,
                "total_cost": 0.0,
                "implementation_shortfall": 0.0,
            }

        # Liquidity constraint check
        executed_shares = self.apply_liquidity_constraint(shares, market_volume)
        order_direction = np.sign(executed_shares)
        abs_shares = abs(executed_shares)

        # 1. Bid-Ask Spread Cost (half spread)
        spread_cost_per_share = arrival_price * self.config.bid_ask_spread_rate

        # 2. Almgren-Chriss Market Impact / Slippage
        # Participation rate
        participation_rate = abs_shares / max(market_volume, 1.0)
        # Non-linear impact (square root law with volatility scaling)
        daily_vol = volatility / np.sqrt(252.0)
        impact_pct = (
            self.config.temporary_impact_gamma
            * daily_vol
            * (participation_rate ** self.config.impact_exponent)
        )
        slippage_cost_per_share = arrival_price * impact_pct

        # Effective execution price experienced by trader
        # Buying pushes price UP (+), Selling pushes price DOWN (-)
        price_shift = order_direction * (spread_cost_per_share + slippage_cost_per_share)
        effective_price = arrival_price + price_shift

        # Dollar costs
        spread_cost = spread_cost_per_share * abs_shares
        slippage_cost = slippage_cost_per_share * abs_shares
        trade_value = effective_price * abs_shares
        fee_cost = trade_value * self.config.transaction_fee_rate
        total_cost = spread_cost + slippage_cost + fee_cost

        # Implementation Shortfall = |Effective Cost - Benchmark Cost| + fees
        benchmark_value = arrival_price * executed_shares
        actual_value = effective_price * executed_shares
        implementation_shortfall = (
            abs(actual_value - benchmark_value) + fee_cost
        )

        return {
            "executed_shares": executed_shares,
            "effective_price": effective_price,
            "spread_cost": spread_cost,
            "slippage_cost": slippage_cost,
            "fee_cost": fee_cost,
            "total_cost": total_cost,
            "implementation_shortfall": implementation_shortfall,
        }
