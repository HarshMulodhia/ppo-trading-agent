"""
Risk Manager Module

Risk management and trading constraints.
"""

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management and position constraints.

    Features:
    - Position limits
    - Capital constraints
    - Risk limits
    - Order validation
    """

    def __init__(
        self,
        initial_capital: float,
        max_position_size: float = 0.1,
        max_daily_loss: float = 0.05,
        max_leverage: float = 1.0,
    ):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting capital
            max_position_size: Max position as fraction of capital
            max_daily_loss: Max daily loss as fraction
            max_leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_leverage = max_leverage

        self.open_positions = {}
        self.daily_pnl = 0.0

    def check_limits(self, position_size: float, current_price: float) -> bool:
        """
        Check position limits.

        Args:
            position_size: Size of position
            current_price: Current price

        Returns:
            True if within limits
        """
        position_value = position_size * current_price
        max_position_value = self.current_capital * self.max_position_size

        if position_value > max_position_value:
            logger.warning(
                f"Position value {position_value:.2f} exceeds limit {max_position_value:.2f}"
            )
            return False

        return True

    def check_daily_loss(self, current_pnl: float) -> bool:
        """
        Check daily loss limit.

        Args:
            current_pnl: Current daily P&L

        Returns:
            True if within limit
        """
        loss_ratio = abs(current_pnl) / self.initial_capital

        if loss_ratio > self.max_daily_loss:
            logger.warning(
                f"Daily loss {loss_ratio:.2%} exceeds limit {self.max_daily_loss:.2%}"
            )
            return False

        return True

    def adjust_position(self, position: float, max_value: float) -> float:
        """
        Adjust position to respect limits.

        Args:
            position: Desired position
            max_value: Maximum position value

        Returns:
            Adjusted position
        """
        return min(position, max_value / max(1e-8, abs(position)))

    def get_available_capital(self) -> float:
        """Get available capital for trading."""
        return self.current_capital * (self.max_leverage - 1)

    def validate_order(
        self,
        order_size: float,
        order_price: float,
    ) -> bool:
        """
        Validate order.

        Args:
            order_size: Size of order
            order_price: Price of order

        Returns:
            True if valid
        """
        # Check capital
        order_value = order_size * order_price
        if order_value > self.current_capital:
            logger.warning("Order value exceeds available capital")
            return False

        # Check position limits
        if not self.check_limits(order_size, order_price):
            return False

        return True

    def update_capital(self, pnl: float) -> None:
        """Update capital after trade."""
        self.current_capital += pnl
        self.daily_pnl += pnl
        logger.info(f"Capital updated: {self.current_capital:.2f}")
