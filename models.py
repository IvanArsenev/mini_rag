"""Models for telegram bot and backend project."""

from aiogram.fsm.state import State, StatesGroup


class UserState(StatesGroup):  # pylint: disable=too-few-public-methods
    """Model for steps of form."""
    waiting_for_request = State()
    waiting_for_file = State()
