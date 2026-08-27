from enum import Enum

class Role(Enum):
    PAYER = 'payer'
    RECEIVER = 'receiver'
    SETTLED = 'settled'

    @classmethod
    def of(cls, balance: int) -> 'Role':
        if balance < 0:
            return cls.PAYER

        if balance > 0:
            return cls.RECEIVER

        return cls.SETTLED

    def color(self) -> str:
        if self is Role.PAYER:
            return '#ffd6d6'

        if self is Role.RECEIVER:
            return '#d6f0d6'

        return '#e8e8e8'
