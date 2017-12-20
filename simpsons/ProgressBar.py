import sys


class ProgressBar:
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.count = 0

    def incr(self, count: int = 1) -> None:
        self.count += count

    def display(self) -> None:
        msg = self.message()
        msg_len = len(msg)
        sys.stderr.write(("\b" * msg_len) + self.message())
        sys.stderr.flush()

    def message(self) -> str:
        return "%d/%d" % (self.count, self.total_items)