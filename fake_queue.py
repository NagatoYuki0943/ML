from loguru import logger


class FakeQueue:
    def __init__(self, *args, **kwargs):
        ...

    def put(self, item, *args, **kwargs):
        logger.debug(f"fake queue received item: {item}")

    def get(self, *args, **kwargs):
        return None

    def qsize(self, *args, **kwargs):
        return 0

    def empty(self, *args, **kwargs):
        return True

    def full(self, *args, **kwargs):
        return False

    def join(self, *args, **kwargs):
        ...

    def task_done(self, *args, **kwargs):
        ...
