from threading import Thread
from queue import Queue
from loguru import logger


class ThreadWrapper(Thread):
    """线程封装
    会初始化一个队列，这个队列会传递给目标函数，要求目标函数接受一个queue参数
    """
    def __init__(
        self,
        target_func: callable,
        queue_maxsize: int=0,
        *args,
        **kwargs
    ):
        super().__init__(target=target_func, args=args, kwargs=kwargs, daemon=True)

        self.args = args
        self.kwargs = kwargs

        # 初始化队列
        self.queue = Queue(queue_maxsize)
        # 将队列传递给目标函数
        self.kwargs['queue'] = self.queue
        self.target_func = target_func

    def run(self):
        # 可以在这里添加一些线程启动前的初始化代码
        logger.info(f"Thread {self.name} starting with target function {self.target_func.__name__}")
        self.target_func(*self.args, **self.kwargs)
        # 可以在这里添加一些线程结束后的清理代码


def example():
    import time


    # 使用示例, 要求目标函数接受一个queue参数
    def my_function(x, y, queue: Queue, *args, **kwargs):
        print(f"Function is running with arguments {x} and {y}")
        for i in range(10):
            queue.put(i)
            print(f"Put {i} into queue")
            time.sleep(1)
        queue.put(None)
        print(f"Function has finished")


    # 创建ThreadWrapper的实例，传递my_function作为目标函数
    thread = ThreadWrapper(my_function, 1, 2)

    thread_queue = thread.queue

    # 启动线程
    thread.start()

    while True:
        item = thread_queue.get()
        if item is None:
            break
        print(f"Got {item} from queue")


if __name__ == "__main__":
    example()
