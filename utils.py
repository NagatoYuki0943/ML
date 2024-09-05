from queue import Queue


def clear_queue(*queues: Queue):
    queue: Queue
    for queue in queues:
        try:
            # 使用 timeout 参数防止无限等待
            while not queue.empty():
                # 设置超时时间为 1 秒以避免死锁
                queue.get(timeout=1)
        except queue.Empty:
            # 当队列为空时继续处理下一个队列
            continue
        except Exception as e:
            # 处理其他可能的异常
            print(f"An error occurred: {e}")
