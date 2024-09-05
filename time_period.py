import time


def start_service():
    print("服务已启动")


def main():
    period = 3000  # 设置时间周期为10秒
    before_time_period = time.time()

    i = 0
    while True:
        current_time_period = time.time()
        _before_time_period = int(before_time_period * 1000 // period)
        _current_time_period = int(current_time_period * 1000 // period)
        if _current_time_period > _before_time_period:
            print(f"{before_time_period = }, {current_time_period = }")
            print(f"{_before_time_period = }, {_current_time_period = }")
            start_service()
            before_time_period = current_time_period

        print(i)
        i += 1
        time.sleep(1)  # 暂停1秒，避免过于频繁的循环


if __name__ == "__main__":
    main()
