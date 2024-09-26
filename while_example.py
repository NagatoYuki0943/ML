import random


i = 0  # while 循环计数
total_cycle_loop_count = 0  # 总的周期内循环次数
cycle_loop_count = 0  # 周期内循环计数


while True:
    print(f"{i = }")
    has_pic = random.randint(0, 1)

    if cycle_loop_count == 0:  # 每个周期的第一次循环
        if has_pic:
            total_cycle_loop_count = random.randint(
                2, 10
            )  # 随机生成2到10之间的整数作为本周期的循环次数
            print(f"本周期将进行 {total_cycle_loop_count} 次循环")

            cycle_loop_count += 1
            print(f"  周期内的第 {cycle_loop_count} 次循环")

    else:
        if has_pic:
            cycle_loop_count += 1
            print(f"  周期内的第 {cycle_loop_count} 次循环")

            if cycle_loop_count == total_cycle_loop_count:
                cycle_loop_count = 0  # 重置周期内循环计数
                if input("本周期结束。是否继续下一个周期? (y/n): ").lower() != "y":
                    break
    i += 1
