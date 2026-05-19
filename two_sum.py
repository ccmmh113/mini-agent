"""
两数之和 (Two Sum)

给定一个整数数组 nums 和一个整数目标值 target，
请你在该数组中找出和为目标值 target 的那两个整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且不能使用两次相同的元素。
你可以按任意顺序返回答案。
"""

from typing import List, Optional


def two_sum(nums: List[int], target: int) -> Optional[List[int]]:
    """
    使用哈希表在 O(n) 时间复杂度内找到两数之和为目标值的下标。

    Args:
        nums: 整数数组
        target: 目标值

    Returns:
        两个下标组成的列表，如果没有找到则返回 None
    """
    # 哈希表：值 -> 下标
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return None


if __name__ == "__main__":
    # 示例测试
    test_cases = [
        ([2, 7, 11, 15], 9),    # 期望: [0, 1]
        ([3, 2, 4], 6),          # 期望: [1, 2]
        ([3, 3], 6),             # 期望: [0, 1]
    ]

    for nums, target in test_cases:
        result = two_sum(nums, target)
        print(f"nums = {nums}, target = {target} -> {result}")
