"""Bound Torch CPU threads for small offline models."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    # 微型模型用单线程减少调度开销，测试结束后恢复原线程设置。
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
