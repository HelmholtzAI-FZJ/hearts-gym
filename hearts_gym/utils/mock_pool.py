"""
Mock objects for a single-process `multiprocessing.ThreadPool`.
"""

from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Iterable, List, Optional


class MockResult:
    """'Asynchronous' result for `async` method versions.

    Has to be fetched using the `get` method.
    """

    __slots__ = ['_x']

    def __init__(self, x: Any) -> None:
        """Construct a new mock result containing the given value as
        its result.
        """
        self._x = x

    def get(self, timeout: Optional[int] = None) -> Any:
        """Return the result value contained in this.

        Args:
            timeout (Optional[int]): Ignored.

        Returns:
            Any: Stored result.
        """
        return self._x


class MockPool(ThreadPool):
    """Single-process pool that sequentially maps over its shards."""

    def __init__(self) -> None:
        """Overridden with NOP for API compatibility."""
        pass

    def __del__(self) -> None:
        """Overridden with NOP for API compatibility."""
        pass

    def terminate(self) -> None:
        """Overridden with NOP for API compatibility."""
        pass

    def map_async(  # type: ignore[override]
            self,
            func: Callable,
            iterable: Iterable,
            chunksize: int = None,
    ) -> MockResult:
        return MockResult([func(elem) for elem in iterable])

    def map(
            self,
            func: Callable,
            iterable: Iterable,
            chunksize: int = None,
    ) -> List:
        return self.map_async(func, iterable, chunksize).get()

    def starmap_async(  # type: ignore[override]
            self,
            func: Callable,
            iterable: Iterable,
            chunksize: int = None,
    ) -> MockResult:
        return MockResult([func(*elems) for elems in iterable])

    def starmap(
            self,
            func: Callable,
            iterable: Iterable,
            chunksize: int = None,
    ) -> List:
        return self.starmap_async(func, iterable, chunksize).get()
