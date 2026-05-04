"""
Simple decorator that prints a formatted traceback on exception instead of
crashing silently — used as a lightweight dev-mode wrapper around async entry
points.
"""
import asyncio
import functools
import traceback


def traceback_display(fn):
    """Wrap an async (or sync) function so exceptions print a full traceback."""
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception:
                traceback.print_exc()
                raise
        return _async_wrapper

    @functools.wraps(fn)
    def _sync_wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            raise
    return _sync_wrapper
