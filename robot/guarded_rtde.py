import logging
import time


class GuardedRtde:
    def __init__(self, rtde_ctl):
        self.rtde_ctl = rtde_ctl

    def __getattribute__(self, name):
        if name == "rtde_ctl":
            return object.__getattribute__(self, name)

        attr = object.__getattribute__(self.rtde_ctl, name)
        if hasattr(attr, '__call__'):
            def f(*args, **kwargs):
                result = attr(*args, **kwargs)

                while not result:
                    timeout = 0.1
                    logging.error(f"Failed to call '{name}' on the robot, retrying in {timeout}s")
                    time.sleep(timeout)

                    result = attr(*args, **kwargs)

                return result
            return f
        else:
            return attr