from datetime import datetime, timedelta

SECONDS_PER_DAY =  86400
SECONDS_PER_HOUR = 3600

class Timer:
    def __init__(self, window_size: int = 10) -> None:
        self.times = []
        self.window_size = window_size
        self.start_time = datetime.now()

    def step(self):
        if len(self.times) >= self.window_size:
            self.times = self.times[1:]
        self.times.append(datetime.now())

    def strformat(self, timedelta: timedelta):
        days, remainder = divmod(timedelta.total_seconds(), SECONDS_PER_DAY)
        hours, remainder = divmod(remainder, SECONDS_PER_HOUR)
        minutes, seconds = divmod(remainder, 60)
        result = ""

        if int(days) > 0:
            result += '{}d, '.format(int(days))
        if int(hours) > 0:
            result += '{}h, '.format(int(hours))
        if int(minutes) > 0:
            result += '{}min, '.format(int(minutes))

        result += '{}s'.format(int(seconds))

        return result

    def get_eta(self, remaining_steps: int):
        if len(self.times) < 2: return "tbd"
        time_delta_sum = None

        for t in range(len(self.times) - 1):
            if time_delta_sum is None:
                time_delta_sum = self.times[t+1] - self.times[t]
            else:
                time_delta_sum += self.times[t+1] - self.times[t]

        avg_delta = time_delta_sum / len(self.times)
        return self.strformat(avg_delta * remaining_steps)

    def get_elapsed(self):
        return self.strformat(datetime.now() - self.start_time)
