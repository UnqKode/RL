"""Fixed-time traffic signal (SPaT) model with a per-episode random offset."""
from dataclasses import dataclass

GREEN, YELLOW, RED = "green", "yellow", "red"


@dataclass
class SignalState:
    phase: str
    time_to_change: float   # s remaining in current phase
    time_until_green: float  # s until green starts (0 if currently green)


class TrafficSignal:
    """Fixed-time cycle: green -> yellow -> red -> green ...

    A random offset (0, cycle_length) shifts the phase pattern each episode so the
    policy cannot memorize a single fixed schedule.
    """

    def __init__(self, green_dur: float, yellow_dur: float, red_dur: float):
        self.green_dur = green_dur
        self.yellow_dur = yellow_dur
        self.red_dur = red_dur
        self.cycle_len = green_dur + yellow_dur + red_dur
        self.offset = 0.0

    def reset(self, rng, max_offset: float = None):
        high = max_offset if max_offset is not None else self.cycle_len
        self.offset = rng.uniform(0.0, high)

    def _phase_at(self, cycle_time: float):
        """Return (phase, time_since_phase_start) for a time within [0, cycle_len)."""
        if cycle_time < self.green_dur:
            return GREEN, cycle_time
        cycle_time -= self.green_dur
        if cycle_time < self.yellow_dur:
            return YELLOW, cycle_time
        cycle_time -= self.yellow_dur
        return RED, cycle_time

    def state(self, t: float) -> SignalState:
        cycle_time = (t + self.offset) % self.cycle_len
        phase, since_start = self._phase_at(cycle_time)

        if phase == GREEN:
            time_to_change = self.green_dur - since_start
            time_until_green = 0.0
        elif phase == YELLOW:
            time_to_change = self.yellow_dur - since_start
            time_until_green = self.red_dur + time_to_change
        else:  # RED
            time_to_change = self.red_dur - since_start
            time_until_green = time_to_change

        return SignalState(phase=phase, time_to_change=time_to_change,
                            time_until_green=time_until_green)

    def is_red_or_yellow(self, t: float) -> bool:
        return self.state(t).phase in (RED, YELLOW)

    def is_red(self, t: float) -> bool:
        return self.state(t).phase == RED
