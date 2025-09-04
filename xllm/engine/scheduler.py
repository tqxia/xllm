from collections import deque
from typing import List, Tuple

from xllm.config import Config
from xllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add_seq(self, seq: Sequence):
        self.waiting.append(seq)

    def remove_seq(self, seq: Sequence):
        assert seq in self.running
        self.running.remove(seq)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule(self) -> Tuple[List[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0

        while len(self.waiting) > 0 and num_seqs < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)
            num_seqs += 1

        if len(scheduled_seqs) > 0:
            return scheduled_seqs, True
        
        while len(self.running) > 0 and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            scheduled_seqs.append(seq)
            num_seqs += 1

        running = deque(scheduled_seqs)
        running.extend(self.running)
        self.running = running

        return scheduled_seqs, False
