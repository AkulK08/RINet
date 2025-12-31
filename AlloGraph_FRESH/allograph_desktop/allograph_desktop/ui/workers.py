from __future__ import annotations
from PySide6.QtCore import QObject, Signal, QRunnable, Slot

class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)

class FunctionWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.error.emit(str(e))
