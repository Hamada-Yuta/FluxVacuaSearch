# --- logger_setup.py ---
import logging, sys, multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener

FMT = '[%(asctime)s %(levelname)s %(processName)s] %(message)s'

def start_logging(queue=None, log_file='run.log', level=logging.INFO):
    if queue:                         # 子プロセス
        root = logging.getLogger()
        root.handlers = [QueueHandler(queue)]
        root.setLevel(level)
    else:                             # 親プロセス
        queue = mp.Manager().Queue(-1)
        console = logging.StreamHandler(sys.stderr)
        file    = logging.FileHandler(log_file, mode='a')
        for h in (console, file):
            h.setFormatter(logging.Formatter(FMT))
        #listener = QueueListener(queue, console, file)
        listener = QueueListener(queue, file)
        listener.start()
        root = logging.getLogger()
        root.handlers = [QueueHandler(queue)]
        root.setLevel(level)
        return queue, listener