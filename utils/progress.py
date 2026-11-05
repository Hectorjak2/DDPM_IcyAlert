import sys

from tqdm import tqdm


def progress(iterable, desc: str):
    """Progress bar for one pass over a dataloader, shared by the DDPM and UNet training loops.

    Written to stdout, not tqdm's default stderr, so it lands in the same log file as the
    epoch summaries (titan.s splits .out and .err). In a log file every refresh is appended
    rather than redrawn, so refresh only every 30 s there to keep the file small.
    """
    interactive = sys.stdout.isatty()
    return tqdm(iterable, desc=desc, file=sys.stdout, dynamic_ncols=interactive,
                ncols=None if interactive else 100, mininterval=0.1 if interactive else 30)
