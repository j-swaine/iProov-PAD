"""Score a folder of face images: `python infer.py --input <folder> --output scores.csv` (same as `screenpad infer`)."""

import sys

from screenpad.cli import main

if __name__ == "__main__":
    main(["infer", *sys.argv[1:]])
