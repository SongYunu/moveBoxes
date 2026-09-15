"""The original resumable StageACT trainer with all-parcel pick sampling."""
import json
import sys
from pathlib import Path
import stage_train
from stage_pick_sampling import StageWindows

if __name__ == '__main__':
    stage_train.StageWindows = StageWindows
    stage_train.train(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
