
import os
import sys
sys.path.append('./src')
from backend import core, img_loader


def test_warper() -> None:
    img_path = os.path.join(os.getcwd(), 'src', 'tests', 'pics')
    core.dry_run(img_path, 'cps')
    core.dry_run(img_path, 'cas')


def test_emoji_parser() -> None:
    parser = img_loader.EmojipediaParser()
    parser.fetch_random()
