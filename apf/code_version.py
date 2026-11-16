"""
The version of this checkout's code, for recording in trained models.  Imports nothing heavy, so it can be imported
before numpy and torch.
"""

import os
import subprocess

# The root of this checkout of the AnimalPoseForecasting repository
REPO_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UncommittedChangesError(RuntimeError):
    """ The checkout has changes that aren't committed, so its code isn't any commit's. """


def _git(*args: str) -> str:
    return subprocess.run(['git', '-C', REPO_DIR_PATH, *args], check=True, capture_output=True, text=True).stdout


def committed_code_version() -> dict:
    """ The commit this checkout is at, as {'commit': its hash, 'commit_time': its commit time, ISO 8601}.

    Raises UncommittedChangesError if the checkout has changes that aren't committed, including untracked files
    that aren't ignored, since then the code isn't that commit's, and RuntimeError if git can't tell.
    """
    try:
        status = _git('status', '--porcelain')
        commit = _git('rev-parse', 'HEAD').strip()
        commit_time = _git('show', '-s', '--format=%cI', 'HEAD').strip()
    except (OSError, subprocess.CalledProcessError) as e:
        detail = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
        raise RuntimeError(f"Couldn't get the commit of the code in {REPO_DIR_PATH} from git: {detail}") from e
    if status:
        raise UncommittedChangesError(f'{REPO_DIR_PATH} has uncommitted changes, so a model trained now would not '
                                      f'record the code it was trained with.  Commit them first:\n{status}')
    return {'commit': commit, 'commit_time': commit_time}
