"""
The version of this checkout's code, for recording in trained models.  Imports nothing heavy, so it can be imported
before numpy and torch.
"""

import os
import subprocess

# The root of this checkout of the AnimalPoseForecasting repository
REPO_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Untracked files up to this size, in bytes, that are text are recorded whole; others by name only
MAX_UNTRACKED_FILE_SIZE = 100_000


class UncommittedChangesError(RuntimeError):
    """ The checkout has changes that aren't committed, so its code isn't any commit's. """


def _git(*args: str) -> str:
    return subprocess.run(['git', '-C', REPO_DIR_PATH, *args], check=True, capture_output=True, text=True).stdout


def _untracked_file_contents_maybe(file_name: str) -> str | None:
    """ The contents of an untracked file, named relative to the repo, or None if it's too big or not text. """
    file_path = os.path.join(REPO_DIR_PATH, file_name)
    if os.path.getsize(file_path) > MAX_UNTRACKED_FILE_SIZE:
        return None
    with open(file_path, 'rb') as f:
        data = f.read()
    # a NUL byte means binary, as for git, though it's valid UTF-8
    if b'\0' in data:
        return None
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return None


def code_version(allow_uncommitted: bool = False) -> dict:
    """ The version of this checkout's code, enough to rebuild it: check out 'commit', git apply 'diff', and write
    out 'untracked_files'.

    Returns:
        dict with
            commit: the hash of the commit the checkout is at.
            commit_time: that commit's time, ISO 8601.
            uncommitted: whether the checkout has changes that aren't committed, including untracked files that
                aren't ignored.
        and, if uncommitted,
            status: the output of git status --porcelain.
            diff: the output of git diff HEAD, the changes to tracked files, staged or not.
            untracked_files: {file name, relative to the repo: its contents, or None if it's over
                MAX_UNTRACKED_FILE_SIZE bytes or isn't text}.

    Raises:
        UncommittedChangesError: if there are uncommitted changes and not allow_uncommitted.
        RuntimeError: if git can't tell.
    """
    try:
        status = _git('status', '--porcelain')
        version = {'commit': _git('rev-parse', 'HEAD').strip(),
                   'commit_time': _git('show', '-s', '--format=%cI', 'HEAD').strip(),
                   'uncommitted': bool(status)}
        if status:
            if not allow_uncommitted:
                raise UncommittedChangesError(
                    f'{REPO_DIR_PATH} has uncommitted changes, so a model trained now would not record the code it '
                    f'was trained with.  Commit them first, or allow them explicitly:\n{status}')
            untracked_file_names = _git('ls-files', '--others', '--exclude-standard', '-z').split('\0')[:-1]
            version.update(status=status, diff=_git('diff', 'HEAD'),
                           untracked_files={n: _untracked_file_contents_maybe(n) for n in untracked_file_names})
    except (OSError, subprocess.CalledProcessError) as e:
        detail = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
        raise RuntimeError(f"Couldn't get the version of the code in {REPO_DIR_PATH} from git: {detail}") from e
    return version


def describe_code_version(version: dict) -> str:
    """ A one-line description of a code_version, for logs. """
    s = f"commit {version['commit']}, of {version['commit_time']}"
    if version['uncommitted']:
        s += f", with uncommitted changes to {len(version['status'].splitlines())} file(s)"
    return s
