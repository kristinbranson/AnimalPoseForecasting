"""Checks of apf/code_version.py, which finds the commit a model is trained with, in a throwaway git repo.

Run from the repository root with
    python -m pytest tests/test_code_version.py
or without pytest as
    python tests/test_code_version.py
"""
import os
import subprocess
import tempfile

import testing_utils  # puts this checkout first on sys.path; must precede the imports below

import apf.code_version as code_version


def in_throwaway_repo(check):
    """ Runs check(repo_dir_path) with apf.code_version pointed at a new git repo with one commit. """
    with tempfile.TemporaryDirectory() as repo_dir_path:
        def git(*args):
            subprocess.run(['git', '-C', repo_dir_path, *args], check=True, capture_output=True)
        git('init', '-q')
        with open(os.path.join(repo_dir_path, 'a.py'), 'w') as f:
            f.write('x = 1\n')
        git('add', 'a.py')
        git('-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-q', '-m', 'first')
        saved = code_version.REPO_DIR_PATH
        code_version.REPO_DIR_PATH = repo_dir_path
        try:
            check(repo_dir_path)
        finally:
            code_version.REPO_DIR_PATH = saved


def raises_uncommitted(repo_dir_path):
    try:
        code_version.committed_code_version()
    except code_version.UncommittedChangesError:
        return True
    return False


def test_clean_checkout_gives_its_commit():
    def check(repo_dir_path):
        head = subprocess.run(['git', '-C', repo_dir_path, 'rev-parse', 'HEAD'], check=True, capture_output=True,
                              text=True).stdout.strip()
        version = code_version.committed_code_version()
        assert version['commit'] == head
        assert version['commit_time'][:4].isdigit() and 'T' in version['commit_time']
    in_throwaway_repo(check)


def test_modified_file_is_refused():
    def check(repo_dir_path):
        with open(os.path.join(repo_dir_path, 'a.py'), 'w') as f:
            f.write('x = 2\n')
        assert raises_uncommitted(repo_dir_path)
    in_throwaway_repo(check)


def test_untracked_file_is_refused():
    def check(repo_dir_path):
        with open(os.path.join(repo_dir_path, 'b.py'), 'w') as f:
            f.write('y = 1\n')
        assert raises_uncommitted(repo_dir_path)
    in_throwaway_repo(check)


if __name__ == '__main__':
    testing_utils.run_as_script(globals())
