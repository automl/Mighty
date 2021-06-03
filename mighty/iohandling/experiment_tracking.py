import argparse
import datetime
import json
import os
import sys
import tempfile

import subprocess


def prepare_output_dir(args, user_specified_dir=None, argv=None,
                       subfolder_naming_scheme=None):
    """
    Largely inspired by chainerRLs prepare output dir (also copied parts of it).
    Differences: We also allow to index experiment subfolders by experiment seeds and we do not check if the repo
    is under git control but always assume this to be true. Lastly we also print to stdout where data will be stored.

    For the original code see:
    (https://github.com/chainer/chainerrl/blob/018a29132d77e5af0f92161250c72aba10c6ce29/chainerrl/experiments/prepare_output_dir.py)
    Prepare a directory for outputting training results.

    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:

        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables

    Args:
        args (dict or argparse.Namespace): Arguments to save
        user_specified_dir (str or None): If str is specified, the output
            directory is created under that path. If not specified, it is
            created as a new temporary directory instead.
        argv (list or None): The list of command line arguments passed to a
            script. If not specified, sys.argv is used instead.
        subfolder_naming_scheme (str): Format used to represent the current datetime. The
        default format is the basic format of ISO 8601.
    Returns:
        Path of the output directory created by this function (str).
    """

    if subfolder_naming_scheme == 'time':
        subfolder_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S.%f')
    else:
        subfolder_str = '{:>05d}'.format(args.seed)
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, subfolder_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=datetime.datetime.now().strftime('%Y%m%dT%H%M%S.%f'))

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        if argv is None:
            argv = sys.argv
        f.write(' '.join(argv))

    try:
        cwd = os.getcwd()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir_path)
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, 'git-head.txt'), 'wb') as f:
            f.write(subprocess.check_output('git rev-parse HEAD'.split()))

        # Save `git status`
        with open(os.path.join(outdir, 'git-status.txt'), 'wb') as f:
            f.write(subprocess.check_output('git status'.split()))

        # Save `git log`
        with open(os.path.join(outdir, 'git-log.txt'), 'wb') as f:
            f.write(subprocess.check_output('git log'.split()))

        # Save `git diff`
        with open(os.path.join(outdir, 'git-diff.txt'), 'wb') as f:
            f.write(subprocess.check_output('git diff'.split()))
        os.chdir(cwd)
    except subprocess.CalledProcessError:
        print('Not in a git environment')

    print('Results stored in {:s}'.format(os.path.abspath(outdir)))
    return outdir

