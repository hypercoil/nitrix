"""
Noxfile
"""
import nox

@nox.session()
def clean(session):
    session.install('coverage[toml]')
    session.run('coverage', 'erase')

@nox.session(python=["3.10", "3.11"])
def tests(session):
    session.install('.[dev]')
    session.run(
        'pytest',
        '--cov', 'nitrix',
        '--cov-append',
        'tests/',
    )
    session.run('ruff', 'check', 'src/nitrix')
    session.run('ruff', 'format', '--check', 'src/nitrix')

@nox.session()
def report(session):
    session.install('coverage[toml]')
    session.run(
        'coverage',
        'report', '--fail-under=99',
        "--omit='*test*,*__init__*'",
    )
    session.run(
        'coverage',
        'html',
        "--omit='*test*,*__init__*'",
    )
    session.run(
        'coverage',
        'xml',
        "--omit='*test*,*__init__*'",
    )
