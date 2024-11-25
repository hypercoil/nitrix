"""
Noxfile
"""
import nox

@nox.session()
def clean(session):
    session.install('coverage[toml]')
    session.run('coverage', 'erase')

@nox.session(venv_backend='uv', python=["3.11", "3.12", "3.13"])
def tests(session):
    #session.install('.[dev]')
    session.run_install(
        "uv",
        "sync",
        "--extra=dev",
        "--frozen",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run(
        'pytest',
        '--cov', 'nitrix',
        '--cov-append',
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
