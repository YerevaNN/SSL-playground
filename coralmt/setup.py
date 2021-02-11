import setuptools
from pathlib import Path
import re

long_description = Path('README.md').read_text(encoding='utf-8', errors='ignore')

vpat = re.compile(r"""__version__\s*=\s*['"]([^'"]*)['"]""")
__version__ = None
for line in Path('coralmt/__init__.py').read_text().splitlines():
    line = line.strip()
    if vpat.match(line):
        __version__ = vpat.match(line)[1]

print(f"Going to install coralmt {__version__}")
assert __version__, 'Could not find __version__ in __init__.py'

setuptools.setup(
    name='coralmt',
    version=__version__,
    author="Thamme Gowda",
    author_email="tg@isi.edu",
    description="CORALMT : USC ISI's machine translation system for LwLL",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/isi-nlp/rtg-xt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'rtg == 0.4.1',
        'requests == 2.24.0',
        'pandas == 1.0.5',
        'sacremoses ==  0.0.43',
        'pyarrow == 0.17.1'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'coral-snmt=coralmt.snmt.pipe:main',
            'coral-unmt=coralmt.unmt.pipe:main'
        ],
    }
)
