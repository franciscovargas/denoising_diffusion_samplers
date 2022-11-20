from setuptools import find_packages, setup
from codecs import open
from os import path
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

here = path.abspath(path.dirname(__file__))
install_reqs = parse_requirements(here + '/requirements.txt', session=False)

# Catering for latest pip version
try:
    reqs = [str(ir.req) for ir in install_reqs]
except:
    reqs = [str(ir.requirement) for ir in install_reqs]

git_reqs = [
    "annealed_flow_transport @ git+https://github.com/franciscovargas/annealed_flow_transport",
    "jaxline @ git+https://github.com/deepmind/jaxline"
]


setup(
    name="DDS",
    version="0.0.1",
    description="Repository for Denoising Diffusion Based Sampling",
    packages=find_packages(),
    license="MIT",
    install_requires=git_reqs + reqs
)
