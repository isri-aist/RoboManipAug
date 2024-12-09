from setuptools import setup
from distutils.util import convert_path

# https://stackoverflow.com/a/24517154
main_ns = {}
version_path = convert_path("robo_manip_aug/version.py")
with open(version_path) as version_file:
    exec(version_file.read(), main_ns)

setup(
    name="robo_manip_aug",
    version=main_ns["__version__"],
)
