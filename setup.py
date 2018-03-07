from setuptools import setup

setup(name='convert_dicom',
      version='0.1',
      description='Converts Dicom Files to numpy arrays',
      url='http://github.com/cairnlabs/convert-dicom',
      author='Cairn Labs',
      author_email='jacob@cairnlabs.com',
      license='-',
      packages=['convert_dicom'],
      zip_safe=True,
      install_requires=['numpy', 'scipy'])