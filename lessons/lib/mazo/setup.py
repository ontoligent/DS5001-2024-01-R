from distutils.core import setup

setup(name='Mazo',
      version='2.0',
      description='A low code wrapper for MALLET to generate topic models.',
      author='Raf Alvarado',
      author_email='ontoligent@gmail.com',
      url='https://github.com/ontoligent/mazo',
      py_modules=['polite.polite'],
      scripts=['mazo']
     )