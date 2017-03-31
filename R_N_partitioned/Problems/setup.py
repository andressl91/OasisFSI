#This code is made using python 2.7

from distutils.core import setup
setup(name="Hron_Turek",
      version="1.0",
      author="Sebastian Gjertsen",
      py_modules=["Hron_Turek"], #only one module to setup.
      )

#If you get an "error: [Errno 13] Permission denied: '/Library/Python/2.7/site-packages/my_unit_testing.py'", when running python setup.py install:
#writing this in the commandline seems to do the trick:  sudo chown -R $USER /Library/Python/2.7/site-packages/
#or use : python setup.py --user
#You may have to change the 2.7 to whatever version you have.

#The code will be put in the folder /build/lib
