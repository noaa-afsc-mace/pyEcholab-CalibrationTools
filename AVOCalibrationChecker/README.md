# AVOCalibrationChecker

This is designed to be used as a command line tool for a quick calibration verification when collecting data using ES80 software (without Simrad calibration tool). For distribution, this is compiled into an executable using [PyInstaller](https://github.com/pyinstaller).

To produce 64-bit executable:
-  Run PyInstaller with the --console argument for the AVOCalibrationChecker.py script
-  Manually copy all mkl_* dlls abd libiomp5md.dll the python environment used for the build into the dist/_internal directory
-  Copy the AVO.ini file into the same directory as the executable

To produce 32-bit executable (required python 3.6.X environment):
- Include "import matplotlib,matplotlib.use('TKAgg')" proir to other matplotlib imports at the start of AVOCalibrationChecker.py
- Run PyInstaller, including 'scipy._lib.messagestream' and 'scipy.special.cython_special' as hidden imports (see AVOCalibrationChecker.spec)
- Manually copy all dlls from the scipy library within your python environment to the distribution folder
- Copy the AVO.ini file into the distribution folder
