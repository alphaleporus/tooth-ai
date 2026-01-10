@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=%PATH%;C:\Users\Student\AppData\Local\Programs\Git\bin
set DISTUTILS_USE_SDK=1
C:\Users\Student\Tooth-ai\venv\Scripts\pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation
