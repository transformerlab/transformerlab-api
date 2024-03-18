@echo off

set ENV_NAME=transformerlab
set TLAB_DIR=%USERPROFILE%\.transformerlab
set TLAB_CODE_DIR=%TLAB_DIR%\%src

set MINICONDA_ROOT=%TLAB_DIR%\miniconda3
set CONDA_BIN=%MINICONDA_ROOT%\Scripts\conda
set ENV_DIR=%TLAB_DIR%\envs\%ENV_NAME%

echo Conda's binary is at %CONDA_BIN%
echo Your current directory is %CD%

call %CONDA_BIN% > NUL
if %ERRORLEVEL%==1 (
  echo ❌ Conda is not installed at %MINICONDA_ROOT%. Please install Conda using 'install_win.bat install_conda' and try again.
  goto :endscript
) else (
  echo ✅ Conda is installed at %CONDA_BIN%
)

echo 👏 Enabling conda in shell

echo 👏 Activating transformerlab conda environment
call "%MINICONDA_ROOT%\Scripts\activate.bat"
call conda activate %ENV_DIR%

@rem Check if the uvicorn command works:
call uvicorn --version > NUL
if %ERRORLEVEL%==0 (
  echo ✅ Uvicorn is installed.
) else (
  echo ❌ Uvicorn is not installed. This usually means that the installation of dependencies failed. Run ./install.sh to install the dependencies.
  goto :endscript
)

echo 👏 Starting the API server
set PYTHONUTF8=1
uvicorn api:app --port 8000 --host 0.0.0.0 

:endscript
