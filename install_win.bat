@echo off

set ENV_NAME=transformerlab
set TLAB_DIR=%USERPROFILE%\.transformerlab
set TLAB_CODE_DIR=%TLAB_DIR%\%src

set MINICONDA_ROOT=%TLAB_DIR%\miniconda3
set CONDA_BIN=%MINICONDA_ROOT%\Scripts\conda
set ENV_DIR=%TLAB_DIR%\envs\%ENV_NAME%

@rem deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

@rem Check if there are arguments to this script, and if so, run the appropriate function.
if [%1]==[] (
  title "Performing a full installation of Transformer Lab."
  call :download_transformer_lab
  call :install_conda
  call :create_conda_environment
  call :install_dependencies
  call :print_success_message
) else (
  call :%1
  @rem TODO fix emoji
  if ERRORLEVEL 1 echo ❌ Unknown argument: %1
)

:end
call conda deactivate
EXIT /B %ERRORLEVEL%


:: ---------------------------------------- INSTALLATION STEPS -------------------------------------

:title
@rem TODO Remove all of the commented lines
echo TITLE
:: printf "%tty_blue%#########################################################################%tty_reset%\n"
:: SET _INTERPOLATION_0=
:: FOR /f "delims=" %%a in ('shell_join "$@"') DO (SET "_INTERPOLATION_0=!_INTERPOLATION_0! %%a")
:: printf "%tty_blue%#### %tty_bold% %s%tty_reset%\n" "!_INTERPOLATION_0:~1!"
:: printf "!tty_blue!#########################################################################!tty_reset!\n"
EXIT /B 0

:download_transformer_lab
echo downloading transformer lab
EXIT /B 0

:: ##############################
:: ## Step 2: Install Conda
:: ##############################
:install_conda
title "Step 2: Install Conda"

@rem check if conda already exists:
call %CONDA_BIN% > NUL
if %ERRORLEVEL%==1 (
  echo Conda is not installed at %MINICONDA_ROOT%.

  MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
  echo Downloading %MINICONDA_URL%
  call curl -o miniconda.exe "$MINICONDA_URL"
  start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\.transformerlab\miniconda3
  rm miniconda.exe
) else (
  @rem TODO ohai
  echo Conda is installed at %MINICONDA_ROOT%, we do not need to install it
)
  
@rem Enable conda in shell
@rem TODO WTF?
:: eval conda shell.bash hook
call :check_conda
EXIT /B 0

:create_conda_environment
echo creating conda environemtn
EXIT /B 0

:install_dependencies
echo installing dependencies
EXIT /B 0

:list_installed_packages
@rem TODO: is this next line necessary given the call to conda activate?
:: eval "$(${CONDA_BIN} shell.bash hook)"
call "%MINICONDA_ROOT%\Scripts\activate.bat" || ( echo Miniconda hook not found.)
call pip list --format json
EXIT /B 0

:list_environments
@rem TODO: is this next line necessary given the call to conda activate?
:: eval "$(${CONDA_BIN} shell.bash hook)"
call "%MINICONDA_ROOT%\Scripts\activate.bat" || ( echo Miniconda hook not found.)
call conda env list
EXIT /B 0

:check_conda
call %CONDA_BIN% > NUL
if %ERRORLEVEL%==1 (
  echo ❌ Conda is not installed at %MINICONDA_ROOT%. Please install Conda using 'install_win.bat install_conda' and try again.
) else (
  echo ✅ Conda is installed at %CONDA_BIN%
)
EXIT /B 0

:check_python
call python --version > NUL
if %ERRORLEVEL%==1 (
  echo ❌ Python is not installed as 'python'. Please install Python and try again or it could be installed as 'python3'
) else (
  echo ✅ Python is installed: $PYTHON_VERSION
)
EXIT /B 0


:doctor
@rem TODO clear all lines strating with ::
title "Doctor"
:: ohai "Checking if everything is installed correctly."
echo Your machine is running: %OS%
@rem Check for Conda installation
call %CONDA_BIN% > NUL
if %ERRORLEVEL%==1 (
  echo Error %ERRORLEVEL%
  echo Conda is not installed at %MINICONDA_ROOT%. Please install Conda using 'install_win.bat install_conda' and try again.
) else (
  @rem TODO figure out how to report conda version
::  echo Your conda version is: %CONDA_BIN%
::  @rem Check for conda in the path
  echo Conda is installed at %CONDA_BIN%
)
@rem Check for Conda installation
call nvidia-smi > NUL
if %ERRORLEVEL%==1 (
  echo nvidia-smi is not installed.
) else (
  echo Your nvidia-smi version is: 
)
call :check_conda
call :check_python
EXIT /B 0


:print_success_message
title "Installation Complete"
echo ------------------------------------------
echo Transformer Lab is installed to:
echo   %TLAB_DIR%
echo Your workspace is located at:
echo   %TLAB_DIR%\workspace
echo Your conda environment is at:
echo   %ENV_DIR%
echo You can run Transformer Lab with:
echo   conda activate %ENV_DIR%
echo   %TLAB_DIR%\src\run.bat
echo ------------------------------------------
echo
EXIT /B 0

