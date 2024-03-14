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
  @rem TEMPORARY STOP IF NO ARGUMENTS
  goto :end
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
@rem TODO: Do we need to deactivate? Maybe within individual functions? End on same environment as start!
call conda deactivate 2>nul
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

:: ###########################################
:: ## Step 1: Download Transformer Lab
:: ## in the ~/.transformerlab/src directory.
:: ###########################################
:download_transformer_lab
title "Step 1: Download the latest release of Transformer Lab"

@rem Figure out the path to the lastest release of Transformer Lab
@rem The second line saves the output of the curl call to a variable called curl_response
@rem The third line to pulls the filename from the full URL: nx is filename n and extension x
set CURL_CMD=curl -Ls -o /dev/null -w %%{url_effective} https://github.com/transformerlab/transformerlab-api/releases/latest
for /F %%G In ('%CURL_CMD%') do set "curl_response=%%G"
echo %curl_response%
for %%f in (%curl_response%) do set LATEST_RELEASE_VERSION=%%~nxf
echo Latest Release on Github: %LATEST_RELEASE_VERSION%

set TLAB_URL="https://github.com/transformerlab/transformerlab-api/archive/refs/tags/%LATEST_RELEASE_VERSION%.zip"
echo Download Location: %TLAB_URL%

@rem If the user has not installed Transformer Lab, then we should install it.
:: ohai "Installing Transformer Lab ${LATEST_RELEASE_VERSION}..."
echo Installing Transformer Lab %LATEST_RELEASE_VERSION%...
@rem TODO: Shut up if it exists already
md %TLAB_DIR%
call curl -L "%TLAB_URL%" -o "%TLAB_DIR%\transformerlab.zip"
set NEW_DIRECTORY_NAME="transformerlab-api-%LATEST_RELEASE_VERSION%"

@rem TODO: Shut up if this is already gone?
rd /s /q "%TLAB_DIR\%NEW_DIRECTORY_NAME%"
rd /s /q "%TLAB_CODE_DIR%"
call tar -xf "%TLAB_DIR%\transformerlab.zip" -C "%TLAB_DIR%"

rename "${TLAB_DIR}/${NEW_DIRECTORY_NAME}" "${TLAB_CODE_DIR}"
del "${TLAB_DIR}/transformerlab.zip"

@rem Create a file called LATEST_VERSION that contains the latest version of Transformer Lab.
echo "${LATEST_RELEASE_VERSION}" > "%TLAB_CODE_DIR%/LATEST_VERSION"

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


:: #########################################
:: ## Step 3: Create the Conda Environment
:: #########################################
:create_conda_environment
title "Step 3: Create the Conda Environment"
call :check_conda

@rem TODO: How do on windows?
:: eval conda shell.bash hook
call "%MINICONDA_ROOT%\Scripts\activate.bat"
call conda env list | FIND "%ENV_DIR%"

@rem NOTE: This will SET TLAB_ENV_CHECK with output of any environment that matches our target
FOR /F %%I IN ('conda env list ^| FIND "%ENV_DIR%"') DO @SET "TLAB_ENV_CHECK=%%I"
if %TLAB_ENV_CHECK%=="" (
    echo conda create -y -n %ENV_DIR% python=3.11
    call conda create -y -k --prefix "%ENV_DIR%" python=3.11
) else (
    echo ✅ Conda environment $ENV_DIR already exists.
)
echo conda activate %ENV_DIR%
call conda activate %ENV_DIR%
EXIT /B 0


:: ################################
:: ## Step 4: Install Dependencies
:: ################################
:install_dependencies
title "Step 4: Install Dependencies"

@rem TODO: How do on windows?
:: eval conda shell.bash hook
call "%MINICONDA_ROOT%\Scripts\activate.bat"
call conda activate %ENV_DIR%

call :check_python

@rem Check if nvidia-smi is available
set HAS_GPU=false
call "nvidia-smi" > NUL
if %ERRORLEVEL%==0 (
    set NVIDIA_CMD=nvidia-smi --query-gpu=name --format=csv,noheader,nounits
    for /F "delims=" %%G in ('call %%NVIDIA_CMD%%') do set GPU_INFO=%%G
    echo nvidia-smi is available
    if "%GPU_INFO%" == "" (
        echo Nvidia SMI exists, No NVIDIA GPU detected. Perhaps you need to re-install NVIDIA drivers.
    ) else (
        echo NVIDIA GPU detected: %GPU_INFO%
        set HAS_GPU=true
    )
)
echo HAS_GPU=%HAS_GPU%

@rem Install CUDA if there is an NVIDIA GPU
if %HAS_GPU%==true (
    echo Your computer has a GPU; installing cuda:
    call conda install -y cuda -c nvidia/label/cuda-12.1.1
    echo Installing requirements:
    call pip install --upgrade -r %TLAB_CODE_DIR%\requirements-no-gpu.txt
    call pip install --upgrade -r %TLAB_CODE_DIR%\requirements.txt
) else (
    echo No NVIDIA GPU detected drivers detected. Install NVIDIA drivers to enable GPU support.
    echo https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions
    echo Installing Tranformer Lab requirements without GPU support
    call pip install --upgrade -r %TLAB_CODE_DIR%\requirements-no-gpu.txt
)

@rem Missing Windows dependencies
call pip install --upgrade colorama

@rem As a test for if things completed, check if the uvicorn command works:
@rem TODO fix the abort and ohai calls
call uvicorn --version > NUL
if %ERRORLEVEL%==0 (
::  ohai "✅ Uvicorn is installed."
  echo ✅ Uvicorn is installed.
) else (
::  abort "❌ Uvicorn is not installed. This usually means that the installation of dependencies failed."
  echo ❌ Uvicorn is not installed. This usually means that the installation of dependencies failed.
)
EXIT /B 0

:list_installed_packages
@rem TODO: is this next line necessary given the call to conda activate?
:: eval "$(${CONDA_BIN} shell.bash hook)"
call "%MINICONDA_ROOT%\Scripts\activate.bat"
call pip list --format json
EXIT /B 0

:list_environments
@rem TODO: is this next line necessary given the call to conda activate?
:: eval "$(${CONDA_BIN} shell.bash hook)"
call "%MINICONDA_ROOT%\Scripts\activate.bat"
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
  for /F "delims=" %%G in ('call python --version') do set "CURRENT_PYTHON_VERSION=%%G"
  echo ✅ Python is installed: %CURRENT_PYTHON_VERSION%
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
@rem Check for nvidia installation
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
