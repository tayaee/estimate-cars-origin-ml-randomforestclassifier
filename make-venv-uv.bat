@echo off

set CURRENT_DRIVE=%CD:~0,2%
set UV_CACHE_DIR=%CURRENT_DRIVE%\.uv-cache
echo Using UV_CACHE_DIR=%UV_CACHE_DIR%

setlocal enabledelayedexpansion

REM 1. Determine the version. If different, re-pin it.
if not .%1. == .. (
	set VERSION=%1
	echo DEBUG 1/8 Using VERSION=!VERSION! from CLI
	if exist .python-version (
		for /f "usebackq delims=" %%a in (".python-version") do set FILE_VERSION=%%a
		if not .!VERSION!. == .!FILE_VERSION!. (
			echo DEBUG Changing python version from !FILE_VERSION! to !VERSION!
			if exist .venv rd /s /q .venv

			echo DEBUG Python version !VERSION! is being re-pinned at .python-version...
			uv python pin !VERSION!
			if exist uv.lock (
				echo DEBUG Removed uv.lock for re-creation...
				del /q uv.lock
			)
		)
	)
) else (
	if exist .python-version (
		for /f "usebackq delims=" %%a in (".python-version") do set VERSION=%%a
		echo DEBUG 1/8 Using VERSION=!VERSION! from .python-version
	) else (
		set VERSION=3.11
		echo DEBUG 1/8 Using VERSION=!VERSION! by default
	)
)

REM 2. Pin the python version
if not exist .python-version (
	echo DEBUG 2/8 uv python pin !VERSION!
	uv python pin !VERSION!
) else (
	echo SKIP 2/8 uv python pin !VERSION!
)

REM 3. Install python.
echo DEBUG 3/8 uv python install
uv python install

REM 4. Create venv
if not exist .venv (
	echo DEBUG 4/8 uv venv .venv
	uv venv .venv
) else (
	echo SKIP 4/8 uv venv .venv
)

REM 5. Activate venv if working directory was changed
set "VENV_LOWER=!VIRTUAL_ENV!"
set "CD_VENV_LOWER=!CD!\.venv"
set TMP_FILE_1=%TEMP%\tmp.%RANDOM%.tmp
set TMP_FILE_2=%TEMP%\tmp.%RANDOM%.tmp
echo !VENV_LOWER!> !TMP_FILE_1!
echo !CD_VENV_LOWER!> !TMP_FILE_2!
fc /c !TMP_FILE_1! !TMP_FILE_2! > NUL 2>&1
IF %ERRORLEVEL% EQU 0 (
	echo SKIP 5/8 call .venv\scripts\activate [!VENV_LOWER!]
) else (
	echo DEBUG 5/8 call .venv\scripts\activate [not .!VENV_LOWER!. == .!CD_VENV_LOWER!.]
	prompt $P$G
	call .venv\scripts\activate
)
del /q !TMP_FILE_1! !TMP_FILE_2! > NUL 2>&1

REM 6. Pin the package versions
if exist requirements.txt (
    if not exist uv.lock (
        echo DEBUG 6/8 uv pip compile requirements.txt -o uv.lock [CREATED]
        uv pip compile requirements.txt -o uv.lock
    ) else (
        for %%A in (requirements.txt) do set REQTIME=%%~tA
        for %%B in (uv.lock) do set LOCKTIME=%%~tB

        if "!REQTIME!" GTR "!LOCKTIME!" (
            echo DEBUG 6/8 uv pip compile requirements.txt -o uv.lock [UPDATED]
            uv pip compile requirements.txt -o uv.lock
        ) else (
            echo SKIP 6/8 uv pip compile requirements.txt -o uv.lock
        )
    )
) else (
    echo SKIP 6/8 uv pip compile requirements.txt -o uv.lock
)

REM 7. Install the packages
if exist uv.lock (
    echo DEBUG 7/8 uv pip sync uv.lock
    uv pip sync uv.lock
) else (
    echo SKIP 7/8 uv pip sync uv.lock
)

endlocal & (
    REM Make sure the variables updated inside the 'setlocal' block are also available outside the parent shell.
    set "VIRTUAL_ENV=%VIRTUAL_ENV%"
    set "PATH=%PATH%"
    set "PROMPT=%PROMPT%"
)

REM 8. Confirm the python path.
python -c "import sys, platform; print('DEBUG 8/8', sys.executable, '-V')"

REM 9. Confirm the python version.
python -V


