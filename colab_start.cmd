@echo off
setlocal
set "PYTHONUTF8=1"
if "%~1"=="" (
  python "%~dp0colab_subtitles.py" --help
  exit /b 2
)
python "%~dp0colab_subtitles.py" %*
exit /b %ERRORLEVEL%
