@echo off
rem Clique duas vezes neste arquivo para rodar todos os testes.

cd /d "%~dp0"
set PYTHONUTF8=1

where python > nul 2> nul
if %errorlevel% == 0 (
    python -m unittest discover testes -v
) else (
    py -m unittest discover testes -v
)

echo.
pause
