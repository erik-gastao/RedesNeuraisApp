@echo off
rem Clique duas vezes neste arquivo para abrir o Construtor de Redes Neurais.

rem Vai para a pasta deste arquivo (funciona mesmo abrindo de outro lugar).
cd /d "%~dp0"

rem Python em modo UTF-8 (acentos corretos ao ler e escrever arquivos).
set PYTHONUTF8=1

rem Usa "python"; se nao achar, tenta o lancador "py" do Windows.
where python > nul 2> nul
if %errorlevel% == 0 (
    python main.py
) else (
    py main.py
)

echo.
pause
