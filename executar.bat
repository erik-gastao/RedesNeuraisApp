@echo off
rem Clique duas vezes neste arquivo para abrir o Construtor de Redes Neurais.

rem Vai para a pasta deste arquivo (funciona mesmo abrindo de outro lugar).
cd /d "%~dp0"

rem Python em modo UTF-8 (acentos corretos ao ler e escrever arquivos).
set PYTHONUTF8=1

if not exist main.py (
    echo O arquivo main.py ainda nao existe. Ele sera criado na etapa 6 do PLANO.md.
    echo Por enquanto, use testar.bat para rodar os testes.
    goto fim
)

rem Usa "python"; se nao achar, tenta o lancador "py" do Windows.
where python > nul 2> nul
if %errorlevel% == 0 (
    python main.py
) else (
    py main.py
)

:fim
echo.
pause
