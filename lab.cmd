@echo off
REM nemo-rl-lab CLI 薄入口（Windows）：委托给项目环境里的 lab 命令（= uv run lab）。
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
uv run --project "%ROOT%" lab %*
