@echo off
REM ============================================================
REM Phase 3: 消融实验 + 损失权重优化 (Windows 串行版)
REM 用法: run_phase3.bat
REM ============================================================

cd /d "%~dp0\.."
echo ========================================
echo Phase 3 Experiment Runner (Windows)
echo Project: %cd%
echo Start: %date% %time%
echo ========================================

set COMMON=--epochs 100 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 --patience 15 --seed 42 --num_workers 0

echo.
echo ========================================
echo Phase 3a: Ablation Study
echo ========================================

echo [1/11] p3a_rgb
python experiments/train_tcn.py %COMMON% --exp_tag phase3/p3a_rgb
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/11] p3a_diff
python experiments/train_tcn.py %COMMON% --use_diff --exp_tag phase3/p3a_diff
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/11] p3a_pose
python experiments/train_tcn.py %COMMON% --pose_npz data/omnifall_pose.npz --exp_tag phase3/p3a_pose
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/11] p3a_diff_pose
python experiments/train_tcn.py %COMMON% --use_diff --pose_npz data/omnifall_pose.npz --exp_tag phase3/p3a_diff_pose
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Phase 3b: Loss Weight Search
echo ========================================

echo [5/11] p3b_fallen_x2
python experiments/train_tcn.py %COMMON% --use_diff --pose_npz data/omnifall_pose.npz --w_fall 0.5 --w_fallen 1.0 --exp_tag phase3/p3b_fallen_x2
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/11] p3b_fall_reduce
python experiments/train_tcn.py %COMMON% --use_diff --pose_npz data/omnifall_pose.npz --w_fall 0.3 --w_fallen 0.5 --exp_tag phase3/p3b_fall_reduce
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/11] p3b_balanced
python experiments/train_tcn.py %COMMON% --use_diff --pose_npz data/omnifall_pose.npz --w_fall 0.3 --w_fallen 1.0 --exp_tag phase3/p3b_balanced
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/11] p3b_fallen_x3
python experiments/train_tcn.py %COMMON% --use_diff --pose_npz data/omnifall_pose.npz --w_fall 0.5 --w_fallen 1.5 --exp_tag phase3/p3b_fallen_x3
if %errorlevel% neq 0 exit /b %errorlevel%

echo [9/11] p3b_equal
python experiments/train_tcn.py %COMMON% --use_diff --pose_npz data/omnifall_pose.npz --w_fall 1.0 --w_fallen 1.0 --exp_tag phase3/p3b_equal
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Phase 3c: Seesaw Fix
echo ========================================

echo [10/11] p3c_pose_fall_reduce
python experiments/train_tcn.py %COMMON% --pose_npz data/omnifall_pose.npz --w_fall 0.3 --w_fallen 0.5 --exp_tag phase3/p3c_pose_fall_reduce
if %errorlevel% neq 0 exit /b %errorlevel%

echo [11/11] p3c_pose_balanced
python experiments/train_tcn.py %COMMON% --pose_npz data/omnifall_pose.npz --w_fall 0.3 --w_fallen 1.0 --exp_tag phase3/p3c_pose_balanced
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Phase 3 Complete!
echo End: %date% %time%
echo ========================================

echo Running analysis...
python experiments/analyze_phase3.py

echo Done.
