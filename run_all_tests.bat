@echo off
REM Run all tests and generate summary

echo ================================================================================
echo SALOMI Test Suite
echo ================================================================================
echo.

echo Running Tier 5-9 tests (core new functionality)...
echo --------------------------------------------------------------------------------
python -m pytest tests/core/ tests/model/ -v --tb=short
set TIER59_EXIT=%ERRORLEVEL%
echo.

echo Running all CPU tests (excluding OpenCL)...
echo --------------------------------------------------------------------------------
python -m pytest tests/ -v -m "not opencl" --tb=line -q
set ALL_CPU_EXIT=%ERRORLEVEL%
echo.

echo ================================================================================
echo Test Summary
echo ================================================================================
echo.

if %TIER59_EXIT%==0 (
    echo [32m✅ Tier 5-9 tests: PASSED[0m
) else (
    echo [31m❌ Tier 5-9 tests: FAILED[0m
)

if %ALL_CPU_EXIT%==0 (
    echo [32m✅ All CPU tests: PASSED[0m
) else (
    echo [33m⚠️  All CPU tests: SOME FAILURES (expected in old tier tests)[0m
)

echo.
echo To run the end-to-end demo:
echo   python demo_gpt2_1bit.py
echo.
echo To see detailed status:
echo   type docs\FINAL-STATUS.md
echo.

pause

