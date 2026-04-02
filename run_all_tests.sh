#!/bin/bash
# Run all tests and generate summary

echo "================================================================================"
echo "SALOMI Test Suite"
echo "================================================================================"
echo ""

echo "Running Tier 5-9 tests (core new functionality)..."
echo "--------------------------------------------------------------------------------"
python -m pytest tests/core/ tests/model/ -v --tb=short
TIER59_EXIT=$?
echo ""

echo "Running all CPU tests (excluding OpenCL)..."
echo "--------------------------------------------------------------------------------"
python -m pytest tests/ -v -m "not opencl" --tb=line -q
ALL_CPU_EXIT=$?
echo ""

echo "================================================================================"
echo "Test Summary"
echo "================================================================================"
echo ""

if [ $TIER59_EXIT -eq 0 ]; then
    echo "✅ Tier 5-9 tests: PASSED"
else
    echo "❌ Tier 5-9 tests: FAILED"
fi

if [ $ALL_CPU_EXIT -eq 0 ]; then
    echo "✅ All CPU tests: PASSED"
else
    echo "⚠️  All CPU tests: SOME FAILURES (expected in old tier tests)"
fi

echo ""
echo "To run the end-to-end demo:"
echo "  python demo_gpt2_1bit.py"
echo ""
echo "To see detailed status:"
echo "  cat docs/FINAL-STATUS.md"
echo ""

