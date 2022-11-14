echo "=================================="
echo "===RUNNING PIP INSTALL ==========="
"%PYTHON%" -m pip install --verbose .
echo "=================================="
if errorlevel 1 exit 1
