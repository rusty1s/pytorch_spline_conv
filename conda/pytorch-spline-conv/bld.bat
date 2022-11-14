echo "=================================="
echo "===RUNNING PIP INSTALL ==========="
"%PYTHON%" setup.py install
echo "=================================="
if errorlevel 1 exit 1
