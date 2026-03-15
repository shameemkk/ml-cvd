@echo off
echo ========================================
echo   Heart Disease Prediction - Django App
echo ========================================
echo.

:: Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

:: Navigate to Django project folder
echo [2/3] Entering project directory...
cd heart_disease

:: Run Django development server
echo [3/3] Starting Django server at http://127.0.0.1:8000/
echo.
echo Press Ctrl+C to stop the server.
echo.
python manage.py runserver

pause
