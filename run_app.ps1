# Start the app on Windows with safe thread settings.
#   .\run_app.ps1
$env:OPENBLAS_NUM_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
streamlit run app/app.py --server.fileWatcherType none
