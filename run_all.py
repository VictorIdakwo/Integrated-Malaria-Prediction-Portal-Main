import subprocess

def run_app(command):
    """Runs a given command as a subprocess."""
    return subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    # Run main (combined front-end) app
    main_process = run_app("python main_app.py")
    
    # Run Clinical Malaria Prediction app
    clinical_process = run_app("python clinical_malaria.py")
    
    # Run Non-Clinical Malaria Prediction app
    non_clinical_process = run_app("python non_clinical_malaria.py")
    
    print("All apps are running. Access the combined system at http://127.0.0.1:5000")
    
    try:
        # Keep the script alive to ensure subprocesses continue running
        main_process.wait()
        clinical_process.wait()
        non_clinical_process.wait()
    except KeyboardInterrupt:
        print("Shutting down all apps...")
        main_process.terminate()
        clinical_process.terminate()
        non_clinical_process.terminate()
