import traceback
import datetime
import sys

#Global debug mode toggle
IS_DEBUG_MODE = True

def log_error(e, log_file="log.txt"):
    """
    Logs error details to a file.
    
    parameter e: The exception object
    parameter log_file: The file where errors will be logged
    """
    error_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_type = type(e).__name__
    
    #Get traceback details
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    with open(log_file, "a") as sw:
        sw.write(f"\n[{error_time}] ERROR: {error_type} - {str(e)}\n")
        sw.write(f"Occurred in: {file_name} (Line {line_number})\n")
        if IS_DEBUG_MODE:
            sw.write(traceback.format_exc() + "\n")

    print("\n Error details written to", log_file)

def error_handling(e):
    """
    Handles errors by displaying information, logging, and giving the option to exit.
    
    parameter e: The exception object
    """
    error_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_type = type(e).__name__
    
    print("\nAn error occurred!")
    print("----------------------------")
    print(f"Exception Type: {error_type}")
    print(f"Message: {str(e)}")
    print(f"Date and Time: {error_time}")

    #Get traceback details
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    print(f"Occurred in: {file_name} (Line {line_number})")

    if IS_DEBUG_MODE:
        print("\nDebug Mode Enabled: Additional Info")
        print("----------------------------")
        print(f"Source: {getattr(e, '__module__', 'Unknown')}")
        print("\nStack Trace: ")
        print(traceback.format_exc())

    #Log error
    log_error(e)

    #Ask user whether to exit or continue
    choice = input("\nPress Enter to exit, or type 'c' to continue: ").strip().lower()
    if choice != 'c':
        print("\nExiting program.")
        exit(0)
    else:
        print("\nContinuing execution...")