import sys
from pathlib import Path
from foampilot.solver import Solver

def main():
    case_dir = Path(__file__).resolve().parent / "case"
    solver = Solver(case_dir)
    
    log_file = "log.foamRun"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        
    solver.run_command(["foamRun"], log_file)

if __name__ == "__main__":
    main()
