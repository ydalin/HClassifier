from run_model import run_model
from gather_data import gather_data


def main():
    """
    Runs model, outputs data
    """
    prompt = "How to run? Input 1 or 2:" + \
             "\n1 Directly (very slow)" + \
             "\n2 From saved 'stats.pkl' file (not as slow)"
    directly = int(input(prompt)) == 1
    if directly:
        print("Running Directly")
        data = gather_data()
    else:
        print("Running from saved stats.pkl file")

    run_model()


if __name__ == "__main__":
    main()
