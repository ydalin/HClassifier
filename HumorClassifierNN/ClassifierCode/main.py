from run_model import run_model
from gather_data import write_to_pickle

def main():
    """
    Runs model, outputs data
    """
    prompt = "How to run? Input 1 or 2 or 3:" + \
             "\n1 Directly (very slow)" + \
             "\n2 From saved 'stats.pkl' file (not as slow)" + \
             "\n3 Write data to 'stats.pkl' file\n"
    action = input(prompt)

    if action not in ['1', '2', '3']:
        raise Exception("Please input 1 or 2 or 3")

    if action == '1':
        print("Running Directly")
        run_model(directly=True)
    elif action == '2':
        print("Running from saved stats.pkl file")
        run_model(directly=False)
    else:
        prompt = 'What test/train split to use (input a float)?'
        split = input(prompt)
        try:
            split = float(split)
        except ValueError:
            print("Please enter a float.")
        if split <= 0 or split >= 1:
            raise Exception("Please enter a float between 0 and 1")
        write_to_pickle(split=split)


if __name__ == "__main__":
    main()
