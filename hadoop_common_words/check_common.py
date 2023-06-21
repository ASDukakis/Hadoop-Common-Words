import sys


def mark(answer_path, output_path):
    with open(answer_path, "r") as f_in:
        ans = list(line.split() for line in (l.strip() for l in f_in) if line)
    with open(output_path, "r") as f_in:
        out = list(line.split() for line in (l.strip() for l in f_in) if line)

    if len(ans) != len(out):
        return -4
    for a, o in zip(ans, out):
        if a != o:
            return -1
    return 1


if __name__ == '__main__':
    result = mark(sys.argv[1], sys.argv[2])
    if result == 1:
        print("Test Passed")
    elif result == -4:
        print("Wrong Answer: wrong number of lines")
    else:
        print("Wrong Answer: wrong output")

