import sys

# input 보다 속도가 빠름

t = sys.stdin.readline()

T = int(t.rstrip())

for i in range(1, T+1):
    line = input()
    a = line.split(" ")[0]
    b = line.split(" ")[1]
    print(int(a) + int(b))