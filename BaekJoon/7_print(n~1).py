# 자연수 N이 주어졌을 때, N부터 1까지 한 줄에 하나씩 출력하는 프로그램을 작성하시오.
# 첫째 줄에 100,000보다 작거나 같은 자연수 N이 주어진다.
# 첫째 줄부터 N번째 줄 까지 차례대로 출력한다.

# my_list = [1,2,3,4,5]
 
# for item in my_list[::-1]: #[::-1] 역으로 슬라이싱
#     print(item)



number = int(input())

for i in range(number, 0, -1) :
    print(i)