weight = 0.5    # 첫번째는 임의 또는 랜덤하게 주는거임
input = 0.5     # 초기값
goal_prediction = 0.8       # 튜닝 불가
lr = 0.05
epochs = 70

for iteration in range(epochs):
    prediction  = input * weight
    error = (prediction - goal_prediction) **2     # mse

    print("Error :" +  str(error) + "\tPrediction :" + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) **2     # mse

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2     # mse

    if(down_error < up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr