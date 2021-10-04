import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def random_tests() :
    for episode in range(5) :
        env.reset()
        for t in range(goal_steps) :
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

#random_tests()

#initial data training 

def initial_population() :
    training_data = []  #we will only append a training data if the score is above 50 (score_requirement)
    scores = []
    accepted_scores = []

    #iterating through games
    for _ in range(initial_games):
        score = 0
        game_memory = []  #to store all the movements, because we won't know till the end of the game if we won or not
        prev_observation = []
        #actual games that happened
        for _ in range(goal_steps):
            #env.render()
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action) 

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action]) #taking prev observation with current action

            prev_observation = observation

            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0],output])

        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores)) 

    return training_data

def nn_model(input_size):
    network = input_data(shape = [None, input_size, 1], name = 'input')

    network = fully_connected(network, 128, activation='relu')   #relu = rectified linear 
    network = dropout(network, 0.8)  #0.8 keep rate

    network = fully_connected(network, 256, activation='relu')   #relu = rectified linear 
    network = dropout(network, 0.8)  #0.8 keep rate

    network = fully_connected(network, 512, activation='relu')   #relu = rectified linear 
    network = dropout(network, 0.8)  #0.8 keep rate

    network = fully_connected(network, 256, activation='relu')   #relu = rectified linear 
    network = dropout(network, 0.8)  #0.8 keep rate

    network = fully_connected(network, 128, activation='relu')   #relu = rectified linear 
    network = dropout(network, 0.8)  #0.8 keep rate

    #output layer
    network = fully_connected(network, 2, activation='softmax')   #2 outputs

    #regression
    network = regression(network, optimizer = 'adam', learning_rate=LR, loss='categorical_crossentropy', name = 'targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)  #training data contains observations and moves so X the observations
    Y = [i[1] for i in training_data]   #the moves

    if not model:
        model = nn_model(input_size = len(X[0]))
    
    model.fit({'input' : X}, {'targets' : Y}, n_epoch = 3, snapshot_step=500, show_metric=True, run_id='openaigym')


    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for games in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:  #if nothing
            action  = random.randrange(0,2)
        else : #after seeing a frame
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0]) 
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
scores.append(score)

print('Average score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))








