import numpy as np
import matplotlib.pyplot as plt


class FictiousPlay:
    # Assume it always have only 2 players in this assignment
    # Then we can take advantages of it when implementing the code
    def __init__(self, matrix, player_num = 2, action_num = 2):
        self.matrix = matrix
        self.player_num = player_num
        self.action_num = action_num
        self.action_name = [i + 1 for i in range(action_num)]
        self.players_payoff = np.zeros(shape = (player_num, action_num))
        self.players_belief = np.ones(shape = (player_num, action_num))
        self.players_action = ["NA" for i in range(player_num)]
        
    
    # calculate payoff depends on current belief
    def calculate_payoff(self):
        for i in range(self.player_num): # for each player
            for j in range(self.action_num): # for each strategy
                payoff = 0
                for k in range(self.action_num): # consider the belief of player i
                    payoff += self.players_belief[i][k] * self.matrix[i][j][k]
                self.players_payoff[i][j] = payoff

    # decide an action and update belief
    def take_action(self):
        for i in range(self.player_num): # each player take best response based on payoff
            max_indices = np.where(self.players_payoff[i] == np.max(self.players_payoff[i]))[0]
            random_max_index = np.random.choice(max_indices)
            #print("Player {} take action {}".format(i, self.action_name[np.argmax(self.players_payoff[i])]))
            for j in range(self.player_num):
                if j != i:
                    self.players_belief[j][random_max_index] += 1
            self.players_action[i] = self.action_name[random_max_index]

    # set initial player belief
    def set_belief(self, belief):
        self.players_belief = belief
    
    # play the game for n round or less when converge
    def play(self, round = 1000, threshold = 0.005, plot = False):
        print("Round\t\t1’s action  2’s action\t1’s belief\t2’s belief\t1’s payoff\t2’s payoff")
        prev1 = 0
        prev2 = 0
        history = []
        y = []
        for i in range(round):
            self.calculate_payoff()
            self.take_action()
            self.calculate_payoff()

            player1_total = sum(self.players_belief[0])
            player2_total = sum(self.players_belief[1])
            prob1_1 = self.players_belief[0][0] / player1_total
            prob1_2 = self.players_belief[0][1] / player1_total
            prob2_1 = self.players_belief[1][0] / player2_total
            prob2_2 = self.players_belief[1][1] / player2_total
            history.append(prob1_1)
            y.append(i+1)
            print("Round {:>5d}\t{:>5d}\t{:>8d}\t{:10s}\t{:10s}\t{:10s}\t{:10s}   ({:.2f}, {:.2f})  ({:.2f}, {:.2f})".format(i+1, 
                                                            self.players_action[0], self.players_action[1],
                                                            str(self.players_belief[0]),
                                                            str(self.players_belief[1]),
                                                            str(self.players_payoff[0]),
                                                            str(self.players_payoff[1]),
                                                            prob1_1, prob1_2,
                                                            prob2_1, prob2_2
                                                            ))
            change1 = abs(prob1_1 - prev1)
            change2 = abs(prob2_1 - prev2)
            if change1 <= threshold and change2 <= threshold:
                print("The game converges in round {}".format(i+1))
                break
            prev1 = prob1_1
            prev2 = prob2_1
        if plot:
            plt.plot(y, history)
            plt.show()


    
    

if __name__ == "__main__":
    # ex1 = FictiousPlay([
    #     [[-1, -3], [0, -2]],
    #     [[-1, -3], [0, -2]]
    # ])
    
    # ex1.players_belief[0][0] = 3
    # ex1.players_belief[0][1] = 2
    # ex1.players_belief[1][0] = 3
    # ex1.players_belief[1][1] = 2
    # ex1.play(10)

    # ex2 = FictiousPlay([
    #     [[1, -1], [-1, 1]],
    #     [[-1, 1], [1, -1]]
    # ])
    # ex2.players_belief[0][0] = 1.5
    # ex2.players_belief[0][1] = 2
    # ex2.players_belief[1][0] = 2
    # ex2.players_belief[1][1] = 1.5
    # ex2.play(10)

    # Q1 = FictiousPlay([
    #     [[-1, 1], [0, 3]],
    #     [[-1, 1], [0, 3]]
    # ])
    # Q1.play(threshold=0.0005, plot=True)

    # Q2 = FictiousPlay([
    #     [[2, 1], [0, 3]],
    #     [[2, 1], [0, 3]]
    # ])
    # Q2.play(threshold=0.00001, plot=True)

    Q3 = FictiousPlay([
        [[1, 0], [0, 0]],
        [[1, 0], [0, 0]]
    ])
    Q3.play(threshold=0.00001, plot=False)

    # Q4 = FictiousPlay([
    #     [[0, 2], [2, 0]],
    #     [[1, 0], [0, 4]]
    # ])
    # Q4.play(round=5000, threshold=0.0001, plot=True)

    # Q5 = FictiousPlay([
    #     [[0, 1], [1, 0]],
    #     [[1, 0], [0, 1]]
    # ])
    # Q5.play(round=6000, threshold=0.0001, plot=True)

    # Q6 = FictiousPlay([
    #     [[10, 0], [0, 10]],
    #     [[10, 0], [0, 10]]
    # ])
    # Q6.set_belief(np.array([[1, 1.5], [1.5, 1]])) # for mixed-strategy only
    # Q6.play(round=1000, threshold=0.0001, plot=True)

    # Q7 = FictiousPlay([
    #     [[0, 1], [1, 0]],
    #     [[0, 1], [1, 0]]
    # ])
    # Q7.set_belief(np.array([[1, 1.5], [1, 1.5]])) # for mixed-strategy only
    # Q7.play(threshold=0.0001, plot=True)

    # Q8 = FictiousPlay([
    #     [[3, 0], [0, 2]],
    #     [[2, 0], [0, 3]]
    # ])
    # Q8.set_belief(np.array([[1.25, 1], [1, 1.25]])) # for mixed-strategy only
    # Q8.play(threshold=0.00001, plot=True)

    # Q9 = FictiousPlay([
    #     [[3, 0], [2, 1]],
    #     [[3, 0], [2, 1]]
    # ])
    # Q9.set_belief(np.array([[1.25, 1], [1, 1.25]])) # for mixed-strategy only
    # Q9.play(threshold=0.0001, plot=True)

    # Q10 = FictiousPlay([
    #     [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    #     [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    # ], player_num=2, action_num=3)
    # Q10.set_belief(np.array([[0, 1, 0], [1, 0, 0]]))
    # Q10.play(round=3000, threshold=0.0001, plot=True)
    print("Finish")