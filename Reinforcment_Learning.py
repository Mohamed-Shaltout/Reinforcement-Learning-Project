import numpy as np
import tkinter as tk

class GridWorldMDP:
    def __init__(self, grid, gamma=0.99):
        self.grid = grid
        self.grid_size = len(grid), len(grid[0])
        self.states = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])]
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = gamma
        self.terminal_states = [(0, 2)]  # Predefined terminal state (for example)
        
        # Add terminal states based on grid rewards
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i][j] == 10:  # +10 reward is a terminal state
                    self.terminal_states.append((i, j))
                if self.grid[i][j] == r:  # r is a terminal state
                    self.terminal_states.append((i, j))

    def get_reward(self, state):
        i, j = state
        return self.grid[i][j]

    def get_next_state(self, state, action):
        i, j = state
        if action == 'up':
            return (max(i - 1, 0), j)
        elif action == 'down':
            return (min(i + 1, self.grid_size[0] - 1), j)
        elif action == 'left':
            return (i, max(j - 1, 0))
        elif action == 'right':
            return (i, min(j + 1, self.grid_size[1] - 1))
        return state

    def transition(self, state, action):
        if state in self.terminal_states:
            return [(state, 1.0)]

        intended = action
        right_angle = {
            'up': ['left', 'right'],
            'down': ['left', 'right'],
            'left': ['up', 'down'],
            'right': ['up', 'down']
        }

        transitions = []
        next_state = self.get_next_state(state, intended)
        transitions.append((next_state, 0.8))

        for side_action in right_angle[action]:
            next_state = self.get_next_state(state, side_action)
            transitions.append((next_state, 0.1))

        transition_dict = {}
        for next_state, prob in transitions:
            if next_state in transition_dict:
                transition_dict[next_state] += prob
            else:
                transition_dict[next_state] = prob

        return list(transition_dict.items())

    def policy_evaluation(self, policy, epsilon=0.01):
        values = {state: 0 for state in self.states}
        while True:
            delta = 0
            new_values = values.copy()
            for state in self.states:
                if state in self.terminal_states:
                    continue
                action = policy[state]
                transitions = self.transition(state, action)
                new_values[state] = sum(
                    prob * (self.get_reward(next_state) + self.gamma * values[next_state])                   
                    for next_state, prob in transitions
                )
                delta = max(delta, abs(values[state] - new_values[state]))
            values = new_values
            if delta < epsilon:
                break
        return values

    def policy_improvement(self, values):
        policy = {state: None for state in self.states}
        for state in self.states:
            if state in self.terminal_states:
                continue
            action_values = {}
            for action in self.actions:
                transitions = self.transition(state, action)
                action_value = sum(
                    prob * (self.get_reward(next_state) + self.gamma * values[next_state])
                    for next_state, prob in transitions
                )
                action_values[action] = action_value
            policy[state] = max(action_values, key=action_values.get)
        return policy

    def policy_iteration(self):
        policy = {state: np.random.choice(self.actions) for state in self.states}
        for state in self.terminal_states:
            policy[state] = None

        while True:
            values = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(values)
            if new_policy == policy:
                break
            policy = new_policy
        return values, policy

    def value_iteration(self, epsilon=0.01):
        values = {state: 0 for state in self.states}
        policy = {state: None for state in self.states}

        while True:
            delta = 0
            new_values = values.copy()

            for state in self.states:
                if state in self.terminal_states:
                    continue
                
                action_values = []
                for action in self.actions:
                    transitions = self.transition(state, action)
                    action_value = sum(
                        prob * (self.get_reward(next_state) + self.gamma * values[next_state])
                        for next_state, prob in transitions
                    )
                    action_values.append(action_value)
                
                new_values[state] = max(action_values)
                delta = max(delta, abs(values[state] - new_values[state]))

            values = new_values
            if delta < epsilon:
                break

        for state in self.states:
            if state in self.terminal_states:
                continue

            action_values = {}
            for action in self.actions:
                transitions = self.transition(state, action)
                action_value = sum(
                    prob * (self.get_reward(next_state) + self.gamma * values[next_state])
                    for next_state, prob in transitions
                )
                action_values[action] = action_value

            policy[state] = max(action_values, key=action_values.get)

        return values, policy

    def display_policy_gui(self, policy, values, r):
        root = tk.Tk()
        root.title(f"Optimal Policy and Values for r = {r}")

        label_r = tk.Label(root, text=f"Current Reward for Non-Terminal States: r = {r}", font=("Arial", 14))
        label_r.grid(row=0, column=0, padx=20, pady=10)

        canvas = tk.Canvas(root, width=300, height=300)
        canvas.grid(row=1, column=0, padx=20, pady=20)

        cell_size = 80
        arrow_map = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                if (i, j) in self.terminal_states:
                    canvas.create_rectangle(x1, y1, x2, y2, fill="yellow")
                    canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="T", font=("Arial", 16, "bold"))
                    value_text = f"{self.get_reward((i, j)):.2f}"  # Show value of terminal state (like +10)
                    canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2 + 15, text=value_text, font=("Arial", 10))

                else:
                    canvas.create_rectangle(x1, y1, x2, y2, fill="lightblue")
                    action = policy.get((i, j), '')
                    arrow = arrow_map.get(action, '')  # Get the corresponding arrow symbol
                    canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=arrow, font=("Arial", 16, "bold"))

                    # Show only v(s) for non-terminal states
                    value_text = f"{values.get((i, j), 0):.2f}"  # Show calculated value for non-terminal states
                    canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2 + 15, text=value_text, font=("Arial", 10))

        root.mainloop()




    def print_policy(self, policy, values):
        print("Optimal Policy:")
        for i in range(self.grid_size[0]):
            row_policy = []
            for j in range(self.grid_size[1]):
                action = policy.get((i, j), None)
                row_policy.append(action[0].upper() if action else 'T')
            print(" ".join(row_policy))
        
        print("\nState Values:")
        for i in range(self.grid_size[0]):
            row_values = []
            for j in range(self.grid_size[1]):
                if (i, j) in self.terminal_states:
                    value = self.get_reward((i, j))  # Terminal states have their own reward
                else:
                    value = values.get((i, j), 0)  # Non-terminal states use calculated values
                row_values.append(f"{value:.2f}")
            print(" ".join(row_values))


grid_template = [
    [None, -1, 10],
    [-1, -1, -1],
    [-1, -1, -1],
]

for r in [100, 3, 0, -3]:
    print(f"\nPolicy Iteration with r = {r}")
    grid = [[r if cell is None else cell for cell in row] for row in grid_template]
    mdp = GridWorldMDP(grid, gamma=0.99)

    values_policy_iter, policy_policy_iter = mdp.policy_iteration()
    mdp.print_policy(policy_policy_iter, values_policy_iter)

    print(f"\nValue Iteration with r = {r}")
    values_value_iter, policy_value_iter = mdp.value_iteration()
    mdp.print_policy(policy_value_iter, values_value_iter)

    mdp.display_policy_gui(policy_policy_iter, values_policy_iter,r)