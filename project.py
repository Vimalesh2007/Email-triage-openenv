import os
import random
from pydantic import BaseModel



API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy")



class Observation(BaseModel):
    email_text: str


class Action(BaseModel):
    action: int


class Reward(BaseModel):
    value: float



EMAIL_DATA = {

    "easy": [
        {"text": "Urgent: server down", "label": 1},
        {"text": "Meeting tomorrow at 10AM", "label": 2},
    ],

    "medium": [
        {"text": "Client asking project update", "label": 3},
        {"text": "Deadline reminder for report", "label": 1},
    ],

    "hard": [
        {"text": "50% discount offer", "label": 0},
        {"text": "Newsletter subscription confirmation", "label": 0},
    ]
}


def calculate_reward(predicted, actual):

    if predicted == actual:
        return 1.0

    if predicted in [1, 2, 3] and actual != 0:
        return 0.5

    return 0.0




class EmailTriageEnv:

    def __init__(self, task="easy"):

        self.task = task
        self.dataset = EMAIL_DATA[task]
        self.pointer = 0


    def reset(self):

        self.pointer = 0
        first_mail = self.dataset[self.pointer]

        return Observation(email_text=first_mail["text"])


    def state(self):

        if self.pointer >= len(self.dataset):
            return None

        mail = self.dataset[self.pointer]
        return Observation(email_text=mail["text"])


    def step(self, action: Action):

        current_mail = self.dataset[self.pointer]

        correct_label = current_mail["label"]

        reward_score = calculate_reward(action.action, correct_label)

        reward = Reward(value=reward_score)

        self.pointer += 1

        finished = self.pointer >= len(self.dataset)

        next_state = None if finished else self.state()

        meta = {
            "correct_action": correct_label,
            "task": self.task
        }

        return next_state, reward.value, finished, meta



Q_MEMORY = {}

LR = 0.1
DISCOUNT = 0.9
EXPLORATION = 0.3


def q_value(state, action):
    return Q_MEMORY.get((state, action), 0)


def select_action(state):

    if random.random() < EXPLORATION:
        return random.randint(0, 3)

    values = [q_value(state, a) for a in range(4)]

    return values.index(max(values))


def learn(state, action, reward, next_state):

    current_q = q_value(state, action)

    if next_state:
        future_q = max([q_value(next_state, a) for a in range(4)])
    else:
        future_q = 0

    updated_q = current_q + LR * (reward + DISCOUNT * future_q - current_q)

    Q_MEMORY[(state, action)] = updated_q



def execute_task(task):

    env = EmailTriageEnv(task)

    observation = env.reset()

    finished = False

    total = 0

    step_number = 0

    print(f"\nRunning Task: {task}")

    while not finished:

        current_state = observation.email_text

        action_id = select_action(current_state)

        action = Action(action=action_id)

        next_obs, reward, finished, info = env.step(action)

        next_state = None if next_obs is None else next_obs.email_text

        learn(current_state, action_id, reward, next_state)

        print(
            f"[STEP] {step_number} | "
            f"email='{current_state}' | "
            f"action={action_id} | "
            f"reward={reward}"
        )

        total += reward

        observation = next_obs

        step_number += 1

    return total



def main():

    print("[START] Email Triage RL Simulation")

    results = {}

    for level in ["easy", "medium", "hard"]:

        score = execute_task(level)

        results[level] = score

    print("\nFinal Scores:", results)

    print("[DONE] Execution Finished")


if __name__ == "__main__":

    main()