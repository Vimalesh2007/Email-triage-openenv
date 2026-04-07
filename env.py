import random
from pydantic import BaseModel


# --------------------------------------------------------------------------- #
#  Data Models
# --------------------------------------------------------------------------- #

class Observation(BaseModel):
    email_text: str


class Action(BaseModel):
    action: int


class Reward(BaseModel):
    value: float


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

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
    ],
}

# Action label mapping (for reference / logging)
ACTION_LABELS = {
    0: "Spam / Promotional",
    1: "Urgent",
    2: "Meeting / Schedule",
    3: "Follow-up / Update",
}


# --------------------------------------------------------------------------- #
#  Reward Function
# --------------------------------------------------------------------------- #

def calculate_reward(predicted: int, actual: int) -> float:
    """
    Returns:
        1.0  – exact match
        0.5  – predicted a non-spam category for a non-spam email (partial credit)
        0.0  – otherwise
    """
    if predicted == actual:
        return 1.0
    if predicted in [1, 2, 3] and actual != 0:
        return 0.5
    return 0.0


# --------------------------------------------------------------------------- #
#  Environment
# --------------------------------------------------------------------------- #

class EmailTriageEnv:
    """Simple sequential email-triage environment."""

    def __init__(self, task: str = "easy"):
        if task not in EMAIL_DATA:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(EMAIL_DATA)}")
        self.task = task
        self.dataset = EMAIL_DATA[task]
        self.pointer: int = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        """Reset environment to the first email."""
        self.pointer = 0
        return Observation(email_text=self.dataset[self.pointer]["text"])

    # ------------------------------------------------------------------ #
    def state(self) -> Observation | None:
        if self.pointer >= len(self.dataset):
            return None
        return Observation(email_text=self.dataset[self.pointer]["text"])

    # ------------------------------------------------------------------ #
    def step(self, action: Action):
        """
        Returns
        -------
        next_state : Observation | None
        reward     : float
        finished   : bool
        meta       : dict
        """
        current_mail = self.dataset[self.pointer]
        correct_label = current_mail["label"]
        reward_score = calculate_reward(action.action, correct_label)

        self.pointer += 1
        finished = self.pointer >= len(self.dataset)
        next_state = None if finished else self.state()

        meta = {
            "correct_action": correct_label,
            "correct_label_name": ACTION_LABELS.get(correct_label, "Unknown"),
            "predicted_label_name": ACTION_LABELS.get(action.action, "Unknown"),
            "task": self.task,
        }
        return next_state, reward_score, finished, meta


# --------------------------------------------------------------------------- #
#  Q-Learning Agent
# --------------------------------------------------------------------------- #

Q_MEMORY: dict = {}
LR = 0.1
DISCOUNT = 0.9
EXPLORATION = 0.3


def q_value(state: str, action: int) -> float:
    return Q_MEMORY.get((state, action), 0.0)


def select_action(state: str) -> int:
    """Epsilon-greedy action selection."""
    if random.random() < EXPLORATION:
        return random.randint(0, 3)
    values = [q_value(state, a) for a in range(4)]
    return values.index(max(values))


def learn(state: str, action: int, reward: float, next_state: str | None) -> None:
    """Q-value update (Bellman equation)."""
    current_q = q_value(state, action)
    future_q = max(q_value(next_state, a) for a in range(4)) if next_state else 0.0
    updated_q = current_q + LR * (reward + DISCOUNT * future_q - current_q)
    Q_MEMORY[(state, action)] = up

def execute_task(task: str) -> float:
    env = EmailTriageEnv(task)
    observation = env.reset()
    finished = False
    total_reward = 0.0
    step_number = 0

    print(f"\n{'='*55}")
    print(f"  Running Task: {task.upper()}")
    print(f"{'='*55}")

    while not finished:
        current_state = observation.email_text
        action_id = select_action(current_state)
        action = Action(action=action_id)

        next_obs, reward, finished, info = env.step(action)
        next_state = None if next_obs is None else next_obs.email_text

        learn(current_state, action_id, reward, next_state)

        print(
            f"  Step {step_number:02d} | email='{current_state}'\n"
            f"          predicted='{info['predicted_label_name']}' "
            f"| correct='{info['correct_label_name']}' "
            f"| reward={reward:.2f}"
        )
        total_reward += reward
        observation = next_obs
        step_number += 1

    print(f"  Task total reward: {total_reward:.2f}")
    return total_reward
