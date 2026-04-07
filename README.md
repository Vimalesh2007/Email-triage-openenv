# Email Triage RL Simulation

A lightweight **Reinforcement Learning** demo that trains a Q-learning agent to classify and triage emails into one of four categories:

| Action ID | Category |
|-----------|----------|
| 0 | Spam / Promotional |
| 1 | Urgent |
| 2 | Meeting / Schedule |
| 3 | Follow-up / Update |

---

## Project Structure

```
.
├── env.py            # Environment, dataset, reward function, Q-learning agent
├── inference.py      # CLI entry-point / runner
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
└── README.md         # This file
```

---

## How It Works

```
┌─────────────┐     Observation(email_text)     ┌─────────────┐
│  EmailTriage│ ─────────────────────────────►  │  Q-Learning │
│     Env     │                                  │    Agent    │
│             │ ◄─────────────────────────────  │             │
└─────────────┘        Action(action_id)         └─────────────┘
       │                                                │
       │  Reward  +  Next State                         │
       └────────────────────────────────────────────────┘
                     learn() → update Q-table
```

1. The environment yields one email at a time as an `Observation`.
2. The agent picks an action using **ε-greedy** exploration (`EXPLORATION = 0.3`).
3. A reward is calculated:
   - **1.0** – exact match
   - **0.5** – predicted a non-spam category for a non-spam email
   - **0.0** – otherwise
4. The Q-table is updated via the **Bellman equation**.

---

## Quick Start

### Local (Python 3.11+)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all difficulty levels (easy / medium / hard)
python inference.py

# 3. Run a single level
python inference.py --task easy

# 4. Run multiple episodes (more training)
python inference.py --episodes 10
```

### Docker

```bash
# Build
docker build -t email-triage-rl .

# Run (default: all tasks, 1 episode)
docker run --rm email-triage-rl

# Run with arguments
docker run --rm email-triage-rl --task hard --episodes 5
```

---

## Configuration

All hyperparameters live at the top of `env.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | `0.1` | Q-learning rate |
| `DISCOUNT` | `0.9` | Future reward discount factor (γ) |
| `EXPLORATION` | `0.3` | ε — probability of random action |

---

## Environment Variables

These are accepted but not required for the core simulation (legacy from OpenAI integration):

| Variable | Default |
|----------|---------|
| `API_BASE_URL` | `https://api.openai.com/v1` |
| `MODEL_NAME` | `gpt-4o-mini` |
| `HF_TOKEN` | `dummy` |

---

## Example Output

```
=======================================================
   Email Triage RL Simulation  —  START
=======================================================
   Tasks    : ['easy', 'medium', 'hard']
   Episodes : 1

=======================================================
  Running Task: EASY
=======================================================
  Step 00 | email='Urgent: server down'
          predicted='Urgent' | correct='Urgent' | reward=1.00
  Step 01 | email='Meeting tomorrow at 10AM'
          predicted='Spam / Promotional' | correct='Meeting / Schedule' | reward=0.00
  Task total reward: 1.00
...
=======================================================
  FINAL AVERAGE SCORES
=======================================================
  easy    : 1.0
  medium  : 1.5
  hard    : 0.5
=======================================================
```
