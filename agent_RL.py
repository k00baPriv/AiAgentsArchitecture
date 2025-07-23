import asyncio
import os
import sys
import json
import secrets
from collections import defaultdict
from dotenv import load_dotenv
from agents import Agent, Runner, trace

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)


class RLPreferencePolicy:
    """
    Reinforcement Learning policy using Q-learning to select response styles
    based on user feedback.

    Attributes:
        styles (list): List of possible answer styles.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        q_table (defaultdict): Q-value table mapping (state, action) to value.
        policy_file (str): File to persist policy to/from.
    """

    def __init__(self, styles, alpha=0.1, gamma=0.9, epsilon=0.1, policy_file="policy.json"):
        self.styles = styles
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.policy_file = policy_file
        self.load()

    def select_action(self, state):
        """
        Select an action (style index) using an epsilon-greedy strategy.

        Args:
            state (tuple): Current state.

        Returns:
            int: Index of the chosen action.
        """
        # Use secrets for random action selection (compliant with Bandit B311)
        if secrets.randbelow(10000) < int(self.epsilon * 10000):
            return secrets.randbelow(len(self.styles))
        qs = [self.q_table[(state, a)] for a in range(len(self.styles))]
        return int(qs.index(max(qs)))

    def update(self, state, action, reward, next_state):
        """
        Update the Q-value for a given state-action pair.

        Args:
            state (tuple): Previous state.
            action (int): Action taken.
            reward (float): Observed reward.
            next_state (tuple): Resulting state after taking action.
        """
        max_next = max([self.q_table[(next_state, a)] for a in range(len(self.styles))])
        old = self.q_table[(state, action)]
        self.q_table[(state, action)] = old + self.alpha * (reward + self.gamma * max_next - old)
        self.save()

    def save(self):
        """Save the Q-table to a JSON file."""
        try:
            # Convert tuple keys to strings for JSON serialization
            serializable = {str(k): v for k, v in self.q_table.items()}
            with open(self.policy_file, "w") as f:
                json.dump(serializable, f)
        except Exception as e:
            print(f"[ERROR] Could not save RL policy: {e}")

    def load(self):
        """Load the Q-table from a JSON file if it exists."""
        try:
            with open(self.policy_file, "r") as f:
                loaded = json.load(f)
                # Convert keys back to tuples (use ast.literal_eval for safety)
                import ast
                self.q_table = defaultdict(
                    float,
                    {ast.literal_eval(k): v for k, v in loaded.items()}
                )
        except Exception as e:
            print(f"[WARN] Could not load RL policy: {e}")


# Definition of the AI agent responsible for answering questions
llm_agent = Agent(
    name="LLM searcher",
    instructions="""
You are a specialized AI assistant with deep knowledge of programming.
Your role is to:
1. Solve programming problems and answer coding questions.
2. When prompted, include code tests, mermaid diagrams,
   or mathematical explanations as requested in the prompt.
3. Always explain your reasoning clearly and be concise and accurate.
""",
    tools=[],
    handoff_description="Solve programming problems and provide extras when prompted."
)

# Different response configurations for the agent
ANSWER_STYLES = [
    {"tests": False, "mermaid": False, "math": False},
    {"tests": True, "mermaid": False, "math": False},
    {"tests": False, "mermaid": True, "math": False},
    {"tests": False, "mermaid": False, "math": True},
    {"tests": True, "mermaid": True, "math": False},
    {"tests": True, "mermaid": True, "math": True},
    {"tests": True, "mermaid": False, "math": True},
]

# Constants
EXTRAS = ["tests", "mermaid", "math"]
REWARD_SUCCESS = 10
REWARD_FAILURE = -2

# Instantiate the RL policy
policy = RLPreferencePolicy(ANSWER_STYLES)


def build_prompt(msg, style):
    """
    Construct a prompt based on the user's question and the chosen style.

    Args:
        msg (str): The user input question.
        style (dict): Style configuration.

    Returns:
        str: The full prompt to send to the agent.
    """
    if style["tests"]:
        msg += "\nPlease include code tests."
    if style["mermaid"]:
        msg += "\nPlease include a mermaid diagram."
    if style["math"]:
        msg += "\nPlease explain the mathematical background."
    return msg


def get_state(preferences):
    """
    Create a state representation from user preferences.

    Args:
        preferences (list): List of missing extras.

    Returns:
        tuple: State tuple to be used by the RL policy.
    """
    return tuple(sorted(preferences)) if preferences else ("none",)


async def main():
    """
    Main interaction loop between the user and the agent.
    Uses reinforcement learning to adapt response styles based on user feedback.
    """
    preference_history = []

    while True:
        msg = input("\nAsk me a programming problem (or type 'exit' to quit):\n")
        if msg.strip().lower() == "exit":
            break

        state = get_state(preference_history)
        style_idx = policy.select_action(state)
        style = ANSWER_STYLES[style_idx]
        prompt = build_prompt(msg, style)

        print(f"\n[Agent is answering with style: {style}]")

        with trace("LLM agent evaluator"):
            llm_result = await Runner.run(llm_agent, prompt)
            print("\n==== AGENT ANSWER ====\n")
            print(llm_result.final_output)
            print("\n======================\n")

        # Ask for feedback on what's missing
        missing = []
        while True:
            followup = input(
                "Anything missing? (type 'tests', 'mermaid', 'math' or 'ok' if happy):\n"
            ).strip().lower()
            if followup == "ok":
                break
            elif followup in EXTRAS and followup not in missing:
                missing.append(followup)

        # Update the policy with feedback
        reward = REWARD_SUCCESS if not missing else REWARD_FAILURE * len(missing)
        next_state = get_state(missing)
        policy.update(state, style_idx, reward, next_state)
        preference_history = missing

        print(f"[RL updated: state={state}, action={style_idx}, "
              f"reward={reward}, next={next_state}]")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
