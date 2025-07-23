import asyncio
import os
import sys
import pickle
import random
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

# --- RL Policy ---
class RLPreferencePolicy:
    """
    Simple Q-learning policy to select answer styles.
    State: tuple of last feedbacks (e.g. ('tests', 'math'))
    Action: answer_style (int, representing which extras to include)
    """
    def __init__(self, styles, alpha=0.1, gamma=0.9, epsilon=0.1, policy_file="policy.pkl"):
        self.styles = styles
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.policy_file = policy_file
        self.load()

    def select_action(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, len(self.styles)-1)
        qs = [self.q_table[(state, a)] for a in range(len(self.styles))]
        return int(qs.index(max(qs)))

    def update(self, state, action, reward, next_state):
        max_next = max([self.q_table[(next_state, a)] for a in range(len(self.styles))])
        old = self.q_table[(state, action)]
        self.q_table[(state, action)] = old + self.alpha * (reward + self.gamma * max_next - old)
        self.save()

    def save(self):
        try:
            with open(self.policy_file, "wb") as f:
                pickle.dump(dict(self.q_table), f)
        except Exception as e:
            print(f"Could not save RL policy: {e}")

    def load(self):
        try:
            with open(self.policy_file, "rb") as f:
                qdict = pickle.load(f)
                self.q_table = defaultdict(float, qdict)
        except Exception:
            pass

# --- Define Your Single LLM Agent ---

from agents import Agent, Runner, trace

llm_agent = Agent(
    name="LLM searcher",
    instructions="""You are a specialized AI assistant with deep knowledge of programming. Your role is to:
1. Solve programming problems and answer coding questions.
2. When prompted, include code tests, mermaid diagrams, or mathematical explanations as requested in the prompt.
3. Always explain your reasoning clearly and be concise and accurate.""",
    tools=[],
    handoff_description="Solve programming problems and provide extras when prompted."
)

# --- RL answer styles ---
ANSWER_STYLES = [
    {"tests": False, "mermaid": False, "math": False},
    {"tests": True,  "mermaid": False, "math": False},
    {"tests": False, "mermaid": True,  "math": False},
    {"tests": False, "mermaid": False, "math": True},
    {"tests": True,  "mermaid": True,  "math": False},
    {"tests": True,  "mermaid": True,  "math": True},
    {"tests": True,  "mermaid": False, "math": True},
]

policy = RLPreferencePolicy(ANSWER_STYLES)

# --- Main conversational RL loop ---

async def main():
    preference_history = []  # Store the last N extras user asked for

    while True:
        msg = input("\nAsk me a programming problem (or type 'exit' to quit):\n")
        if msg.strip().lower() == "exit":
            break

        # RL STATE: Use last feedback as state (could expand to N-feedbacks)
        state = tuple(sorted(preference_history)) if preference_history else ("none",)

        # RL selects the answer style (i.e., extras to include)
        style_idx = policy.select_action(state)
        style = ANSWER_STYLES[style_idx]

        # Build prompt for LLM agent
        prompt = msg
        if style["tests"]:
            prompt += "\nPlease include code tests."
        if style["mermaid"]:
            prompt += "\nPlease include a mermaid diagram."
        if style["math"]:
            prompt += "\nPlease explain the mathematical background."

        print(f"\n[Agent is answering with style: {style}]")

        # Call your single LLM agent
        with trace("LLM agent evaluator"):
            llm_result = await Runner.run(llm_agent, prompt)
            print("\n==== AGENT ANSWER ====\n")
            print(llm_result.final_output)
            print("\n=====================\n")

        # --- USER FEEDBACK/INTERACTION ---
        extras = ["tests", "mermaid", "math"]
        missing = []
        while True:
            followup = input("Anything missing? (type 'tests', 'mermaid', 'math' or 'ok' if happy):\n").strip().lower()
            if followup == "ok":
                break
            elif followup in extras and followup not in missing:
                missing.append(followup)

        # REWARD: if nothing missing, big reward; else, penalize per missing
        if not missing:
            reward = 10
            print("Agent: Yay, I anticipated your needs!")
        else:
            reward = -2 * len(missing)
            print(f"Agent: I'll remember you like: {missing}")

        # Update RL policy
        next_state = tuple(sorted(missing)) if missing else ("none",)
        policy.update(state, style_idx, reward, next_state)
        preference_history = missing  # Move to next state

        print(f"[RL updated: state={state} action={style_idx} reward={reward} next={next_state}]")

if __name__ == "__main__":
    asyncio.run(main())
