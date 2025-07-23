import asyncio
import os
import sys
import json
import secrets
import ast
from collections import defaultdict
from dotenv import load_dotenv
from agents import Agent, Runner, trace

from typing import List, Tuple

import logging

# Set root logger to WARNING to suppress most info logs from dependencies
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)


class RLProfilePolicy:
    def __init__(
            self,
            profiles: List[str],
            alpha: float = 0.1,
            gamma: float = 0.9,
            epsilon: float = 0.2,
            policy_file: str = "psych_policy.json"
    ):
        self.profiles = profiles
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.policy_file = policy_file
        self.load()

    def select_action(self, state: Tuple) -> int:
        """Epsilon-greedy action selection."""
        # Use secrets instead of random for Bandit compliance (not needed for RL, but silences warning)
        if secrets.randbelow(1000000) / 1000000 < self.epsilon:
            return secrets.randbelow(len(self.profiles))
        qs = [self.q_table[(state, a)] for a in range(len(self.profiles))]
        return int(qs.index(max(qs)))

    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> None:
        max_next = max([self.q_table[(next_state, a)] for a in range(len(self.profiles))])
        old = self.q_table[(state, action)]
        self.q_table[(state, action)] = old + self.alpha * (reward + self.gamma * max_next - old)
        self.save()

    def save(self) -> None:
        try:
            qdict = {str(k): v for k, v in self.q_table.items()}
            with open(self.policy_file, "w") as f:
                json.dump(qdict, f)
        except Exception as e:
            logger.error(f"Could not save RL policy: {e}")

    def load(self) -> None:
        if not os.path.exists(self.policy_file):
            return
        try:
            with open(self.policy_file, "r") as f:
                qdict = json.load(f)
                # Use ast.literal_eval for safety (not eval)
                self.q_table = defaultdict(float, {
                    ast.literal_eval(k): v for k, v in qdict.items()
                })
        except Exception as e:
            logger.warning(f"Could not load RL policy: {e}")


PSYCH_PROFILES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

REWARD_CORRECT = 10
REWARD_INCORRECT = -3

policy = RLProfilePolicy(PSYCH_PROFILES)

llm_agent = Agent(
    name="LLM Question Generator",
    instructions="""
You are a psychological assessment assistant. Your job is to:
1. Generate one short, open-ended question that helps reveal if a person matches the following personality 
    trait: {profile}.
2. The question should be about everyday life, not clinical, and phrased neutrally. Example topics: work, social life, 
    routines, preferences.
3. Ask only one question. Never mention any trait name in the question.
4. Do not reference personality theory. Avoid closed (yes/no) questions.
""",
    tools=[],
    handoff_description="Generate neutral everyday questions for psychological profiling."
)

llm_classifier = Agent(
    name="LLM Profile Classifier",
    instructions="""
You are a psychological traits classifier. Your job is to:
1. Analyze the user's answer and decide which of the following Big Five personality traits it best reflects:
   Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.
2. Respond with only the name of the trait that most closely matches the answer, or 'Unknown' if you cannot decide.
3. Do not explain your answer, do not mention personality theory, just output the trait name.
""",
    tools=[],
    handoff_description="Classify user answers into Big Five traits."
)


def get_state(history: List[str]) -> Tuple:
    return tuple(history[-3:]) if history else ("start",)


async def main() -> None:
    print("Welcome, let's have a quick conversation.")
    history: List[str] = []
    try:
        for step in range(7):
            state = get_state(history)
            profile_idx = policy.select_action(state)
            profile = PSYCH_PROFILES[profile_idx]
            prompt = f"Generate a question for {profile} trait."
            print(f"\n[Agent is generating a question to test: {profile}]")
            with trace("LLM question generation"):
                llm_result = await Runner.run(llm_agent, prompt.format(profile=profile))
                question = llm_result.final_output.strip().split('\n')[0]
            print(f"[Agent] Q{step + 1}: {question}")
            answer = input("Your answer: ")
            classify_prompt = (f"User answer: {answer}\n\nWhich of these traits does this most reflect? "
                               "Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.")
            with trace("LLM answer classification"):
                classification_result = await Runner.run(llm_classifier, classify_prompt)
                classified_profile = classification_result.final_output.strip().split('\n')[0]
            if classified_profile == profile:
                reward = REWARD_CORRECT
                print(f"[Agent] Your answer matched the profile: {profile}")
            else:
                reward = REWARD_INCORRECT
                print(f"[Agent] Your answer was classified as: {classified_profile}. Target was: {profile}")
            history.append(answer)
            next_state = get_state(history)
            policy.update(state, profile_idx, reward, next_state)
            print(f"[RL Q-table update] state={state}, action={profile}, reward={reward}, next_state={next_state}")
            print(f"[Current Q-values for state]: {[policy.q_table[(state, a)] for a in range(len(PSYCH_PROFILES))]}")
        print("\n[Agent] Thanks! My best guess for your profile is:")
        final_state = get_state(history)
        best_profile_idx = policy.select_action(final_state)
        print(f"{PSYCH_PROFILES[best_profile_idx]}")
    except KeyboardInterrupt:
        print("\n[Agent] Conversation interrupted.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
