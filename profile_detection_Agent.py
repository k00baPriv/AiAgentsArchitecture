import asyncio
import os
import sys
import pickle
import random
from collections import defaultdict
from dotenv import load_dotenv
from agents import Agent, Runner, trace

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)


class RLProfilePolicy:
    def __init__(self, profiles, alpha=0.1, gamma=0.9, epsilon=0.2, policy_file="psych_policy.pkl"):
        self.profiles = profiles
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.policy_file = policy_file
        self.load()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.profiles) - 1)
        qs = [self.q_table[(state, a)] for a in range(len(self.profiles))]
        return int(qs.index(max(qs)))

    def update(self, state, action, reward, next_state):
        max_next = max([self.q_table[(next_state, a)] for a in range(len(self.profiles))])
        old = self.q_table[(state, action)]
        self.q_table[(state, action)] = old + self.alpha * (reward + self.gamma * max_next - old)
        self.save()

    def save(self):
        try:
            with open(self.policy_file, "wb") as f:
                pickle.dump(dict(self.q_table), f)
        except Exception as e:
            print(f"[ERROR] Could not save RL policy: {e}")

    def load(self):
        try:
            with open(self.policy_file, "rb") as f:
                qdict = pickle.load(f)
                self.q_table = defaultdict(float, qdict)
        except Exception:
            pass


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
1. Generate one short, open-ended question that helps reveal if a person matches the following personality trait: {profile}.
2. The question should be about everyday life, not clinical, and phrased neutrally. Example topics: work, social life, routines, preferences.
3. Ask only one question. Never mention any trait name in the question.
4. Do not reference personality theory. Avoid closed (yes/no) questions.
""",
    tools=[],
    handoff_description="Generate neutral everyday questions for psychological profiling."
)

# NEW: An agent to classify user's answer to a psych profile
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


def get_state(history):
    return tuple(history[-3:]) if history else ("start",)


async def main():
    print(
        "Welcome to the RL Psychological Profiling Agent with LLM-generated questions!\nLet's have a quick conversation.")
    history = []
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
        # Auto-evaluate user's answer using classifier LLM
        classify_prompt = f"User answer: {answer}\n\nWhich of these traits does this most reflect? Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism."
        with trace("LLM answer classification"):
            classification_result = await Runner.run(llm_classifier, classify_prompt)
            classified_profile = classification_result.final_output.strip().split('\n')[0]
        # Assign reward based on match
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


if __name__ == "__main__":
    asyncio.run(main())
