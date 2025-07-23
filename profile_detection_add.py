import pickle
import os
from agents import Agent, Runner, trace
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PSYCH_PROFILES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

class PolicyReader:
    def __init__(self, profiles, policy_file="psych_policy.pkl"):
        self.profiles = profiles
        self.policy_file = policy_file
        self.q_table = self.load()

    def load(self):
        try:
            with open(self.policy_file, "rb") as f:
                qdict = pickle.load(f)
            return qdict
        except Exception as e:
            print(f"[ERROR] Could not load RL policy: {e}")
            return {}

    def best_profile(self, state=("start",)):
        # Find which profile has the max Q-value for given state (default: start)
        best_idx = None
        best_val = float("-inf")
        for idx, profile in enumerate(self.profiles):
            q = self.q_table.get((state, idx), 0)
            if q > best_val:
                best_val = q
                best_idx = idx
        return self.profiles[best_idx] if best_idx is not None else "Openness"

# LLM agent for ad generation
gen_ad_agent = Agent(
    name="LLM Home Ad Generator",
    instructions="""
You are a creative real estate copywriter. Write a short (max 100 words) advertisement for a home, personalized for a person whose dominant personality trait is: {profile}.
- Highlight features that match this trait and would attract someone with this personality.
- Make it engaging, positive, and lifestyle-focused.
- Do not mention the trait or personality directly in the text.
""",
    tools=[],
    handoff_description="Write personalized real estate ads based on dominant personality trait."
)

import asyncio

def get_state_from_user():
    # In a real app, use saved last state. For demo, use start state
    return ("start",)

async def main():
    reader = PolicyReader(PSYCH_PROFILES)
    state = get_state_from_user()
    best_profile = reader.best_profile(state)
    print(f"[INFO] Detected dominant profile: {best_profile}")
    prompt = f"Write a real estate ad for a home, tailored for someone whose dominant trait is {best_profile}."
    with trace("LLM home ad generation"):
        ad_result = await Runner.run(gen_ad_agent, prompt.format(profile=best_profile))
        ad_text = ad_result.final_output.strip().split('\n')[0]
    print("\n[Personalized Home Advertisement]\n")
    print(ad_text)

if __name__ == "__main__":
    asyncio.run(main())
